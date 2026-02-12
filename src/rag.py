import os
import re
import json
import psycopg2
from sqlalchemy import make_url
from loguru import logger
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

from src.misc import unix_timestamp_to_timestamp


class PGVectorStoreRagAgent:
    """Custom Postgres Vector Store RAG that uses SQLAlchemy for connection management."""

    def __init__(
        self,
        logger: logger,
        conn_kwargs: dict,
        emb_model_name: str = "BAAI/bge-large-en-v1.5",
        emb_dim: int = 1024,
        llm_model_name: str = "qwen3-4B-instruct-2507",
        llm_api_base: str = "http://localhost:1234/v1",  # LlamaCPP server URL
        llm_api_key: str = None,
        llm_request_timeout: float = 120.0,
        llm_temperature: float = 0.4,
        llm_system_prompt: str = "You are a helpful assistant that can search through documents to answer questions.",
        hnsw_kwargs: dict = {
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
        hybrid_search: bool = True,
        text_search_config="english",
        tools: list = [],
    ):
        """Initializes the PostgresVectorStore with connection parameters and vector store settings."""
        self.logger = logger

        self.connection_str = (
            f"postgresql://{conn_kwargs['user']}:"
            + f"{conn_kwargs['password']}@{conn_kwargs['host']}:"
            + f"{conn_kwargs['port']}/{conn_kwargs['database']}"
        )
        self.connection_url = make_url(
            self.connection_str
        )  # Validate connection string
        self.conn = psycopg2.connect(self.connection_str)
        self.db_name = conn_kwargs.get("database")
        self.table_name = conn_kwargs.get("table")
        self.conn.autocommit = True  # Ensure autocommit is enabled for DDL operations

        # RAG Application settings
        if emb_model_name:
            Settings.embed_model = HuggingFaceEmbedding(model_name=emb_model_name)

        if llm_model_name and llm_api_base:
            Settings.llm = OpenAILike(
                model=llm_model_name,
                api_key=llm_api_key if llm_api_key else "placeholder_api_key",
                api_base=llm_api_base,
                is_chat_model=True,
                is_function_calling_model=True,
                temperature=llm_temperature,
                request_timeout=llm_request_timeout,
            )

        # Components for RAG pipeline
        self.vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=self.connection_url.host,
            password=self.connection_url.password,
            port=self.connection_url.port,
            user=self.connection_url.username,
            table_name=self.table_name,
            embed_dim=emb_dim,
            hnsw_kwargs=hnsw_kwargs,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # The Agent itself
        self.agent = AgentWorkflow.from_tools_or_functions(
            [self.search_documents] + tools,
            llm=Settings.llm,
            system_prompt=llm_system_prompt,
        )
        self.ctx = Context(self.agent)

    def clean_markdown(self, md_text: str) -> str:
        """Remove code blocks and excessive whitespace from markdown text."""
        text = re.sub(r"```[\s\S]*?```", "", md_text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple punctuation-based splitting."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def extract_main_content(self, md_text: str) -> str:
        """Extracts the main content from markdown, starting from 'Introduction'
        and ending before 'Appendix' or 'References'.
        """
        import re

        # Find the Introduction section header - handle various formats like "## Introduction" or "## 1 Introduction"
        intro_match = re.search(
            r"^##\s*(?:\d+\.?\s+)?Introduction\b", md_text, re.IGNORECASE | re.MULTILINE
        )
        if not intro_match:
            return md_text  # If no introduction, return whole text

        start_idx = intro_match.start()

        # Find the next section header that is either References or Appendix
        end_match = re.search(
            r"^##\s*(?:\d+\.?\s+)?(References|Appendix)\b",
            md_text,
            re.IGNORECASE | re.MULTILINE,
        )
        end_idx = end_match.start() if end_match else len(md_text)

        # Extract everything from Introduction to References/Appendix (including Introduction header)
        main_content = md_text[start_idx:end_idx].strip()
        return main_content

    def extract_metadata(self, filename: str) -> dict:
        """Extract metadata from a file.

        Returns a dictionary with the following keys:
        - filename: Name of the file
        - size: Size of the file in bytes
        - created_at: Creation time of the file in standard timestamp format
        - type: File extension (without the dot)
        """
        metadata = {}
        metadata["filename"] = filename
        metadata["size"] = os.path.getsize(filename)  # Size in bytes
        metadata["created_at"] = unix_timestamp_to_timestamp(
            os.path.getctime(filename)
        )  # Creation time
        metadata["type"] = os.path.splitext(filename)[1][
            1:
        ]  # Get file extension without dot
        return metadata

    def parse_documents(self, path: str) -> list[dict]:
        """Parse documents from the specified data directory."""
        docs = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    with open(json_path, "r", encoding="utf-8") as f:
                        paper_data = json.load(f)

                    markdown_path = json_path.replace(".json", ".md")
                    if os.path.exists(markdown_path):
                        with open(markdown_path, "r", encoding="utf-8") as md_file:
                            markdown_content = md_file.read()
                            # Attach the markdown content to the JSON data
                            paper_data["text"] = markdown_content

                    metadata = self.extract_metadata(json_path)
                    paper_data.update(metadata)

                    docs.append(paper_data)

        return docs

    def convert_raw_documents(self, raw_docs: list[dict]) -> list[Document]:
        """Convert raw document dicts into Document objects with text and metadata."""
        documents = []
        for d in raw_docs:
            main_md = self.extract_main_content(d["text"])
            text = self.clean_markdown(main_md)
            metadata = {k: v for k, v in d.items() if k != "text"}
            document = Document(
                text=text,
                metadata=metadata,
                excluded_llm_metadata_keys=[
                    "id",
                    "link",
                    "published",
                    "updated",
                    "primary_category",
                    "categories",
                    "filename",
                    "size",
                    "created_at",
                    "type",
                    "header_path",
                    "_node_type",
                    "document_id",
                    "doc_id",
                    "ref_doc_id",
                ],
            )
            documents.append(document)
        return documents

    def run_ingestion_pipeline(
        self,
        documents_path: str,
    ):
        raw_docs = self.parse_documents(documents_path)
        self.logger.info(f"Parsed {len(raw_docs)} raw documents from {documents_path}.")
        documents = self.convert_raw_documents(raw_docs)
        self.logger.info(
            f"Converted raw documents into {len(documents)} Document objects."
        )

        pipeline = IngestionPipeline(
            transformations=[
                MarkdownNodeParser(),
                QuestionsAnsweredExtractor(questions=2),
                KeywordExtractor(keywords=5),
            ],
            vector_store=self.vector_store,
        )
        nodes = pipeline.run(documents=documents[:2])
        self.logger.info(
            f"Transformed documents into {len(nodes)} nodes after splitting."
        )

        _ = VectorStoreIndex(nodes, storage_context=self.storage_context)
        self.logger.info("Index created from documents successfully.")

    def search_documents(
        self,
        query: str,
        hybrid_search: bool,
        sparse_top_k: int,
        similarity_top_k: int,
        verbose: bool = False,
        truncate: int = -1,
    ) -> str:
        """Useful for answering natural language questions from text documents."""
        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.logger.info("Index loaded from vector store successfully.")

        if hybrid_search:
            query_engine = index.as_query_engine(
                vector_store_query_mode="hybrid", sparse_top_k=sparse_top_k
            )
            vector_retriever = index.as_retriever(
                vector_store_query_mode="default",
                similarity_top_k=similarity_top_k,
            )
            text_retriever = index.as_retriever(
                vector_store_query_mode="sparse",
                similarity_top_k=similarity_top_k,
            )
            retriever = QueryFusionRetriever(
                [vector_retriever, text_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=1,  # set this to 1 to disable query generation
                mode="relative_score",
                use_async=True,
            )
        else:
            # Use vector-only search
            retriever = index.as_retriever(
                vector_store_query_mode="default",
                similarity_top_k=similarity_top_k,
            )

        response_synthesizer = CompactAndRefine()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        nodes = retriever.retrieve(query)

        if verbose:
            self.logger.info(f"Retrieved {len(nodes)} nodes for query: {query}")
            for i, node in enumerate(nodes):
                self.logger.info(
                    f"Node {i+1} {node.id_} (score: {node.score:.4f}): {node.text[:truncate]}..."
                )

        response = query_engine.query(query)
        if verbose:
            self.logger.info(f"Response from query engine: {response}")

        return str(response)
