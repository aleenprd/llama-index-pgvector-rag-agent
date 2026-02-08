import psycopg2
from sqlalchemy import make_url
from loguru import logger
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context


class PGVectorStoreRagAgent():
    """Custom Postgres Vector Store RAG that uses SQLAlchemy for connection management."""

    def __init__(
        self,
        logger: logger,
        documents: SimpleDirectoryReader,
        conn_kwargs: dict,
        emb_model_name: str = "BAAI/bge-large-en-v1.5",
        emb_dim: int = 1024,
        llm_model_name: str = "qwen3-4B-instruct-2507",
        llm_api_base: str = "http://localhost:1234/v1",  # LlamaCPP server URL
        llm_api_key: str = None,
        llm_request_timeout: float = 120.0,
        llm_temperature: float = 0.4,
        llm_system_prompt: str = "You are a helpful assistant that can search through documents to answer questions.",
        similarity_top_k=5,
        sparse_top_k=12,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        hnsw_kwargs: dict = {
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
        hybrid_search: bool = True,
        text_search_config="english",
        tools: list = [],
        full_refresh: bool = False,
    ):
        """Initializes the PostgresVectorStore with connection parameters and vector store settings."""
        self.logger = logger
        
        self.connection_str = (
            f"postgresql://{conn_kwargs['user']}:"
            + f"{conn_kwargs['password']}@{conn_kwargs['host']}:"
            + f"{conn_kwargs['port']}/{conn_kwargs['database']}"
        )
        self.connection_url = make_url(self.connection_str)  # Validate connection string
        self.conn = psycopg2.connect(self.connection_str)
        self.db_name = conn_kwargs.get("database")
        self.table_name = conn_kwargs.get("table")
        self.conn.autocommit = True  # Ensure autocommit is enabled for DDL operations
        self.documents = documents

        if full_refresh:
            with self.conn.cursor() as c:
                try:
                    c.execute(f"DROP TABLE IF EXISTS {self.table_name};")
                except Exception as e:
                    raise Exception("Failed to drop existing vector store table for full refresh.")
                    
        # RAG Application settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=emb_model_name)
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

        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                TitleExtractor(),
                OpenAIEmbedding(),
            ],
            vector_store=self.vector_store,
        )
        
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        if full_refresh:
            self.index = VectorStoreIndex.from_documents(
                self.documents, storage_context=self.storage_context, show_progress=True
            )
            self.logger.info("Index created from documents successfully.")
        else:
            self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
            self.logger.info("Index loaded from vector store successfully.")

        # Configure query engine based on hybrid_search setting
        if hybrid_search:
            self.query_engine = self.index.as_query_engine(
                vector_store_query_mode="hybrid", sparse_top_k=sparse_top_k
            )
            self.vector_retriever = self.index.as_retriever(
                vector_store_query_mode="default",
                similarity_top_k=similarity_top_k,
            )
            self.text_retriever = self.index.as_retriever(
                vector_store_query_mode="sparse",
                similarity_top_k=similarity_top_k,
            )
            self.retriever = QueryFusionRetriever(
                [self.vector_retriever, self.text_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=1,  # set this to 1 to disable query generation
                mode="relative_score",
                use_async=True,
            )
        else:
            # Use vector-only search
            self.retriever = self.index.as_retriever(
                vector_store_query_mode="default",
                similarity_top_k=similarity_top_k,
            )

        self.response_synthesizer = CompactAndRefine()
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
        )
        
        # The Agennt itself
        self.agent = AgentWorkflow.from_tools_or_functions(
            [self.search_documents] + tools,
            llm=Settings.llm,
            system_prompt=llm_system_prompt,
        )
        # self.ctx = Context(self.agent)

    async def search_documents(self, query: str, verbose: bool = False, truncate: int = -1) -> str:
        """Useful for answering natural language questions from text documents."""
        nodes = await self.retriever.aretrieve(query)
        
        if verbose:
            self.logger.info(f"Retrieved {len(nodes)} nodes for query: {query}")
            for i, node in enumerate(nodes):
                self.logger.info(f"Node {i+1} {node.id_} (score: {node.score:.4f}): {node.text[:truncate]}...")
        
        response = await self.query_engine.aquery(query)
        
        if verbose:
            self.logger.info(f"Response from query engine: {response}")
            
        return str(response)
