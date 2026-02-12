import re
from typing import Union
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer
from utils.psql_handler import PSQLHandler
import hashlib
import uuid


# ----------------------------
# CONFIG
# ----------------------------
PASSAGE_INDEX_MODEL = "BAAI/bge-m3"
MAX_TOKENS = 1200  # max tokens per chunk (800-1200 is common for BGE models)
MIN_CHARS = 50  # optional: skip tiny chunks (50-100 chars)
CHUNK_OVERLAP = 180  # optional: allow overlapping context (15-20% of max tokens)
PASSAGE_N_DIMS = 1024  # Number of dimensions for the passage embeddings

PAPER_INDEX_MODEL = "allenai/specter2"
PAPER_N_DIMS = 768  # Number of dimensions for the paper embeddings


# ----------------------------
# FUNCTION: CLEAN AND SPLIT MARKDOWN
# ----------------------------
def clean_markdown(md_text):
    # Remove code blocks and excessive whitespace
    text = re.sub(r"```[\s\S]*?```", "", md_text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_into_sentences(text):
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def extract_main_content(md_text: str) -> str:
    """
    Extracts the main content from markdown, starting from 'Introduction'
    and ending before 'Appendix' or 'References'.
    """
    import re
    # Find the Introduction section header - handle various formats like "## Introduction" or "## 1 Introduction"
    intro_match = re.search(r"^##\s*(?:\d+\.?\s+)?Introduction\b", md_text, re.IGNORECASE | re.MULTILINE)
    if not intro_match:
        return md_text  # If no introduction, return whole text

    start_idx = intro_match.start()

    # Find the next section header that is either References or Appendix
    end_match = re.search(r"^##\s*(?:\d+\.?\s+)?(References|Appendix)\b", md_text, re.IGNORECASE | re.MULTILINE)
    end_idx = end_match.start() if end_match else len(md_text)

    # Extract everything from Introduction to References/Appendix (including Introduction header)
    main_content = md_text[start_idx:end_idx].strip()
    return main_content


class RAGHandler:
    """Handler for Retrieval-Augmented Generation (RAG) tasks using a sentence transformer model."""

    def __init__(self, psql_handler: PSQLHandler):
        """Initialize the RAG handler with a sentence transformer model."""
        self.psql_handler = psql_handler
        self.paper_index_tokenizer = AutoTokenizer.from_pretrained(
            PAPER_INDEX_MODEL + "_base"
        )
        self.paper_index_model = AutoAdapterModel.from_pretrained(
            PAPER_INDEX_MODEL + "_base"
        )
        self.paper_index_model.load_adapter(
            PAPER_INDEX_MODEL, source="hf", load_as="specter2", set_active=True
        )
        self.passage_index_tokenizer = AutoTokenizer.from_pretrained(
            PASSAGE_INDEX_MODEL
        )
        self.passage_index_model = SentenceTransformer(PASSAGE_INDEX_MODEL)

    def embed_paper_docs(self, docs: list[dict]) -> list[list[float]]:
        """Create embeddings for a list of documents.

        This method expects a list of dictionaries, each containing at least 'title' and 'abstract' keys.
        It combines the title and abstract for each document, generates an embedding using the Specter2 model,
        and attaches the embedding to the document dictionary under the 'embedding' key.

        Args:
            docs (list[dict]): The documents to embed, each as a dictionary with 'title' and 'abstract'.

        Raises:
            ValueError: If the input is not a list, or if any document is missing 'title' or 'abstract'.

        Returns:
            list[dict]: The input list of document dictionaries, each with an added 'embedding' key.
        """
        if not isinstance(docs, list):
            raise ValueError("Input must be a list of documents.")

        if not isinstance(docs[0], dict):
            raise ValueError(
                "Document must be a dictionary with 'title' and 'abstract' keys."
            )

        if not all("title" in doc and "abstract" in doc for doc in docs):
            raise ValueError("Each document must contain 'title' and 'abstract' keys.")

        text_batch = [
            doc["title"]
            + self.paper_index_tokenizer.sep_token
            + (doc.get("abstract") or "")
            for doc in docs
        ]

        inputs = self.paper_index_tokenizer(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )

        output = self.paper_index_model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]

        # Add embeddings directly to each document
        for i, doc in enumerate(docs):
            doc["embedding"] = embeddings[i].cpu().detach().numpy().tolist()

        return docs

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a user query.

        Args:
            query (str): The user query string.

        Returns:
            list[float]: The embedding vector for the query.
        """
        inputs = self.paper_index_tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        output = self.paper_index_model(**inputs)
        embedding = output.last_hidden_state[:, 0, :]

        return embedding[0].cpu().detach().numpy().tolist()

    def search_paper_docs(self, query: str, n: int = 5) -> list[dict]:
        """Search for documents in the database based on a query, in order of relevance.

        Args:
            query (str): The search query.
            n (int): The number of top results to return.

        Returns:
            list[dict]: A list of documents matching the query.
        """
        query_embedding = self.embed_query(query)
        search_query = f"""
            SELECT id, title, abstract, 1 - (embedding <=> %s::vector) AS similarity
            FROM papers
            ORDER BY embedding <=> %s::vector
            LIMIT {n};
            """
        results = self.psql_handler.run_query(
            query=search_query, params=(query_embedding, query_embedding), fetch=True
        )
        return results

    def chunk_text(
        self, text: str, max_tokens: int = MAX_TOKENS, overlap: int = CHUNK_OVERLAP
    ) -> list[str]:
        """Chunk text into smaller pieces based on token count, allowing for overlap.

        Args:
            text (str): The text to chunk.
            max_tokens (int): The maximum number of tokens per chunk.
            overlap (int): The number of overlapping tokens between chunks.

        Returns:
            list[str]: A list of text chunks.
        """
        sentences = split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            token_count = len(self.passage_index_tokenizer(sent)["input_ids"])
            if current_len + token_count > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                    if overlap > 0:
                        overlap_tokens = []
                        while current_len > 0 and overlap_tokens:
                            overlap_tokens.append(current_chunk.pop())
                            current_len -= len(
                                self.passage_index_tokenizer(overlap_tokens[-1])[
                                    "input_ids"
                                ]
                            )
                current_chunk = []
                current_len = 0
            current_chunk.append(sent)
            current_len += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Filter out very short chunks
        chunks = [c for c in chunks if len(c) >= MIN_CHARS]
        return chunks

    def embed_passage_docs(self, docs: Union[str, list[str]]) -> list[dict]:
        """Chunk documents into smaller pieces and generate embeddings for each chunk.

        Args:
            docs (Union[str, list[str]]): A list of document dicts with 'markdown' key.

        Returns:
            list[dict]: Each dict contains 'chunk' and 'embedding' keys.
        """
        output = []
        for d in docs:
            # Pre-process markdown to extract main content
            main_md = extract_main_content(d["markdown"])
            text = clean_markdown(main_md)
            chunks = self.chunk_text(text)
            embeddings = self.passage_index_model.encode(chunks, convert_to_numpy=True)

            for chunk, emb in zip(chunks, embeddings):
                out = d.copy()
                chunk_hash = hashlib.md5(chunk.encode()).digest()
                out["chunk_id"] = str(uuid.UUID(bytes=chunk_hash[:16]))
                out["chunk"] = chunk
                out["embedding"] = emb.tolist()  # pgvector expects a list of floats
                out["tokens"] = len(self.passage_index_tokenizer(chunk)["input_ids"])
                output.append(out)
        return output
