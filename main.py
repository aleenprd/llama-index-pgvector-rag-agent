import os
import asyncio
import argparse
import nest_asyncio
from loguru import logger
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

from src.rag import PGVectorStoreRagAgent
from src.misc import (
    setup_logging,
)

load_dotenv()  # Load environment variables from .env file
nest_asyncio.apply()  # Apply nest_asyncio to allow nested event loops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postgres Vector Store RAG Agent")
    parser.add_argument(
        "--full_refresh",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        help="Whether to perform a full refresh of the vector store (drops and recreates the table)",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/examples",
        help="Directory containing documents to index",
        required=False,
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="qwen3-4B-instruct-2507",
        help="Name of the LMStudio model to use",
        required=False,
    )
    parser.add_argument(
        "--embeddings_model_name",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Name of the HuggingFace embeddings model to use",
        required=False,
    )
    parser.add_argument(
        "--embeddings_dim",
        type=int,
        default=1024,
        help="Dimension of the embeddings model",
        required=False,
    )
    parser.add_argument(
        "--llm_api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the LLM API",
        required=False,
    )
    parser.add_argument(
        "--llm_request_timeout",
        type=float,
        default=120.0,
        help="Request timeout for LLM API calls in seconds",
        required=False,
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM generation",
        required=False,
    )
    parser.add_argument(
        "--similarity_top_k",
        type=int,
        default=5,
        help="Number of top similar documents to retrieve",
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Size of text chunks for splitting documents",
        required=False,
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Overlap between text chunks",
        required=False,
    )
    parser.add_argument(
        "--hybrid_search",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Whether to use hybrid search (vector + keyword)",
        required=False,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: str(x).upper(),
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        required=False,
    )

    return parser.parse_args()


async def main():
    args = parse_args()
    setup_logging(logger, log_level=args.log_level)
    
    documents = None
    if args.full_refresh:
        logger.info("Performing full refresh: dropping and recreating the vector store table.")
        
        documents = SimpleDirectoryReader(args.data_dir).load_data()
        if documents:
            logger.info(f"Documents loaded successfully: {len(documents)} documents found.")
    
    rag_agent = PGVectorStoreRagAgent(
        logger=logger,
        documents=documents,
        conn_kwargs={
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": int(os.getenv("POSTGRES_PORT")),
            "database": os.getenv("POSTGRES_DB"),
            "table": os.getenv("POSTGRES_TABLE"),
        },
        emb_model_name=args.embeddings_model_name,
        emb_dim=args.embeddings_dim,
        llm_model_name=args.llm_model_name,
        llm_api_base=args.llm_api_base,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_request_timeout=args.llm_request_timeout,
        llm_temperature=args.llm_temperature,
        similarity_top_k=args.similarity_top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        hybrid_search=args.hybrid_search,
        full_refresh=args.full_refresh,
    )

    # Test different queries
    queries = [
        "What was being taught in the florence academy?",
    ]
    
    for query in queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*60}")
        await rag_agent.search_documents(query, verbose=True, truncate=500)


if __name__ == "__main__":
    asyncio.run(main())