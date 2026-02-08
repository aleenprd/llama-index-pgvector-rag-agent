import os
import sys
import asyncio
import argparse
import nest_asyncio
from loguru import logger
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

from src.rag.rag import PGVectorStoreRagAgent

load_dotenv()  # Load environment variables from .env file
nest_asyncio.apply()  # Apply nest_asyncio to allow nested event loops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postgres Vector Store RAG Agent")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
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
        "--log_level",
        type=lambda x: str(x).upper(),
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        required=False,
    )

    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logger.remove()  # Remove default logger
    logger.add(sink=sys.stdout, level=log_level)  # Set new logging level


async def main():
    args = parse_args()
    setup_logging(log_level=args.log_level)
    

    documents = SimpleDirectoryReader(args.data_dir).load_data()
    logger.info("Document ID:", documents[0].doc_id)
    
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
        emb_dim=1024,
        llm_model_name=args.llm_model_name,
        llm_api_base="http://localhost:8000/",
        llm_api_key=None,
        llm_request_timeout=30.0,
        llm_temperature=0.1,
        similarity_top_k=5,
        chunk_size=1024,
        chunk_overlap=50,
        hybrid_search=True,
        full_refresh=True,
    )


    # # https://developers.llamaindex.ai/python/examples/vector_stores/postgres/
    # with rag_agent.conn.cursor() as c:
    #     c.execute(f"DROP TABLE IF EXISTS {rag_agent.table_name};")
    #     #c.execute(f"SELECT 'CREATE DATABASE {rag_agent.db_name}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{rag_agent.db_name}')")

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