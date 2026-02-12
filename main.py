import os
import json
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

# Default configuration constants
DEFAULT_POSTGRES_TABLE = "llama_index_embeddings"
MIN_POSTGRES_PORT = 1
MAX_POSTGRES_PORT = 65535


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
        default="data/arxiv",
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
        "--sparse_top_k",
        type=int,
        default=5,
        help="Number of top keyword-matched documents to retrieve (if hybrid search is enabled)",
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


def main():
    args = parse_args()
    setup_logging(logger, log_level=args.log_level)
    
    # Validate required Postgres environment variables
    # Note: We use explicit None and empty string checks instead of 'if not value'
    # to handle edge cases where '0' or 'false' strings are valid values
    postgres_table = os.getenv("POSTGRES_TABLE")
    if postgres_table is None or postgres_table == "":
        postgres_table = DEFAULT_POSTGRES_TABLE
        logger.warning(
            f"POSTGRES_TABLE not set, using default: '{postgres_table}'. "
            f"Set POSTGRES_TABLE in .env to customize the table name."
        )
    
    required_env_vars = {
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB"),
    }
    
    # Explicit check for None and empty string to allow values like '0' or 'false'
    missing_vars = [var for var, value in required_env_vars.items() if value is None or value == ""]
    if missing_vars:
        error_msg = (
            f"Missing required environment variable(s): {', '.join(missing_vars)}. "
            f"Please set them in your .env file or environment. "
            f"See .env.example for reference."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert and validate POSTGRES_PORT
    try:
        postgres_port = int(required_env_vars["POSTGRES_PORT"])
    except ValueError as e:
        error_msg = f"Invalid POSTGRES_PORT value '{required_env_vars['POSTGRES_PORT']}': must be a valid integer"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    # Validate port range
    if not (MIN_POSTGRES_PORT <= postgres_port <= MAX_POSTGRES_PORT):
        error_msg = f"Invalid POSTGRES_PORT value {postgres_port}: must be between {MIN_POSTGRES_PORT} and {MAX_POSTGRES_PORT}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    documents = None
    if args.full_refresh:
        logger.info("Performing full refresh: dropping and recreating the vector store table.")
        
        documents = SimpleDirectoryReader(args.data_dir).load_data()
        if documents:
            logger.info(f"Documents loaded successfully: {len(documents)} documents found.")
    
    rag_agent = PGVectorStoreRagAgent(
        logger=logger,
        conn_kwargs={
            "user": required_env_vars["POSTGRES_USER"],
            "password": required_env_vars["POSTGRES_PASSWORD"],
            "host": required_env_vars["POSTGRES_HOST"],
            "port": postgres_port,
            "database": required_env_vars["POSTGRES_DB"],
            "table": postgres_table,
        },
        emb_model_name=args.embeddings_model_name,
        emb_dim=args.embeddings_dim,
        llm_model_name=args.llm_model_name,
        llm_api_base=args.llm_api_base,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_request_timeout=args.llm_request_timeout,
        llm_temperature=args.llm_temperature,
        hybrid_search=args.hybrid_search,
    )
    
    if args.full_refresh:
        logger.info(
            "Performing full refresh: dropping and recreating the vector store table."
        )
        with rag_agent.conn.cursor() as c:
            try:
                c.execute(f"DROP TABLE IF EXISTS {os.getenv('POSTGRES_TABLE')};")
            except Exception as e:
                raise Exception(
                    "Failed to drop existing vector store table for full refresh."
                )

        rag_agent.run_ingestion_pipeline(args.data_dir)

    while True:
        try:
            query = input("\nEnter a query (or 'exit' to quit): ")
            if query.lower() == "exit":
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"Query: {query}")
            logger.info(f"{'='*60}")
            rag_agent.search_documents(
                query,
                hybrid_search=args.hybrid_search,
                sparse_top_k=args.sparse_top_k,
                similarity_top_k=args.similarity_top_k,
                verbose=True,
                truncate=500,
            )
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue
        
    logger.info("Done querying. Goodbye!")


if __name__ == "__main__":
    # asyncio.run(main())
    main()
