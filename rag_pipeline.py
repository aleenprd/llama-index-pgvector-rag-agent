import os
import json
from dotenv import load_dotenv
from utils.psql_handler import PSQLHandler
from utils.rag_handler import RAGHandler
from loguru import logger

load_dotenv()

ARXIV_DIR = "arxiv"
OUTPUT_DIR = "arxiv_processed"


psql_handler = PSQLHandler(
    dbname=os.getenv("TGB_POSTGRES_DB", "dev"),
    user=os.getenv("TGB_POSTGRES_USER", "dev"),
    password=os.getenv("TGB_POSTGRES_PASSWORD", "dev"),
    host=os.getenv("TGB_POSTGRES_HOST", "localhost"),
    port=os.getenv("TGB_POSTGRES_PORT", "5432"),
)
rag_handler = RAGHandler(psql_handler=psql_handler)


def main():
    # Create the papers table if it doesn't exist
    # logger.info("Creating papers table if it doesn't exist...")
    # psql_handler.create_papers_table()
    # logger.info("Papers table created successfully.")

    # Create the passages table if it doesn't exist
    logger.info("Creating passages table if it doesn't exist...")
    psql_handler.create_passages_table()
    logger.info("Passages table created successfully.")

    # Embed and insert sample documents
    # logger.info("\n" + "=" * 50)
    # logger.info("Embedding and inserting sample documents")
    # docs = []
    # for root, _, files in os.walk(ARXIV_DIR):
    #     for file in files:
    #         if file.endswith(".json"):
    #             with open(os.path.join(root, file), "r", encoding="utf-8") as f:
    #                 docs.append(json.load(f))

    # logger.info(f"Found {len(docs)} documents to embed.")
    # if not docs:
    #     logger.warning("No documents found to embed. Exiting.")
    #     return
    # docs = rag_handler.embed_paper_docs(docs)
    # logger.info(f"Embedded {len(docs)} documents.")

    # # Insert documents into the database
    # logger.info("Inserting documents into the database...")
    # psql_handler.insert_paper_documents(docs)

    # Create the embeddings for passages
    logger.info("Creating embeddings for passages...")
    docs = []
    for root, _, files in os.walk(ARXIV_DIR):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                with open(json_path, "r", encoding="utf-8") as f:
                    paper_data = json.load(f)

                # Find the corresponding markdown file
                markdown_path = json_path.replace(".json", ".md")
                if os.path.exists(markdown_path):
                    with open(markdown_path, "r", encoding="utf-8") as md_file:
                        markdown_content = md_file.read()
                        # Attach the markdown content to the JSON data
                        paper_data["markdown"] = markdown_content

                docs.append(paper_data)

    docs = rag_handler.embed_passage_docs(docs)
    logger.info(f"Created embeddings for {len(docs)} passages.")

    # Insert passages into the database
    logger.info("Inserting passages into the database...")
    psql_handler.insert_passage_documents(docs)
    logger.info("Passages inserted successfully.")

    # Test the search functionality
    logger.info("\n" + "=" * 50)
    logger.info("Testing search functionality")
    logger.info("=" * 50)

    # query = input("Enter your search query: ")
    # query_embedding = rag_handler.embed_query(query)

    # n = 3
    # results = psql_handler.run_query(
    #     query=f"""
    #     SELECT id, title, abstract, 1 - (embedding <=> %s::vector) AS similarity
    #     FROM papers
    #     ORDER BY embedding <=> %s::vector
    #     LIMIT {n};
    #     """,
    #     params=(query_embedding, query_embedding),
    #     fetch=True,
    # )
    # logger.info(f"\nTop {len(results)} results for query: '{query}'")
    # logger.info("=" * 80)
    # for i, row in enumerate(results, 1):
    #     logger.info(f"{i}. Similarity: {row[3]:.4f}")
    #     logger.info(f"   ID: {row[0]}")
    #     logger.info(f"   Title: {row[1]}")
    #     logger.info(f"   Abstract: {row[2][:200]}...")
    #     logger.info("-" * 80)

    logger.info("\nRAG pipeline completed successfully!")


if __name__ == "__main__":
    main()
