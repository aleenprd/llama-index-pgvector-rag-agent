#!/usr/bin/env python3
"""Convert PDF to Markdown using ArxivHandler."""

from pathlib import Path
from src.arxiv_handler import ArxivHandler
from loguru import logger

def main():
    # Initialize the handler
    handler = ArxivHandler()
    
    # Input PDF path
    pdf_path = "/home/aleen/Desktop/repos/llama-index-pgvector-rag-agent/docs/tmp/language-models-are-injective-and-hence-invertible.pdf"
    
    # Convert to markdown
    logger.info(f"Converting {pdf_path} to markdown...")
    markdown_content = handler.convert_document_to_markdown(pdf_path)
    
    if markdown_content:
        # Save the markdown
        output_path = Path(pdf_path).with_suffix('.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Markdown saved to {output_path}")
        print(f"Successfully converted PDF to markdown: {output_path}")
    else:
        logger.error("Failed to convert PDF to markdown")
        print("Error: Conversion failed")

if __name__ == "__main__":
    main()
