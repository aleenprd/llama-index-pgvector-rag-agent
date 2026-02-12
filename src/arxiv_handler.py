import os
from pathlib import Path
import json
import requests
import feedparser
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger
from docling.document_converter import DocumentConverter  # Bit heavy (but good);


@dataclass
class ArxivPaper:
    """Data class representing an arXiv paper."""

    id: str
    link: str
    published: str
    updated: str
    title: str
    authors: List[str]
    abstract: str
    primary_category: str
    categories: List[str]


class ArxivHandler:
    """Handler for interacting with the arXiv API."""

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.converter = DocumentConverter()

    def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a single paper by its arXiv ID.

        Args:
            arxiv_id: The arXiv ID (e.g., "2504.11651v1" or "2504.11651")

        Returns:
            ArxivPaper object or None if not found
        """
        url = f"{self.base_url}?id_list={arxiv_id}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            feed = feedparser.parse(response.text)

            if not feed.entries:
                return None

            entry = feed.entries[0]

            return ArxivPaper(
                id=entry.id,
                link=entry.link,
                published=entry.published,
                updated=entry.updated,
                title=entry.title.strip(),
                authors=[author.name for author in entry.authors],
                abstract=entry.summary.strip(),
                primary_category=entry.arxiv_primary_category["term"],
                categories=[tag.term for tag in entry.tags],
            )

        except requests.RequestException as e:
            logger.debug(f"Error fetching paper {arxiv_id}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error parsing paper {arxiv_id}: {e}")
            return None

    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """
        Search for papers using arXiv API query.

        Args:
            query: Search query (e.g., "machine learning", "cat:cs.LG", "au:Smith")
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort criteria ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending", "descending")

        Returns:
            List of ArxivPaper objects
        """
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            papers = []

            for entry in feed.entries:
                paper = ArxivPaper(
                    id=entry.id,
                    link=entry.link,
                    published=entry.published,
                    updated=entry.updated,
                    title=entry.title.strip(),
                    authors=[author.name for author in entry.authors],
                    abstract=entry.summary.strip(),
                    primary_category=entry.arxiv_primary_category["term"],
                    categories=[tag.term for tag in entry.tags],
                )
                papers.append(paper)

            return papers

        except requests.RequestException as e:
            logger.debug(f"Error searching papers: {e}")
            return []
        except Exception as e:
            logger.debug(f"Error parsing search results: {e}")
            return []

    def get_recent_papers(
        self, category: str = "cs.LG", max_results: int = 10
    ) -> List[ArxivPaper]:
        """
        Get recent papers from a specific category.

        Args:
            category: arXiv category (e.g., "cs.LG", "cs.AI", "stat.ML")
            max_results: Maximum number of results

        Returns:
            List of recent ArxivPaper objects
        """
        query = f"cat:{category}"
        return self.search_papers(
            query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
        )

    def download_paper(self, url: str, formats: list = ["html", "pdf"], dir: str = "arxiv") -> str:
        """Download a paper from its arXiv URL.

        Args:
            url: Full arXiv URL (e.g., "https://arxiv.org/abs/2504.11651")
            formats: List of formats to download (e.g., ["html", "pdf"])
            dir: Directory to save the downloaded files

        Raises:
            ValueError: If the URL is not a valid arXiv URL.

        Returns:
            str: The arXiv paper ID.
        """
        if not formats:
            raise ValueError("At least one format must be specified for download")
        
        if not url.startswith("https://arxiv.org/"):
            raise ValueError("Invalid arXiv URL")

        paper_id = url.split("/")[-1]
        logger.debug(f"Fetching paper with ID: {paper_id}")
        paper = self.get_paper_by_id(paper_id)

        # Create a sub-directory for this paper
        paper_dir = f"{dir}/{paper_id}"
        if not os.path.exists(paper_dir):
            os.makedirs(paper_dir)

        if paper and paper.link:
            # Save the paper metadata as JSON
            json_path = f"{paper_dir}/{paper_id}.json"
            with open(json_path, "w") as f:
                json.dump(paper.__dict__, f, indent=4)
            logger.debug(f"Saved paper {paper.id} to {json_path}")

            # Download the PDF file
            if "pdf" in formats:
                pdf_path = f"{paper_dir}/{paper_id}.pdf"
                pdf_download_url = paper.link.replace("abs", "pdf")
                response = requests.get(pdf_download_url, stream=True)

                if response.status_code == 200:
                    with open(pdf_path, "wb") as pdf_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            pdf_file.write(chunk)
                    logger.debug(f"Downloaded PDF for {paper_id} to {pdf_path}")
                else:
                    logger.debug(
                        f"Failed to download PDF for {paper_id}: HTTP {response.status_code}"
                    )

            # Download the HTML
            if "html" in formats:
                html_path = f"{paper_dir}/{paper_id}.html"
                html_download_url = paper.link.replace("abs", "html")
                response = requests.get(html_download_url, stream=True)

                if response.status_code == 200:
                    with open(html_path, "wb") as html_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            html_file.write(chunk)
                    logger.debug(f"Downloaded HTML for {paper_id} to {html_path}")
                else:
                    logger.debug(
                        f"Failed to download HTML for {paper_id}: HTTP {response.status_code}"
                    )

        return paper_id

    def convert_document_to_markdown(self, file_path: str) -> Optional[str]:
        """Convert a document (PDF or HTML) to Markdown format.

        Args:
            file_path: Path to the input document (PDF or HTML).

        Returns:
            str: The converted Markdown content, or None if conversion fails.
        """
        try:
            md_content = self.converter.convert(file_path).document.export_to_markdown()
            return md_content
        except Exception as e:
            logger.debug(f"Error converting {file_path} to Markdown: {e}")
            return None

    def load_arxiv_document(self, id: str, dir: str = "arxiv") -> dict:
        """Load a single arXiv document by its ID.

        Args:
            id: The arXiv paper ID (e.g., "2504.11651").
            dir: Directory containing the arXiv JSON files.

        Raises:
            FileNotFoundError: If the document files do not exist.
            ValueError: If the document is not found in the specified directory.

        Returns:
            dict: The document metadata and markdown content.
        """
        paper_dir = Path(dir) / id
        json_path = paper_dir / f"{id}.json"
        md_path = paper_dir / f"{id}.md"

        if not json_path.exists() or not md_path.exists():
            logger.debug(f"Document {id} not found in {dir}")
            raise FileNotFoundError(f"Document {id} not found in {dir}")

        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        with open(md_path, "r", encoding="utf-8") as f:
            doc["markdown_document"] = f.read()

        return doc

    def load_arxiv_documents(self, dir: str) -> List[dict]:
        """Parse the arxiv directory and load all JSON documents.

        Args:
            dir: Directory containing arxiv JSON files.

        Returns:
            List of dictionaries containing paper metadata and markdown content.
        """
        documents = []
        arxiv_path = Path(dir)

        # Get all subdirectories in arxiv folder
        for subdir in arxiv_path.iterdir():
            if subdir.is_dir():
                # Look for JSON file in the subdirectory
                json_files = list(subdir.glob("*.json"))
                if json_files:
                    json_file = json_files[0]  # Take the first JSON file found
                    logger.debug(f"Processing: {json_file}")

                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            doc = json.load(f)

                        # Check if markdown file exists and load it
                        markdown_files = list(subdir.glob("*.md"))
                        if markdown_files:
                            markdown_file = markdown_files[0]
                            with open(markdown_file, "r", encoding="utf-8") as f:
                                doc["markdown_document"] = f.read()
                        else:
                            doc["markdown_document"] = ""

                        documents.append(doc)

                    except Exception as e:
                        logger.error(f"Error processing {json_file}: {e}")
                        continue

        return documents
