"""Client for interacting with the arXiv API."""

import os
import time
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request

import arxiv

from ..config.settings import get_settings
from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentMetadata, FilePath

logger = get_logger(__name__)


class ArxivError(LangChainGPTError):
    """Error raised by arXiv operations."""
    
    def __init__(self, message: str = "arXiv error"):
        super().__init__(f"arXiv error: {message}")


class ArxivClient:
    """Client for searching and retrieving papers from arXiv."""
    
    def __init__(
        self,
        max_results: int = None,
        wait_time: int = None,
        output_dir: Optional[FilePath] = None,
    ):
        """Initialize arXiv client.
        
        Args:
            max_results: Maximum number of results per query
            wait_time: Wait time between requests in seconds (rate limiting)
            output_dir: Directory for saving downloaded papers
            
        Raises:
            ArxivError: If arXiv client initialization fails
        """
        settings = get_settings()
        self.max_results = max_results or settings.ARXIV_QUERY_LIMIT
        self.wait_time = wait_time or settings.ARXIV_WAIT_TIME
        self.output_dir = Path(output_dir) if output_dir else Path(settings.DATA_DIR) / "papers" / "arxiv"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client with connection pooling
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=1,
            num_retries=3,
        )
        
        logger.info("Initialized arXiv client")
    
    def search(
        self,
        query: str,
        max_results: int = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ) -> List[arxiv.Result]:
        """Search for papers on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort_by: Sort criterion (relevance, submittedDate, lastUpdatedDate)
            sort_order: Sort order (ascending, descending)
            
        Returns:
            List[arxiv.Result]: List of arXiv results
            
        Raises:
            ArxivError: If search fails
        """
        if not query:
            raise ArxivError("Search query cannot be empty")
        
        # Set default max results
        if max_results is None:
            max_results = self.max_results
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            
            # Execute search with rate limiting
            results = list(self.client.results(search))
            
            # Wait after query to avoid rate limiting
            time.sleep(self.wait_time)
            
            logger.info(f"Found {len(results)} papers matching query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"arXiv search failed: {str(e)}")
            raise ArxivError(f"Search failed for query '{query}': {str(e)}")
    
    def download_paper(
        self,
        result: arxiv.Result,
        output_dir: Optional[FilePath] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Download paper PDF from arXiv.
        
        Args:
            result: arXiv result object
            output_dir: Directory to save paper (overrides default)
            filename: Custom filename (default: uses paper ID)
            
        Returns:
            str: Path to downloaded PDF
            
        Raises:
            ArxivError: If download fails
        """
        # Use provided directory or default
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from arXiv ID if not provided
        if filename is None:
            # Extract numeric ID from URL
            arxiv_id = result.entry_id.split("/")[-1]
            # Remove version (e.g., v1, v2) if present
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            filename = f"arxiv_{arxiv_id}.pdf"
        
        # Ensure filename ends with .pdf
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        
        # Full path to output file
        output_path = output_dir / filename
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"Paper already downloaded: {output_path}")
            return str(output_path)
        
        try:
            # Get paper PDF URL
            pdf_url = result.pdf_url
            
            # Download PDF
            logger.info(f"Downloading paper from {pdf_url}")
            urllib.request.urlretrieve(pdf_url, output_path)
            
            # Wait after download to avoid rate limiting
            time.sleep(self.wait_time)
            
            logger.info(f"Successfully downloaded paper to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to download paper: {str(e)}")
            raise ArxivError(f"Download failed for paper {result.entry_id}: {str(e)}")
    
    def search_and_download(
        self,
        query: str,
        max_results: int = None,
        output_dir: Optional[FilePath] = None,
    ) -> List[Tuple[arxiv.Result, str]]:
        """Search for papers and download them.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            output_dir: Directory to save papers (overrides default)
            
        Returns:
            List[Tuple[arxiv.Result, str]]: List of (result, pdf_path) tuples
            
        Raises:
            ArxivError: If search or download fails
        """
        # Search for papers
        results = self.search(query, max_results)
        
        # Download each paper
        downloaded = []
        for result in results:
            try:
                pdf_path = self.download_paper(result, output_dir)
                downloaded.append((result, pdf_path))
            except ArxivError as e:
                logger.warning(f"Skipping paper {result.entry_id}: {str(e)}")
                continue
        
        logger.info(f"Downloaded {len(downloaded)}/{len(results)} papers for query '{query}'")
        return downloaded
    
    def result_to_document_metadata(self, result: arxiv.Result) -> DocumentMetadata:
        """Convert arXiv result to DocumentMetadata.
        
        Args:
            result: arXiv result object
            
        Returns:
            DocumentMetadata: Document metadata
        """
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Extract arXiv ID
        arxiv_id = result.entry_id.split("/")[-1]
        
        # Extract basic metadata
        title = result.title
        authors = [author.name for author in result.authors]
        abstract = result.summary.replace("\n", " ").strip()
        
        # Format publication date
        pub_date = None
        if result.published:
            pub_date = result.published.isoformat()
        
        # Extract DOI if available
        doi = None
        if hasattr(result, "doi") and result.doi:
            doi = result.doi
        
        # Extract categories/subjects
        categories = result.categories if hasattr(result, "categories") else []
        
        # Create metadata object
        metadata = DocumentMetadata(
            document_id=doc_id,
            title=title,
            authors=authors,
            publication_date=pub_date,
            abstract=abstract,
            document_type="research_paper",
            source="arxiv",
            source_id=arxiv_id,
            doi=doi,
            categories=categories,
            extraction_method="arxiv_api",
        )
        
        return metadata
    
    def get_paper_by_id(self, paper_id: str) -> Optional[arxiv.Result]:
        """Get paper by arXiv ID.
        
        Args:
            paper_id: arXiv paper ID (with or without version)
            
        Returns:
            Optional[arxiv.Result]: arXiv result object or None if not found
            
        Raises:
            ArxivError: If retrieval fails
        """
        try:
            # Clean the ID
            paper_id = paper_id.strip()
            
            # Add arxiv prefix if not present
            if not paper_id.startswith("arxiv:"):
                paper_id = f"arxiv:{paper_id}"
            
            # Search for the specific paper
            search = arxiv.Search(
                id_list=[paper_id],
                max_results=1,
            )
            
            # Get the results
            results = list(self.client.results(search))
            
            # Wait after query to avoid rate limiting
            time.sleep(self.wait_time)
            
            if results:
                logger.info(f"Found paper with ID: {paper_id}")
                return results[0]
            else:
                logger.warning(f"No paper found with ID: {paper_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve paper with ID {paper_id}: {str(e)}")
            raise ArxivError(f"Failed to retrieve paper with ID {paper_id}: {str(e)}")
    
    def download_paper_by_id(
        self,
        paper_id: str,
        output_dir: Optional[FilePath] = None,
    ) -> Optional[str]:
        """Download paper by arXiv ID.
        
        Args:
            paper_id: arXiv paper ID
            output_dir: Directory to save paper (overrides default)
            
        Returns:
            Optional[str]: Path to downloaded PDF or None if paper not found
            
        Raises:
            ArxivError: If download fails
        """
        # Get paper by ID
        result = self.get_paper_by_id(paper_id)
        
        if result:
            # Download paper
            return self.download_paper(result, output_dir)
        else:
            return None 