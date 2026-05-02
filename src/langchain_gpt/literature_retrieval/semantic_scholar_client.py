"""Client for interacting with the Semantic Scholar API."""

import os
import time
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import requests
from requests.adapters import HTTPAdapter, Retry

from ..config.settings import get_settings
from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentMetadata, FilePath

logger = get_logger(__name__)


class SemanticScholarError(LangChainGPTError):
    """Error raised by Semantic Scholar operations."""
    
    def __init__(self, message: str = "Semantic Scholar error"):
        super().__init__(f"Semantic Scholar error: {message}")


class SemanticScholarClient:
    """Client for querying Semantic Scholar API."""
    
    # Base URL for the Semantic Scholar API
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # API fields to request (customize these based on your needs)
    PAPER_FIELDS = [
        "paperId", "title", "abstract", "url", "venue", "year", 
        "authors.name", "authors.authorId", "authors.url",
        "journal", "publicationVenue", "publicationDate",
        "doi", "externalIds", "fieldsOfStudy", "s2FieldsOfStudy",
        "tldr", "openAccessPdf"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        wait_time: Optional[int] = None,
        cache_dir: Optional[FilePath] = None,
        use_cache: bool = True,
    ):
        """Initialize Semantic Scholar client.
        
        Args:
            api_key: Semantic Scholar API key (optional)
            wait_time: Wait time between requests in seconds (rate limiting)
            cache_dir: Directory for caching API responses
            use_cache: Whether to use caching
            
        Raises:
            SemanticScholarError: If initialization fails
        """
        settings = get_settings()
        self.api_key = api_key or settings.SEMANTIC_SCHOLAR_API_KEY
        self.wait_time = wait_time or settings.SEMANTIC_SCHOLAR_WAIT_TIME
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.SEMANTIC_SCHOLAR_CACHE_DIR)
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure session with retry capabilities
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        
        # Set up headers for API requests
        self.headers = {
            "Accept": "application/json",
        }
        
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            logger.info("Using Semantic Scholar API with authentication")
        else:
            logger.info("Using Semantic Scholar API without authentication (rate limited)")
        
        # Initialize the request cache
        self._cache: Dict[str, Any] = {}
        
        # Load cache if caching is enabled
        if self.use_cache:
            self._load_cache()
        
        logger.info("Initialized Semantic Scholar client")
    
    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Semantic Scholar API with caching and rate limiting.
        
        Args:
            endpoint: API endpoint to request
            params: Request parameters
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            SemanticScholarError: If the request fails
        """
        # Generate cache key
        cache_key = self._get_cache_key(endpoint, params)
        
        # Check cache first
        if self.use_cache and cache_key in self._cache:
            logger.debug(f"Using cached response for {endpoint}")
            return self._cache[cache_key]
        
        # Construct URL
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            # Make the request
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30,
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", self.wait_time * 2))
                logger.warning(f"Rate limited by Semantic Scholar API. Waiting {wait_time}s")
                time.sleep(wait_time)
                return self._request(endpoint, params)
            
            # Raise for other errors
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Cache the response
            if self.use_cache:
                self._cache[cache_key] = data
                
                # Periodically save cache to disk
                if len(self._cache) % 10 == 0:
                    self._save_cache()
            
            # Wait to avoid rate limiting
            time.sleep(self.wait_time)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar API request failed: {str(e)}")
            raise SemanticScholarError(f"API request failed: {str(e)}")
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for a request.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            str: Cache key
        """
        # Create a string representation of the request
        key_parts = [endpoint]
        if params:
            # Sort params to ensure consistent keys
            for k in sorted(params.keys()):
                key_parts.append(f"{k}={params[k]}")
        
        key_str = "|".join(key_parts)
        
        # Generate a hash
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()
    
    def _save_cache(self) -> None:
        """Save API response cache to disk."""
        if not self.use_cache or not self._cache:
            return
        
        try:
            cache_file = self.cache_dir / "semantic_scholar_cache.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} Semantic Scholar responses to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save Semantic Scholar cache: {str(e)}")
    
    def _load_cache(self) -> None:
        """Load API response cache from disk."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / "semantic_scholar_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.debug(f"Loaded {len(self._cache)} Semantic Scholar responses from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load Semantic Scholar cache: {str(e)}")
                self._cache = {}
    
    def get_paper(self, paper_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get paper details by ID.
        
        Args:
            paper_id: Paper ID (can be DOI, arXiv ID, etc.)
            fields: Fields to retrieve (defaults to self.PAPER_FIELDS)
            
        Returns:
            Dict[str, Any]: Paper details
            
        Raises:
            SemanticScholarError: If the paper cannot be retrieved
        """
        try:
            # Use default fields if none provided
            if fields is None:
                fields = self.PAPER_FIELDS
            
            # Prepare request
            endpoint = f"paper/{paper_id}"
            params = {
                "fields": ",".join(fields),
            }
            
            # Make the request
            response = self._request(endpoint, params)
            
            logger.info(f"Retrieved paper with ID: {paper_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get paper with ID {paper_id}: {str(e)}")
            raise SemanticScholarError(f"Failed to get paper with ID {paper_id}: {str(e)}")
    
    def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
        year: Optional[int] = None,
        venue: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for papers by query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            fields: Fields to retrieve (defaults to self.PAPER_FIELDS)
            year: Filter by publication year
            venue: Filter by publication venue
            
        Returns:
            List[Dict[str, Any]]: List of paper data
            
        Raises:
            SemanticScholarError: If the search fails
        """
        try:
            # Use default fields if none provided
            if fields is None:
                fields = self.PAPER_FIELDS
            
            # Prepare request
            endpoint = "paper/search"
            params = {
                "query": query,
                "limit": min(limit, 100),  # API limit is 100
                "fields": ",".join(fields),
            }
            
            # Add optional filters
            if year:
                params["year"] = year
            if venue:
                params["venue"] = venue
            
            # Make the request
            response = self._request(endpoint, params)
            
            # Extract paper data
            papers = response.get("data", [])
            total = response.get("total", 0)
            
            logger.info(f"Found {total} papers for query '{query}', returning {len(papers)}")
            return papers
            
        except Exception as e:
            logger.error(f"Paper search failed for query '{query}': {str(e)}")
            raise SemanticScholarError(f"Paper search failed for query '{query}': {str(e)}")
    
    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get papers that cite the specified paper.
        
        Args:
            paper_id: Paper ID
            limit: Maximum number of citations to retrieve
            fields: Fields to retrieve for each citation
            
        Returns:
            List[Dict[str, Any]]: List of citing papers
            
        Raises:
            SemanticScholarError: If the citations cannot be retrieved
        """
        try:
            # Use default fields if none provided
            if fields is None:
                fields = self.PAPER_FIELDS
            
            # Prepare request
            endpoint = f"paper/{paper_id}/citations"
            params = {
                "limit": min(limit, 1000),  # API limit is 1000
                "fields": ",".join(fields),
            }
            
            # Make the request
            response = self._request(endpoint, params)
            
            # Extract citation data
            citations = response.get("data", [])
            total = response.get("total", 0)
            
            # Extract only the citing paper from each citation object
            citing_papers = [citation.get("citingPaper", {}) for citation in citations]
            
            logger.info(f"Retrieved {len(citing_papers)} citations for paper {paper_id} (total: {total})")
            return citing_papers
            
        except Exception as e:
            logger.error(f"Failed to get citations for paper {paper_id}: {str(e)}")
            raise SemanticScholarError(f"Failed to get citations for paper {paper_id}: {str(e)}")
    
    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get papers that are referenced by the specified paper.
        
        Args:
            paper_id: Paper ID
            limit: Maximum number of references to retrieve
            fields: Fields to retrieve for each reference
            
        Returns:
            List[Dict[str, Any]]: List of referenced papers
            
        Raises:
            SemanticScholarError: If the references cannot be retrieved
        """
        try:
            # Use default fields if none provided
            if fields is None:
                fields = self.PAPER_FIELDS
            
            # Prepare request
            endpoint = f"paper/{paper_id}/references"
            params = {
                "limit": min(limit, 1000),  # API limit is 1000
                "fields": ",".join(fields),
            }
            
            # Make the request
            response = self._request(endpoint, params)
            
            # Extract reference data
            references = response.get("data", [])
            total = response.get("total", 0)
            
            # Extract only the referenced paper from each reference object
            referenced_papers = [reference.get("citedPaper", {}) for reference in references]
            
            logger.info(f"Retrieved {len(referenced_papers)} references for paper {paper_id} (total: {total})")
            return referenced_papers
            
        except Exception as e:
            logger.error(f"Failed to get references for paper {paper_id}: {str(e)}")
            raise SemanticScholarError(f"Failed to get references for paper {paper_id}: {str(e)}")
    
    def get_citation_graph(
        self,
        paper_id: str,
        depth: int = 1,
        max_papers_per_level: int = 20,
        direction: str = "both",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get citation graph for a paper.
        
        Args:
            paper_id: Paper ID
            depth: Depth of the graph traversal
            max_papers_per_level: Maximum papers to include per level
            direction: Direction of traversal ('citations', 'references', 'both')
            fields: Fields to retrieve for each paper
            
        Returns:
            Dict[str, Any]: Dictionary containing nodes and edges of the citation graph
            
        Raises:
            SemanticScholarError: If the citation graph cannot be retrieved
        """
        if depth < 1:
            raise SemanticScholarError("Depth must be at least 1")
        
        if direction not in ["citations", "references", "both"]:
            raise SemanticScholarError("Direction must be 'citations', 'references', or 'both'")
        
        try:
            # Initialize graph data
            nodes: Dict[str, Dict[str, Any]] = {}
            edges: List[Dict[str, Any]] = []
            
            # Start with the seed paper
            seed_paper = self.get_paper(paper_id, fields)
            seed_id = seed_paper.get("paperId")
            
            if not seed_id:
                raise SemanticScholarError(f"Could not find paper with ID {paper_id}")
            
            # Add seed paper to nodes
            nodes[seed_id] = seed_paper
            
            # Track visited papers to avoid duplicates
            visited: Set[str] = {seed_id}
            
            # Queue for BFS traversal
            queue: List[Tuple[str, int]] = [(seed_id, 0)]  # (paper_id, depth_level)
            
            # Process queue
            while queue:
                current_id, current_depth = queue.pop(0)
                
                # Stop if we've reached the maximum depth
                if current_depth >= depth:
                    continue
                
                # Get citations (papers that cite the current paper)
                if direction in ["citations", "both"]:
                    try:
                        citing_papers = self.get_paper_citations(
                            current_id, 
                            limit=max_papers_per_level,
                            fields=fields,
                        )
                        
                        for citing_paper in citing_papers:
                            citing_id = citing_paper.get("paperId")
                            
                            if citing_id and citing_id not in visited:
                                visited.add(citing_id)
                                nodes[citing_id] = citing_paper
                                
                                # Add edge from citing paper to current paper
                                edges.append({
                                    "source": citing_id,
                                    "target": current_id,
                                    "type": "citation",
                                })
                                
                                # Add to queue for next level
                                if current_depth + 1 < depth:
                                    queue.append((citing_id, current_depth + 1))
                    except SemanticScholarError as e:
                        logger.warning(f"Error getting citations for {current_id}: {str(e)}")
                
                # Get references (papers cited by the current paper)
                if direction in ["references", "both"]:
                    try:
                        referenced_papers = self.get_paper_references(
                            current_id,
                            limit=max_papers_per_level,
                            fields=fields,
                        )
                        
                        for referenced_paper in referenced_papers:
                            referenced_id = referenced_paper.get("paperId")
                            
                            if referenced_id and referenced_id not in visited:
                                visited.add(referenced_id)
                                nodes[referenced_id] = referenced_paper
                                
                                # Add edge from current paper to referenced paper
                                edges.append({
                                    "source": current_id,
                                    "target": referenced_id,
                                    "type": "reference",
                                })
                                
                                # Add to queue for next level
                                if current_depth + 1 < depth:
                                    queue.append((referenced_id, current_depth + 1))
                    except SemanticScholarError as e:
                        logger.warning(f"Error getting references for {current_id}: {str(e)}")
            
            # Convert nodes dictionary to list
            node_list = list(nodes.values())
            
            logger.info(f"Generated citation graph with {len(node_list)} nodes and {len(edges)} edges")
            return {
                "nodes": node_list,
                "edges": edges,
            }
            
        except Exception as e:
            logger.error(f"Failed to generate citation graph for {paper_id}: {str(e)}")
            raise SemanticScholarError(f"Failed to generate citation graph for {paper_id}: {str(e)}")
    
    def paper_to_document_metadata(self, paper: Dict[str, Any]) -> DocumentMetadata:
        """Convert Semantic Scholar paper data to DocumentMetadata.
        
        Args:
            paper: Semantic Scholar paper data
            
        Returns:
            DocumentMetadata: Document metadata
        """
        # Extract basic metadata
        paper_id = paper.get("paperId")
        title = paper.get("title", "Untitled Paper")
        
        # Extract authors
        authors = []
        for author in paper.get("authors", []):
            if author.get("name"):
                authors.append(author.get("name"))
        
        # Extract abstract
        abstract = paper.get("abstract", "")
        
        # Extract publication date
        pub_date = paper.get("publicationDate")
        
        # Extract DOI
        doi = None
        ext_ids = paper.get("externalIds", {})
        if ext_ids and "DOI" in ext_ids:
            doi = ext_ids["DOI"]
        
        # Extract venue/journal
        venue = paper.get("venue") or paper.get("journal", {}).get("name")
        
        # Extract year
        year = paper.get("year")
        
        # Extract fields of study
        fields_of_study = paper.get("fieldsOfStudy", [])
        
        # Extract PDF URL
        pdf_url = None
        open_access = paper.get("openAccessPdf", {})
        if open_access:
            pdf_url = open_access.get("url")
        
        # Create metadata object
        metadata = DocumentMetadata(
            document_id=paper_id,
            title=title,
            authors=authors,
            publication_date=pub_date,
            abstract=abstract,
            document_type="research_paper",
            source="semantic_scholar",
            source_id=paper_id,
            doi=doi,
            url=paper.get("url"),
            venue=venue,
            year=year,
            fields_of_study=fields_of_study,
            pdf_url=pdf_url,
            extraction_method="semantic_scholar_api",
        )
        
        return metadata

    def get_citation_count(self, paper_id):
        url = f"{self.BASE_URL}/paper/{paper_id}?fields=citationCount"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("citationCount", 0)
        else:
            logger.error(f"Failed to retrieve citation count for {paper_id}: {response.status_code}")
            return 0

# Example usage
if __name__ == "__main__":
    client = SemanticScholarClient(api_key="your_api_key_here")
    citation_count = client.get_citation_count("paper_id_here")
    logger.info(f"Citation count for paper_id_here: {citation_count}") 