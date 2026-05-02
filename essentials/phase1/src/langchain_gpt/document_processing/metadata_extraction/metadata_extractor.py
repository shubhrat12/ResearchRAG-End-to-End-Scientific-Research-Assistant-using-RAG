"""Extract and enhance document metadata from PDFs."""

import re
import datetime
from typing import Dict, List, Optional, Tuple, Any

from ...utils.logging import get_logger
from ...utils.types import Document, DocumentMetadata

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract and enhance document metadata from PDF documents."""
    
    def __init__(self, **kwargs):
        """Initialize the metadata extractor.
        
        Args:
            **kwargs: Additional configuration options
        """
        # Configure extractor options
        self.extract_keywords = kwargs.get("extract_keywords", True)
        self.detect_publication = kwargs.get("detect_publication", True)
        self.enhance_authors = kwargs.get("enhance_authors", True)
        self.min_confidence = kwargs.get("min_confidence", 0.6)
        
        # Compile metadata patterns
        self._compile_patterns()
        
        logger.debug(
            f"Initialized MetadataExtractor with extract_keywords={self.extract_keywords}, "
            f"detect_publication={self.detect_publication}"
        )
    
    def _compile_patterns(self):
        """Compile regex patterns for metadata extraction."""
        # Title pattern (looks for large text at the beginning)
        self.title_pattern = re.compile(r'^\s*(?:[A-Z][A-Za-z0-9\s,:-]+){3,}(?:\n|$)')
        
        # Author patterns
        # Example: "John Smith¹, Jane Doe²"
        self.authors_pattern = re.compile(r'(?:^|\n)(?!Abstract|Introduction)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)\s*(?:¹|²|³|⁴|⁵|\*|\d)?(?:,|$|\n)')
        
        # Affiliation patterns
        # Example: "¹Department of Computer Science, University of Example"
        self.affiliation_pattern = re.compile(r'(?:^|\n)(?:¹|²|³|⁴|⁵|\*)?\s*([A-Z][a-zA-Z\s]+,\s+(?:University|Institute|College|Department|School|Lab)[^,\n]+)(?:,|$|\n)')
        
        # Date patterns
        # Examples: "Published: January 2020", "Received: 15 March 2020, Accepted: 2 April 2020"
        self.date_pattern = re.compile(r'(?:submitted|received|accepted|published):\s*(\d{1,2}\s+)?([a-z]+\s+\d{4}|\d{4})', re.IGNORECASE)
        
        # Keywords pattern
        # Example: "Keywords: machine learning, artificial intelligence, data science"
        self.keywords_pattern = re.compile(r'(?:keywords|key\s+words|index\s+terms):\s*([^.]+)(?:\.|$)', re.IGNORECASE)
        
        # Journal/Conference patterns
        # Example: "Journal of Computer Science, Vol. 10, No. 2, 2020"
        self.journal_pattern = re.compile(r'((?:Journal|Proceedings|Transactions|Conference)[^,\n]{3,}),\s+(?:Vol\.|Volume)\s+(\d+)', re.IGNORECASE)
        
        # DOI pattern
        self.doi_pattern = re.compile(r'(?:DOI|doi):\s*(10\.\d{4,}(?:\.\d+)*\/(?:(?!["&\'])\S)+)')
    
    def extract_metadata(self, document: Document) -> DocumentMetadata:
        """Extract and enhance metadata from a document.
        
        Args:
            document: Document to process
            
        Returns:
            DocumentMetadata: Enhanced document metadata
        """
        if not document.full_text:
            logger.warning("Document has no full text content")
            return document.metadata
        
        # Start with existing metadata
        enhanced_metadata = self._clone_metadata(document.metadata)
        
        # Get text sample from beginning of document (first page or so)
        text_sample = document.full_text[:5000]
        
        # Extract title if not already present or enhance existing
        if not enhanced_metadata.title or len(enhanced_metadata.title) < 5:
            title = self._extract_title(text_sample)
            if title:
                enhanced_metadata.title = title
                logger.debug(f"Extracted title: {title}")
        
        # Extract or enhance authors
        if not enhanced_metadata.authors or len(enhanced_metadata.authors) == 0:
            authors = self._extract_authors(text_sample)
            if authors:
                enhanced_metadata.authors = authors
                logger.debug(f"Extracted authors: {authors}")
        elif self.enhance_authors:
            authors = self._extract_authors(text_sample)
            if len(authors) > len(enhanced_metadata.authors):
                enhanced_metadata.authors = authors
                logger.debug(f"Enhanced authors: {authors}")
        
        # Extract publication date if not present
        if not enhanced_metadata.date or enhanced_metadata.date == "":
            date = self._extract_date(text_sample)
            if date:
                enhanced_metadata.date = date
                logger.debug(f"Extracted date: {date}")
        
        # Extract keywords if enabled
        if self.extract_keywords:
            keywords = self._extract_keywords(text_sample)
            if keywords:
                enhanced_metadata.keywords = keywords
                logger.debug(f"Extracted keywords: {keywords}")
        
        # Extract publication information if enabled
        if self.detect_publication:
            publication = self._extract_publication(text_sample)
            if publication:
                enhanced_metadata.publication = publication
                logger.debug(f"Extracted publication: {publication}")
        
        # Extract DOI if not present
        if not enhanced_metadata.doi or enhanced_metadata.doi == "":
            doi = self._extract_doi(document.full_text)
            if doi:
                enhanced_metadata.doi = doi
                logger.debug(f"Extracted DOI: {doi}")
        
        # Record extraction method
        enhanced_metadata.extraction_method = "enhanced"
        
        logger.info(f"Enhanced metadata for document: {enhanced_metadata.title}")
        return enhanced_metadata
    
    def _clone_metadata(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """Create a copy of the metadata to avoid modifying the original.
        
        Args:
            metadata: Original metadata
            
        Returns:
            DocumentMetadata: Copy of metadata
        """
        # Create a new metadata object with the same attributes
        return DocumentMetadata(
            title=metadata.title,
            authors=metadata.authors.copy() if metadata.authors else [],
            date=metadata.date,
            source=metadata.source,
            document_type=metadata.document_type,
            pages=metadata.pages,
            file_path=metadata.file_path,
            file_size=metadata.file_size,
            extraction_date=metadata.extraction_date,
            doi=metadata.doi,
            abstract=metadata.abstract,
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract the document title from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            str: Extracted title or empty string
        """
        # Check for title pattern at start of document
        title_match = self.title_pattern.search(text)
        if title_match:
            title = title_match.group(0).strip()
            
            # Clean up title
            title = re.sub(r'\s+', ' ', title)
            
            return title
        
        # If no match, try to find the first line with significant capitalized text
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 20 and line.isupper():
                return line
            if len(line) > 30 and sum(1 for c in line if c.isupper()) > len(line) / 3:
                return line
        
        return ""
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List[str]: List of author names
        """
        authors = []
        
        # Find author patterns
        for match in self.authors_pattern.finditer(text):
            author_text = match.group(1).strip()
            
            # Skip if too short or contains unwanted terms
            if len(author_text) < 5 or any(term in author_text.lower() for term in ["abstract", "introduction", "university", "keywords"]):
                continue
            
            # Split multiple authors if separated by commas
            if "," in author_text:
                for author in author_text.split(","):
                    author = author.strip()
                    if author and author not in authors:
                        authors.append(author)
            else:
                if author_text not in authors:
                    authors.append(author_text)
        
        return authors
    
    def _extract_date(self, text: str) -> str:
        """Extract publication date from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            str: Extracted date or empty string
        """
        # Look for explicit date mentions
        date_match = self.date_pattern.search(text)
        if date_match:
            date_str = (date_match.group(1) or "") + date_match.group(2)
            return date_str.strip()
        
        # Look for year mentions
        year_match = re.search(r'©\s*(\d{4})', text) or re.search(r'\bCopyright\s+(?:\(c\))?\s*(\d{4})', text, re.IGNORECASE)
        if year_match:
            return year_match.group(1)
        
        # Default to current year if nothing found
        return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List[str]: List of keywords
        """
        keywords = []
        
        # Find keywords section
        keywords_match = self.keywords_pattern.search(text)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            
            # Split by commas or semicolons
            if ";" in keywords_text:
                keywords = [k.strip() for k in keywords_text.split(";") if k.strip()]
            else:
                keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        
        return keywords
    
    def _extract_publication(self, text: str) -> Dict[str, str]:
        """Extract publication information from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Dict[str, str]: Publication information
        """
        publication = {}
        
        # Look for journal/conference information
        journal_match = self.journal_pattern.search(text)
        if journal_match:
            publication["name"] = journal_match.group(1).strip()
            publication["volume"] = journal_match.group(2)
            
            # Try to find issue number
            issue_match = re.search(r'(?:No\.|Number|Issue)\s+(\d+)', text, re.IGNORECASE)
            if issue_match:
                publication["issue"] = issue_match.group(1)
            
            # Try to find pages
            pages_match = re.search(r'(?:pp\.|pages)\s+(\d+)[-–](\d+)', text, re.IGNORECASE)
            if pages_match:
                publication["pages"] = f"{pages_match.group(1)}–{pages_match.group(2)}"
        
        return publication
    
    def _extract_doi(self, text: str) -> str:
        """Extract DOI from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            str: Extracted DOI or empty string
        """
        doi_match = self.doi_pattern.search(text)
        if doi_match:
            return doi_match.group(1)
        return ""
    
    def enhance_document(self, document: Document) -> Document:
        """Enhance document with improved metadata.
        
        Args:
            document: Document to enhance
            
        Returns:
            Document: Enhanced document
        """
        # Extract enhanced metadata
        enhanced_metadata = self.extract_metadata(document)
        
        # Create a new document with enhanced metadata
        enhanced_document = Document(
            metadata=enhanced_metadata,
            chunks=document.chunks,
            full_text=document.full_text,
        )
        
        # Copy over sections if present
        if hasattr(document, 'sections') and document.sections:
            enhanced_document.sections = document.sections
        
        return enhanced_document 