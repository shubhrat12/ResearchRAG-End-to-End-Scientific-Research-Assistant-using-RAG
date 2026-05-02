"""Extract citation markers from document text."""

import re
from typing import Dict, List, Optional, Tuple, Any

from ...utils.logging import get_logger
from ...utils.types import Document, DocumentSection, Citation

logger = get_logger(__name__)


class CitationExtractor:
    """Extract citation markers from document text."""
    
    def __init__(self, **kwargs):
        """Initialize the citation extractor.
        
        Args:
            **kwargs: Additional configuration options
        """
        # Configure extraction options
        self.detect_numeric = kwargs.get("detect_numeric", True)
        self.detect_author_year = kwargs.get("detect_author_year", True)
        self.min_confidence = kwargs.get("min_confidence", 0.6)
        
        # Compile citation patterns
        self._compile_patterns()
        
        logger.debug(
            f"Initialized CitationExtractor with detect_numeric={self.detect_numeric}, "
            f"detect_author_year={self.detect_author_year}"
        )
    
    def _compile_patterns(self):
        """Compile regex patterns for citation detection."""
        # Numbered citation patterns
        # Examples: [1], [1,2], [1-3], [1, 2, 3]
        self.numbered_pattern = re.compile(r'\[(\d+(?:\s*[,-]\s*\d+)*)\]')
        
        # Parenthetical citation patterns
        # Examples: (Smith, 2020), (Smith and Jones, 2020), (Smith et al., 2020)
        author_year_basic = r'\(([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+et\s+al\.?)?),\s+(\d{4}[a-z]?)\)'
        self.author_year_pattern = re.compile(author_year_basic)
        
        # Multiple citations pattern
        # Example: (Smith, 2020; Jones, 2019)
        self.multi_citation_pattern = re.compile(r'\(([^()]+(?:;\s*[^()]+)+)\)')
    
    def extract_citations(self, text: str) -> List[Citation]:
        """Extract citation markers from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List[Citation]: List of extracted citations
        """
        if not text:
            return []
        
        citations = []
        
        # Extract numbered citations if enabled
        if self.detect_numeric:
            numbered_citations = self._extract_numbered_citations(text)
            citations.extend(numbered_citations)
        
        # Extract author-year citations if enabled
        if self.detect_author_year:
            author_year_citations = self._extract_author_year_citations(text)
            citations.extend(author_year_citations)
        
        # Sort citations by position
        citations.sort(key=lambda c: c.position)
        
        logger.debug(f"Extracted {len(citations)} citations from text")
        return citations
    
    def _extract_numbered_citations(self, text: str) -> List[Citation]:
        """Extract numbered citation markers.
        
        Args:
            text: Text to extract from
            
        Returns:
            List[Citation]: Extracted numbered citations
        """
        citations = []
        
        # Find all numbered citations
        for match in self.numbered_pattern.finditer(text):
            # Get citation text (e.g., "[1]")
            citation_text = match.group(0)
            # Get citation numbers (e.g., "1" from "[1]")
            citation_numbers = match.group(1)
            
            # Process citation numbers (handle ranges and lists)
            ref_ids = self._process_citation_numbers(citation_numbers)
            
            # Create citation object
            citation = Citation(
                text=citation_text,
                ref_ids=ref_ids,
                citation_type="numeric",
                position=match.start(),
                confidence=0.95
            )
            
            citations.append(citation)
        
        return citations
    
    def _process_citation_numbers(self, numbers_text: str) -> List[str]:
        """Process citation numbers text into individual reference IDs.
        
        Args:
            numbers_text: Text with citation numbers (e.g., "1, 2-4")
            
        Returns:
            List[str]: List of reference IDs
        """
        ref_ids = []
        
        # Split by comma
        parts = [p.strip() for p in numbers_text.split(',')]
        
        for part in parts:
            if '-' in part:
                # Handle ranges like "1-3"
                try:
                    start, end = part.split('-')
                    start_num = int(start.strip())
                    end_num = int(end.strip())
                    # Add all numbers in range
                    for num in range(start_num, end_num + 1):
                        ref_ids.append(str(num))
                except ValueError:
                    # If splitting fails, add as is
                    ref_ids.append(part.strip())
            else:
                # Add single number
                ref_ids.append(part.strip())
        
        return ref_ids
    
    def _extract_author_year_citations(self, text: str) -> List[Citation]:
        """Extract author-year citation markers.
        
        Args:
            text: Text to extract from
            
        Returns:
            List[Citation]: Extracted author-year citations
        """
        citations = []
        
        # Find all author-year citations
        for match in self.author_year_pattern.finditer(text):
            # Get citation text (e.g., "(Smith, 2020)")
            citation_text = match.group(0)
            # Get author and year parts
            author = match.group(1)
            year = match.group(2)
            
            # Create citation object
            citation = Citation(
                text=citation_text,
                ref_ids=[f"{author.strip()}{year}"],
                citation_type="author_year",
                authors=[author.strip()],
                year=year,
                position=match.start(),
                confidence=0.9
            )
            
            citations.append(citation)
        
        # Find multiple citations (e.g., (Smith, 2020; Jones, 2019))
        for match in self.multi_citation_pattern.finditer(text):
            # Avoid matches that are already captured by the author-year pattern
            if self.author_year_pattern.match(match.group(0)):
                continue
            
            # Get citation text (e.g., "(Smith, 2020; Jones, 2019)")
            citation_text = match.group(0)
            citation_parts = match.group(1).split(';')
            
            ref_ids = []
            authors = []
            years = []
            
            # Process each part of the multiple citation
            for part in citation_parts:
                part = part.strip()
                
                # Try to match author-year pattern
                author_year_match = re.match(r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?(?:\s+et\s+al\.?)?),\s+(\d{4}[a-z]?)', part)
                
                if author_year_match:
                    author = author_year_match.group(1)
                    year = author_year_match.group(2)
                    ref_ids.append(f"{author.strip()}{year}")
                    authors.append(author.strip())
                    years.append(year)
            
            # Only add if we successfully parsed at least one part
            if ref_ids:
                citation = Citation(
                    text=citation_text,
                    ref_ids=ref_ids,
                    citation_type="multiple_author_year",
                    authors=authors,
                    year=", ".join(years),
                    position=match.start(),
                    confidence=0.85
                )
                
                citations.append(citation)
        
        return citations
    
    def extract_document_citations(self, document: Document) -> Dict[str, List[Citation]]:
        """Extract citations from all sections of a document.
        
        Args:
            document: Document to process
            
        Returns:
            Dict[str, List[Citation]]: Dictionary mapping section titles to citations
        """
        if not document.full_text:
            logger.warning("Document has no full text content")
            return {}
        
        result = {}
        
        # Extract from full text if no sections
        if not hasattr(document, 'sections') or not document.sections:
            citations = self.extract_citations(document.full_text)
            if citations:
                result["document"] = citations
            return result
        
        # Extract from each section
        for section in document.sections:
            if section.content:
                section_citations = self.extract_citations(section.content)
                if section_citations:
                    result[section.title or "Unnamed Section"] = section_citations
        
        return result 