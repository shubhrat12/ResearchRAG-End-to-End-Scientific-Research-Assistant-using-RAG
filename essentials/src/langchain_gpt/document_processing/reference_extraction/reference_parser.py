"""Parse and structure references from document text."""

import re
from typing import Dict, List, Optional, Tuple, Any

from ...utils.logging import get_logger
from ...utils.types import Document, DocumentSection, Reference

logger = get_logger(__name__)


class ReferenceParser:
    """Parse and structure references from document text."""
    
    def __init__(self, **kwargs):
        """Initialize the reference parser.
        
        Args:
            **kwargs: Additional configuration options
        """
        # Configure parser options
        self.detect_doi = kwargs.get("detect_doi", True)
        self.detect_urls = kwargs.get("detect_urls", True)
        self.min_confidence = kwargs.get("min_confidence", 0.6)
        
        # Compile reference patterns
        self._compile_patterns()
        
        logger.debug(
            f"Initialized ReferenceParser with detect_doi={self.detect_doi}, "
            f"detect_urls={self.detect_urls}"
        )
    
    def _compile_patterns(self):
        """Compile regex patterns for reference parsing."""
        # Numbered reference pattern
        # Example: "[1] Smith, J. (2020). Title. Journal, 10(2), 100-110."
        self.numbered_ref_pattern = re.compile(r'(?:^|\n)\s*\[?(\d+)\]?\.?\s+(.+?)(?=(?:^|\n)\s*\[?(?:\d+)\]?\.?\s+|\Z)', re.DOTALL)
        
        # Author-year reference pattern
        # Example: "Smith, J. (2020). Title. Journal, 10(2), 100-110."
        self.author_year_ref_pattern = re.compile(r'(?:^|\n)\s*([A-Z][a-z]+(?:,\s+[A-Z]\.)+(?:\s+and\s+[A-Z][a-z]+(?:,\s+[A-Z]\.)+)?)(?:\s+\((\d{4}[a-z]?)\))\.?\s+(.+?)(?=(?:^|\n)\s*[A-Z][a-z]+(?:,\s+[A-Z]\.)+|\Z)', re.DOTALL)
        
        # DOI pattern
        self.doi_pattern = re.compile(r'(?:DOI|doi):\s*(10\.\d{4,}(?:\.\d+)*\/(?:(?!["&\'])\S)+)')
        
        # URL pattern
        self.url_pattern = re.compile(r'https?://[^\s<>"]+(?<![\.,;:!?])')
    
    def parse_references(self, text: str) -> List[Reference]:
        """Parse references from text.
        
        Args:
            text: Text to parse references from
            
        Returns:
            List[Reference]: List of parsed references
        """
        if not text:
            return []
        
        references = []
        
        # Try numbered references first
        numbered_refs = self._parse_numbered_references(text)
        if numbered_refs:
            references.extend(numbered_refs)
        
        # If few numbered references found, try author-year
        if len(numbered_refs) < 3:
            author_year_refs = self._parse_author_year_references(text)
            
            # Avoid duplicates - only add if reference doesn't overlap with numbered refs
            if author_year_refs:
                # Create sets of text spans for numbered references
                numbered_spans = set()
                for ref in numbered_refs:
                    if hasattr(ref, 'span') and ref.span:
                        start, end = ref.span
                        numbered_spans.update(range(start, end + 1))
                
                # Add author-year references that don't overlap
                for ref in author_year_refs:
                    if hasattr(ref, 'span') and ref.span:
                        start, end = ref.span
                        # Check for overlap
                        if not any(p in numbered_spans for p in range(start, end + 1)):
                            references.append(ref)
        
        # Sort references by ID
        references.sort(key=lambda r: int(r.ref_id) if r.ref_id.isdigit() else float('inf'))
        
        logger.debug(f"Parsed {len(references)} references from text")
        return references
    
    def _parse_numbered_references(self, text: str) -> List[Reference]:
        """Parse numbered references from text.
        
        Args:
            text: Text to parse from
            
        Returns:
            List[Reference]: Parsed numbered references
        """
        references = []
        
        # Find all numbered references
        for match in self.numbered_ref_pattern.finditer(text):
            # Get reference parts
            ref_id = match.group(1)
            ref_text = match.group(2).strip()
            
            # Skip if too short (likely not a reference)
            if len(ref_text) < 20:
                continue
            
            # Extract structured data
            structured_data = self._extract_reference_data(ref_text)
            
            # Create reference object
            reference = Reference(
                ref_id=ref_id,
                text=ref_text,
                reference_type="numbered",
                confidence=0.9,
                span=(match.start(), match.end()),
                **structured_data
            )
            
            references.append(reference)
        
        return references
    
    def _parse_author_year_references(self, text: str) -> List[Reference]:
        """Parse author-year references from text.
        
        Args:
            text: Text to parse from
            
        Returns:
            List[Reference]: Parsed author-year references
        """
        references = []
        
        # Find all author-year references
        for match in self.author_year_ref_pattern.finditer(text):
            # Get reference parts
            authors = match.group(1)
            year = match.group(2) if match.group(2) else ""
            ref_text = match.group(3).strip()
            
            # Skip if too short (likely not a reference)
            if len(ref_text) < 20:
                continue
            
            # Generate ref_id from author and year
            ref_id = f"{authors.split(',')[0].strip()}{year}"
            
            # Extract structured data
            structured_data = self._extract_reference_data(ref_text)
            structured_data["authors"] = [authors] if authors else []
            structured_data["year"] = year
            
            # Create reference object
            reference = Reference(
                ref_id=ref_id,
                text=f"{authors} ({year}). {ref_text}",
                reference_type="author_year",
                confidence=0.85,
                span=(match.start(), match.end()),
                **structured_data
            )
            
            references.append(reference)
        
        return references
    
    def _extract_reference_data(self, ref_text: str) -> Dict[str, Any]:
        """Extract structured data from reference text.
        
        Args:
            ref_text: Reference text
            
        Returns:
            Dict[str, Any]: Structured reference data
        """
        data = {
            "title": "",
            "journal": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": "",
            "url": "",
            "authors": [],
            "year": ""
        }
        
        # Extract title (simplistic approach - first sentence)
        title_match = re.search(r'^(.*?)\.', ref_text)
        if title_match:
            data["title"] = title_match.group(1).strip()
        
        # Extract journal name (after title, in italics or quotes)
        journal_match = re.search(r'\.(?:.*?)"(.*?)"', ref_text) or re.search(r'\.(.*?),\s+\d+', ref_text)
        if journal_match:
            data["journal"] = journal_match.group(1).strip()
        
        # Extract volume and issue
        volume_match = re.search(r'(\d+)\s*\((\d+(?:-\d+)?)\)', ref_text)
        if volume_match:
            data["volume"] = volume_match.group(1)
            data["issue"] = volume_match.group(2)
        
        # Extract pages
        pages_match = re.search(r'(?:p\.?|pp\.?|pages?)\s*(\d+[-–—]?\d*)', ref_text, re.IGNORECASE) or re.search(r'(\d+)[-–—](\d+)(?!\d)', ref_text)
        if pages_match:
            if len(pages_match.groups()) == 2:
                data["pages"] = f"{pages_match.group(1)}–{pages_match.group(2)}"
            else:
                data["pages"] = pages_match.group(1)
        
        # Extract year
        year_match = re.search(r'\((\d{4}[a-z]?)\)', ref_text)
        if year_match:
            data["year"] = year_match.group(1)
        
        # Extract DOI if enabled
        if self.detect_doi:
            doi_match = self.doi_pattern.search(ref_text)
            if doi_match:
                data["doi"] = doi_match.group(1)
        
        # Extract URL if enabled
        if self.detect_urls:
            url_match = self.url_pattern.search(ref_text)
            if url_match:
                data["url"] = url_match.group(0)
        
        # Extract authors
        authors_match = re.match(r'([A-Z][a-z]+(?:,\s+[A-Z]\.)+(?:(?:,|,?\s+and)\s+[A-Z][a-z]+(?:,\s+[A-Z]\.)+)*)', ref_text)
        if authors_match:
            # Split authors by "and" or comma
            author_text = authors_match.group(1)
            author_parts = re.split(r',?\s+and\s+|\s*,\s*(?=[A-Z])', author_text)
            data["authors"] = [part.strip() for part in author_parts if part.strip()]
        
        return data
    
    def extract_references_section(self, document: Document) -> List[Reference]:
        """Extract and parse the references section from a document.
        
        Args:
            document: Document to process
            
        Returns:
            List[Reference]: List of parsed references
        """
        if not document.full_text:
            logger.warning("Document has no full text content")
            return []
        
        references = []
        
        # Check if document has sections
        if hasattr(document, 'sections') and document.sections:
            for section in document.sections:
                # Look for references/bibliography section
                if section.section_type == "references" or \
                   section.title and any(term in section.title.lower() for term in ["reference", "bibliography", "works cited"]):
                    logger.info(f"Found references section: {section.title}")
                    if section.content:
                        section_refs = self.parse_references(section.content)
                        references.extend(section_refs)
                        break
        
        # If no references found in sections, try to find references section in full text
        if not references:
            # Look for references section heading
            ref_section_matches = re.finditer(r'(?:^|\n)\s*(References|Bibliography|Works Cited|Literature Cited)(?:\s|\n|:|\.)', document.full_text, re.IGNORECASE)
            
            for match in ref_section_matches:
                # Extract text from the heading to the end of document (or next major section)
                start_pos = match.end()
                end_pos = len(document.full_text)
                
                # Try to find the next major section (if any)
                next_section_match = re.search(r'(?:^|\n)\s*[A-Z][A-Z\s]{2,}(?:\s|\n|:|\.)', document.full_text[start_pos:], re.MULTILINE)
                if next_section_match:
                    end_pos = start_pos + next_section_match.start()
                
                # Extract references section text
                ref_section_text = document.full_text[start_pos:end_pos].strip()
                
                # Parse references
                section_refs = self.parse_references(ref_section_text)
                references.extend(section_refs)
                break
        
        logger.info(f"Extracted {len(references)} references from document")
        return references 