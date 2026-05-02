"""Rule-based section detector for scientific papers."""

import re
from typing import Dict, List, Optional, Tuple, Pattern

from ...utils.logging import get_logger
from ...utils.types import Document, DocumentSection
from .section_detector import SectionDetector

logger = get_logger(__name__)


class RuleBasedSectionDetector(SectionDetector):
    """Rule-based section detector for scientific papers."""
    
    def __init__(self, **kwargs):
        """Initialize the rule-based section detector.
        
        Args:
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)
        
        # Configure detection patterns
        self.numbered_pattern = kwargs.get(
            "numbered_pattern", 
            r"^\s*(\d+\.?\d*\.?\d*)\s+([A-Z][\w\s:,]+)(?:\n|\s{2,}|\t)"
        )
        
        self.uppercase_pattern = kwargs.get(
            "uppercase_pattern",
            r"^\s*([A-Z][A-Z\s]{3,}[A-Z])(?:\n|\s{2,}|\t)"
        )
        
        self.capitalized_pattern = kwargs.get(
            "capitalized_pattern",
            r"^\s*([A-Z][\w\s:,-]+)(?:\n|\s{2,}|\t)"
        )
        
        self.common_sections = kwargs.get("common_sections", [
            "abstract", "introduction", "background", "method", "methodology",
            "approach", "experiment", "evaluation", "result", "discussion",
            "conclusion", "reference", "appendix", "acknowledgment"
        ])
        
        # Compile patterns
        self._numbered_re = re.compile(self.numbered_pattern, re.MULTILINE)
        self._uppercase_re = re.compile(self.uppercase_pattern, re.MULTILINE)
        self._capitalized_re = re.compile(self.capitalized_pattern, re.MULTILINE)
        
        # Common section regex for case-insensitive searching
        self._common_sections_pattern = r"^\s*((?:" + "|".join(self.common_sections) + r")s?(?::|\.|$))"
        self._common_sections_re = re.compile(self._common_sections_pattern, re.IGNORECASE | re.MULTILINE)
        
        logger.debug(f"Initialized RuleBasedSectionDetector with {len(self.common_sections)} common sections")
    
    def detect_sections(self, document: Document) -> List[DocumentSection]:
        """Detect sections in a document using rule-based patterns.
        
        Args:
            document: Document to analyze
            
        Returns:
            List[DocumentSection]: List of detected sections
        """
        logger.info(f"Detecting sections in document: {document.metadata.title}")
        
        if not document.full_text:
            logger.warning("Document has no full text content")
            return []
        
        text = document.full_text
        
        # First try structured detection (numbered sections)
        sections = self._detect_numbered_sections(text)
        
        # If we found fewer than 3 sections, try uppercase headers
        if len(sections) < 3:
            logger.debug("Few numbered sections found, trying uppercase detection")
            sections = self._detect_uppercase_sections(text)
        
        # If we still have fewer than 3 sections, try capitalized headers
        if len(sections) < 3:
            logger.debug("Few uppercase sections found, trying common section detection")
            sections = self._detect_common_sections(text)
        
        # If we still have fewer than 3 sections, try capitalized headers
        if len(sections) < 3:
            logger.debug("Few common sections found, trying capitalized detection")
            sections = self._detect_capitalized_sections(text)
        
        # If we found sections, combine small ones
        if sections:
            sections = self.combine_sections(sections, self.min_section_length)
            logger.info(f"Detected {len(sections)} sections after combining small ones")
        else:
            # If we still don't have good sections, create a single section
            logger.warning("No sections detected, creating a single document section")
            sections = [
                DocumentSection(
                    title="Document",
                    content=text,
                    section_type="document",
                    confidence=0.5
                )
            ]
        
        return sections
    
    def _detect_numbered_sections(self, text: str) -> List[DocumentSection]:
        """Detect numbered sections in the document text.
        
        Args:
            text: Document text
            
        Returns:
            List[DocumentSection]: List of detected sections
        """
        sections = []
        matches = list(self._numbered_re.finditer(text))
        
        if not matches:
            return []
        
        logger.debug(f"Found {len(matches)} potential numbered section headers")
        
        # Process each match to extract the section content
        for i, match in enumerate(matches):
            section_number = match.group(1)
            section_title = match.group(2).strip()
            start_pos = match.end()
            
            # Get end position (either the next section or end of text)
            end_pos = len(text)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            
            # Determine section type
            section_type, confidence = self.get_section_type(section_title)
            
            # Create section
            section = DocumentSection(
                title=section_title,
                content=section_content,
                section_type=section_type,
                confidence=confidence,
                metadata={"section_number": section_number}
            )
            
            sections.append(section)
        
        logger.info(f"Detected {len(sections)} numbered sections")
        return sections
    
    def _detect_uppercase_sections(self, text: str) -> List[DocumentSection]:
        """Detect uppercase sections in the document text.
        
        Args:
            text: Document text
            
        Returns:
            List[DocumentSection]: List of detected sections
        """
        sections = []
        matches = list(self._uppercase_re.finditer(text))
        
        if not matches:
            return []
        
        logger.debug(f"Found {len(matches)} potential uppercase section headers")
        
        # Process each match to extract the section content
        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # Get end position (either the next section or end of text)
            end_pos = len(text)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            
            # Determine section type
            section_type, confidence = self.get_section_type(section_title)
            
            # Create section
            section = DocumentSection(
                title=section_title,
                content=section_content,
                section_type=section_type,
                confidence=confidence
            )
            
            sections.append(section)
        
        logger.info(f"Detected {len(sections)} uppercase sections")
        return sections
    
    def _detect_common_sections(self, text: str) -> List[DocumentSection]:
        """Detect common section names in the document text.
        
        Args:
            text: Document text
            
        Returns:
            List[DocumentSection]: List of detected sections
        """
        sections = []
        matches = list(self._common_sections_re.finditer(text))
        
        if not matches:
            return []
        
        logger.debug(f"Found {len(matches)} potential common section headers")
        
        # Process each match to extract the section content
        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # Get end position (either the next section or end of text)
            end_pos = len(text)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            
            # Determine section type
            section_type, confidence = self.get_section_type(section_title)
            
            # Skip if section content is too short
            if len(section_content) < self.min_section_length and i < len(matches) - 1:
                continue
            
            # Create section
            section = DocumentSection(
                title=section_title,
                content=section_content,
                section_type=section_type,
                confidence=confidence
            )
            
            sections.append(section)
        
        logger.info(f"Detected {len(sections)} common sections")
        return sections
    
    def _detect_capitalized_sections(self, text: str) -> List[DocumentSection]:
        """Detect capitalized sections in the document text.
        
        Args:
            text: Document text
            
        Returns:
            List[DocumentSection]: List of detected sections
        """
        sections = []
        matches = list(self._capitalized_re.finditer(text))
        
        if not matches:
            return []
        
        logger.debug(f"Found {len(matches)} potential capitalized section headers")
        
        # Filter out false positives
        filtered_matches = []
        for match in matches:
            title = match.group(1).strip()
            # Skip likely false positives
            if len(title) < 3 or title.endswith((".", ",", ":", "-")):
                continue
            filtered_matches.append(match)
        
        logger.debug(f"Filtered to {len(filtered_matches)} likely section headers")
        
        # Process each match to extract the section content
        for i, match in enumerate(filtered_matches):
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # Get end position (either the next section or end of text)
            end_pos = len(text)
            if i < len(filtered_matches) - 1:
                end_pos = filtered_matches[i + 1].start()
            
            # Extract section content
            section_content = text[start_pos:end_pos].strip()
            
            # Determine section type
            section_type, confidence = self.get_section_type(section_title)
            
            # Skip if section content is too short (less likely to be a real section)
            if len(section_content) < self.min_section_length * 2 and i < len(filtered_matches) - 1:
                continue
            
            # Create section
            section = DocumentSection(
                title=section_title,
                content=section_content,
                section_type=section_type,
                confidence=confidence * 0.8  # Reduce confidence for capitalized headers
            )
            
            sections.append(section)
        
        logger.info(f"Detected {len(sections)} capitalized sections")
        return sections 