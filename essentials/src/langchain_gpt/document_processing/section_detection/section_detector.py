"""Base section detector for identifying document sections."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

from ...utils.logging import get_logger
from ...utils.types import Document, DocumentSection

logger = get_logger(__name__)


class SectionDetector(ABC):
    """Base class for section detection in documents."""
    
    def __init__(self, **kwargs):
        """Initialize the section detector.
        
        Args:
            **kwargs: Additional configuration options
        """
        self.min_section_length = kwargs.get("min_section_length", 100)
        self.detect_subsections = kwargs.get("detect_subsections", True)
        self.assign_confidence = kwargs.get("assign_confidence", True)
        
        logger.debug(
            f"Initialized {self.__class__.__name__} with min_section_length={self.min_section_length}, "
            f"detect_subsections={self.detect_subsections}"
        )
    
    @abstractmethod
    def detect_sections(self, document: Document) -> List[DocumentSection]:
        """Detect sections in a document.
        
        Args:
            document: Document to analyze
            
        Returns:
            List[DocumentSection]: List of detected sections
        """
        pass
    
    def get_section_type(self, section_title: str) -> Tuple[str, float]:
        """Determine the type of a section based on its title.
        
        Args:
            section_title: The title of the section
            
        Returns:
            Tuple[str, float]: Section type and confidence score
        """
        # Default implementation with common section types
        section_title = section_title.lower().strip()
        
        # Map of section title keywords to section types
        common_sections = {
            "abstract": "abstract",
            "introduction": "introduction",
            "background": "background",
            "related work": "related_work",
            "literature review": "related_work",
            "method": "methods",
            "methodology": "methods",
            "approach": "methods",
            "experiment": "experiments",
            "evaluation": "evaluation",
            "result": "results",
            "discussion": "discussion",
            "conclusion": "conclusion",
            "reference": "references",
            "appendix": "appendix",
            "acknowledgment": "acknowledgments",
        }
        
        # Try exact match first
        for key, section_type in common_sections.items():
            if section_title == key:
                return section_type, 1.0
        
        # Then try contains match
        for key, section_type in common_sections.items():
            if key in section_title:
                # Calculate confidence based on how closely the title matches
                confidence = len(key) / len(section_title) if len(section_title) > 0 else 0.0
                return section_type, min(confidence + 0.2, 1.0)  # Add a boost for partial matches
        
        # Unknown section type
        return "other", 0.3
    
    def combine_sections(
        self, 
        sections: List[DocumentSection],
        min_length: int = 100
    ) -> List[DocumentSection]:
        """Combine very small sections with the next section.
        
        Args:
            sections: List of sections to process
            min_length: Minimum length for a standalone section
            
        Returns:
            List[DocumentSection]: Merged sections
        """
        if not sections:
            return []
        
        merged_sections = []
        current_section = None
        
        for section in sections:
            if not current_section:
                current_section = section
                continue
                
            # Check if current section is too small
            if len(current_section.content) < min_length:
                # Merge with the next section
                logger.debug(f"Merging small section '{current_section.title}' with '{section.title}'")
                
                merged_content = current_section.content
                if merged_content and not merged_content.endswith("\n"):
                    merged_content += "\n\n"
                merged_content += section.content
                
                # Create a new merged section
                merged_section = DocumentSection(
                    title=section.title,  # Use the title of the larger section
                    content=merged_content,
                    section_type=section.section_type,
                    confidence=section.confidence,
                    metadata=section.metadata or {},
                    subsections=section.subsections or []
                )
                
                current_section = merged_section
            else:
                merged_sections.append(current_section)
                current_section = section
        
        # Add the last section
        if current_section:
            merged_sections.append(current_section)
        
        return merged_sections 