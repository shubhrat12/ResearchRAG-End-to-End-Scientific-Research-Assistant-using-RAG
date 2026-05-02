"""Section classifier for scientific paper sections."""

import re
from typing import Dict, List, Optional, Tuple, Any

from ...utils.logging import get_logger
from ...utils.types import Document, DocumentSection

logger = get_logger(__name__)


class SectionClassifier:
    """Classify document sections based on content and structure."""
    
    def __init__(self, **kwargs):
        """Initialize the section classifier.
        
        Args:
            **kwargs: Additional configuration options
        """
        # Configure classification options
        self.use_keywords = kwargs.get("use_keywords", True)
        self.use_position = kwargs.get("use_position", True)
        self.min_confidence = kwargs.get("min_confidence", 0.6)
        
        # Build keyword dictionaries for section types
        self._build_keyword_dictionaries()
        
        logger.debug(
            f"Initialized SectionClassifier with use_keywords={self.use_keywords}, "
            f"use_position={self.use_position}"
        )
    
    def _build_keyword_dictionaries(self):
        """Build keyword dictionaries for different section types."""
        self.keyword_dict = {
            "abstract": [
                "abstract", "summary", "overview", "synopsis"
            ],
            "introduction": [
                "introduction", "background", "motivation", "context", "overview"
            ],
            "related_work": [
                "related work", "literature review", "previous work", "state of the art",
                "prior research", "related research"
            ],
            "methods": [
                "method", "methodology", "approach", "design", "procedure", "algorithm",
                "implementation", "technique", "process", "system"
            ],
            "experiments": [
                "experiment", "evaluation", "test", "assessment", "validation", "empirical",
                "study", "studies", "analysis", "investigate"
            ],
            "results": [
                "result", "finding", "observation", "outcome", "performance", "metric"
            ],
            "discussion": [
                "discussion", "analysis", "interpretation", "implication", "significance"
            ],
            "conclusion": [
                "conclusion", "summary", "future work", "limitation", "implication",
                "recommendation", "future direction"
            ],
            "references": [
                "reference", "bibliography", "citation", "literature cited", "works cited"
            ]
        }
        
        # Build regex patterns for each section type
        self.patterns = {}
        for section_type, keywords in self.keyword_dict.items():
            pattern = r"\b(" + "|".join(keywords) + r")s?\b"
            self.patterns[section_type] = re.compile(pattern, re.IGNORECASE)
    
    def classify_section(self, section: DocumentSection, position: float = 0.0) -> Tuple[str, float]:
        """Classify a document section based on its content.
        
        Args:
            section: Document section to classify
            position: Relative position in document (0.0 to 1.0)
            
        Returns:
            Tuple[str, float]: Section type and confidence score
        """
        # Start with existing classification if available
        if section.section_type and section.section_type != "other" and section.confidence > self.min_confidence:
            return section.section_type, section.confidence
        
        # Get title-based classification
        title_type, title_confidence = self._classify_by_title(section.title)
        
        # Get content-based classification
        content_type, content_confidence = self._classify_by_content(section.content)
        
        # Get position-based classification
        position_type, position_confidence = self._classify_by_position(position)
        
        # Combine classifications with weights
        # Title has highest weight, then content, then position
        weights = {"title": 0.6, "content": 0.3, "position": 0.1}
        
        # Track scores for each section type
        scores = {}
        
        # Add title score
        if title_type:
            scores[title_type] = scores.get(title_type, 0) + title_confidence * weights["title"]
        
        # Add content score
        if content_type:
            scores[content_type] = scores.get(content_type, 0) + content_confidence * weights["content"]
        
        # Add position score
        if position_type and self.use_position:
            scores[position_type] = scores.get(position_type, 0) + position_confidence * weights["position"]
        
        # Get type with highest score
        if not scores:
            return "other", 0.5
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, confidence
    
    def _classify_by_title(self, title: str) -> Tuple[str, float]:
        """Classify section based on its title.
        
        Args:
            title: Section title
            
        Returns:
            Tuple[str, float]: Section type and confidence
        """
        if not title:
            return "", 0.0
        
        title = title.lower()
        
        # Check for exact matches first
        for section_type, keywords in self.keyword_dict.items():
            if title in keywords:
                return section_type, 1.0
            
            # Check if title starts with any keyword
            for keyword in keywords:
                if title.startswith(keyword):
                    return section_type, 0.9
        
        # Check for partial matches
        best_match = ""
        best_confidence = 0.0
        
        for section_type, keywords in self.keyword_dict.items():
            for keyword in keywords:
                if keyword in title:
                    # Calculate similarity based on keyword length vs title length
                    similarity = len(keyword) / len(title) if len(title) > 0 else 0
                    confidence = min(0.8, 0.4 + similarity)
                    
                    if confidence > best_confidence:
                        best_match = section_type
                        best_confidence = confidence
        
        return best_match, best_confidence
    
    def _classify_by_content(self, content: str) -> Tuple[str, float]:
        """Classify section based on its content.
        
        Args:
            content: Section content text
            
        Returns:
            Tuple[str, float]: Section type and confidence
        """
        if not content or not self.use_keywords:
            return "", 0.0
        
        # Use only first 500 chars for efficient classification
        sample = content[:500].lower()
        
        # Count keyword matches for each section type
        matches = {}
        
        for section_type, pattern in self.patterns.items():
            # Find all matches in content sample
            found = pattern.findall(sample)
            if found:
                matches[section_type] = len(found)
        
        # If no matches found
        if not matches:
            return "", 0.0
        
        # Get section type with most matches
        best_type = max(matches, key=matches.get)
        match_count = matches[best_type]
        
        # Calculate confidence based on match count
        confidence = min(0.7, 0.3 + (match_count * 0.1))
        
        return best_type, confidence
    
    def _classify_by_position(self, position: float) -> Tuple[str, float]:
        """Classify section based on its position in the document.
        
        Args:
            position: Relative position in document (0.0 to 1.0)
            
        Returns:
            Tuple[str, float]: Section type and confidence
        """
        if position < 0.05:
            return "abstract", 0.7
        elif position < 0.15:
            return "introduction", 0.6
        elif position < 0.25:
            return "related_work", 0.5
        elif position < 0.5:
            return "methods", 0.5
        elif position < 0.7:
            return "results", 0.5
        elif position < 0.85:
            return "discussion", 0.6
        else:
            return "conclusion", 0.7
    
    def classify_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Classify a list of document sections.
        
        Args:
            sections: List of document sections
            
        Returns:
            List[DocumentSection]: List of classified sections
        """
        if not sections:
            return []
        
        classified_sections = []
        total_sections = len(sections)
        
        for i, section in enumerate(sections):
            # Calculate relative position in document
            position = i / total_sections
            
            # Classify section
            section_type, confidence = self.classify_section(section, position)
            
            # Only update if confidence is higher than existing classification
            if not section.section_type or section.section_type == "other" or confidence > section.confidence:
                # Create new section with classification
                classified_section = DocumentSection(
                    title=section.title,
                    content=section.content,
                    section_type=section_type,
                    confidence=confidence,
                    metadata=section.metadata or {},
                    subsections=section.subsections or []
                )
                classified_sections.append(classified_section)
            else:
                # Keep existing classification
                classified_sections.append(section)
        
        return classified_sections 