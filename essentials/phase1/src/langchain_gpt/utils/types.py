"""Type definitions for LangChainGPT."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

from pydantic import BaseModel


class DocumentType(Enum):
    """Type of document."""
    
    PDF = "pdf"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_extension(cls, extension: str) -> "DocumentType":
        """Get document type from file extension.
        
        Args:
            extension: File extension (e.g., ".pdf")
            
        Returns:
            DocumentType: Document type
        """
        extension = extension.lower().lstrip(".")
        
        if extension == "pdf":
            return cls.PDF
        elif extension in ["txt", "text"]:
            return cls.TEXT
        elif extension in ["html", "htm"]:
            return cls.HTML
        elif extension in ["md", "markdown"]:
            return cls.MARKDOWN
        else:
            return cls.UNKNOWN


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    
    title: str = ""
    authors: List[str] = field(default_factory=list)
    date: str = ""
    source: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    pages: int = 0
    file_path: Optional[Path] = None
    file_size: int = 0
    document_id: str = ""
    extraction_date: str = ""
    extraction_method: str = "default"
    doi: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    publication: Dict[str, str] = field(default_factory=dict)
    references: List[Dict] = field(default_factory=list)
    toc: List[Dict] = field(default_factory=list)
    publication_date: Optional[str] = None


@dataclass
class DocumentChunk:
    """Chunk of a document."""
    
    text: str
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class DocumentSection:
    """Section of a document."""
    
    title: str
    content: str
    section_type: str = "other"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List["DocumentSection"] = field(default_factory=list)


@dataclass
class Document:
    """Document with metadata and chunks."""
    
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = field(default_factory=list)
    full_text: str = ""
    sections: List[DocumentSection] = field(default_factory=list)
    document_id: str = ""
    
    def __post_init__(self):
        """Initialize document ID if not provided."""
        if not self.document_id and hasattr(self.metadata, 'document_id'):
            self.document_id = self.metadata.document_id


@dataclass
class Citation:
    """Citation reference in a document."""
    
    text: str
    ref_ids: List[str]
    citation_type: str
    position: int
    confidence: float = 1.0
    authors: List[str] = field(default_factory=list)
    year: str = ""
    span: Optional[Tuple[int, int]] = None


@dataclass
class Reference:
    """Reference entry in a document."""
    
    ref_id: str
    text: str
    reference_type: str
    confidence: float = 1.0
    title: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    span: Optional[Tuple[int, int]] = None


@dataclass
class LinkedCitation:
    """Citation linked to its reference entries."""
    
    citation: Citation
    references: List[Reference]
    confidence: float = 1.0


# Type aliases for clarity
FilePath = Union[str, Path]
DocumentId = str
ChunkId = str
ModelName = str
APIResponse = Dict[str, Any]
TokenCount = int
EmbeddingVector = List[float] 