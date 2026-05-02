"""Factory for selecting and creating PDF extraction methods."""

from enum import Enum
from typing import Optional

from ..utils.logging import get_logger
from .pdf_extractor import PDFExtractor
from .pdf_extractors.pymupdf_extractor import PyMuPDFExtractor
from .pdf_extractors.hybrid_extractor import HybridPDFExtractor

logger = get_logger(__name__)


class PDFExtractionMethod(Enum):
    """Supported PDF extraction methods."""
    
    PYPDF = "pypdf"
    PYMUPDF = "pymupdf"
    HYBRID = "hybrid"
    AUTO = "auto"


class PDFExtractorFactory:
    """Factory for selecting and creating PDF extraction methods."""
    
    @staticmethod
    def create(
        method: str = "auto",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        """Create a PDF extractor instance based on the specified method.
        
        Args:
            method: The extraction method to use (pymupdf, pypdf, hybrid, auto)
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            **kwargs: Additional arguments passed to the extractor
            
        Returns:
            An instance of a PDF extractor
        """
        method = method.lower()
        logger.info(f"Creating PDF extractor with method: {method}")
        
        if method == PDFExtractionMethod.PYPDF.value:
            logger.debug("Using PyPDF extractor")
            return PDFExtractor(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
        elif method == PDFExtractionMethod.PYMUPDF.value:
            logger.debug("Using PyMuPDF extractor")
            return PyMuPDFExtractor(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
        elif method == PDFExtractionMethod.HYBRID.value:
            logger.debug("Using Hybrid PDF extractor")
            return HybridPDFExtractor(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
        elif method == PDFExtractionMethod.AUTO.value:
            logger.debug("Using Auto-select PDF extractor (defaulting to Hybrid)")
            # Auto will use Hybrid by default as it provides the best accuracy
            return HybridPDFExtractor(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
        else:
            logger.warning(f"Unknown extraction method: {method}, defaulting to PyPDF")
            return PDFExtractor(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs) 