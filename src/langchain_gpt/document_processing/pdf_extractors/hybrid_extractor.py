"""Hybrid PDF extraction module combining multiple extraction methods."""

import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ...utils.errors import DocumentProcessingError
from ...utils.logging import get_logger
from ...utils.types import Document, DocumentChunk, DocumentMetadata, DocumentType, FilePath
from ..pdf_extractor import PDFExtractor
from .pymupdf_extractor import PyMuPDFExtractor

logger = get_logger(__name__)


class HybridPDFExtractor:
    """Extract text and metadata from PDF files using multiple methods."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        """Initialize hybrid PDF extractor.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            **kwargs: Additional configuration options
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create extractors
        self.pymupdf_extractor = PyMuPDFExtractor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        self.pypdf_extractor = PDFExtractor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Advanced extraction options
        self.primary_method = kwargs.get("primary_method", "pymupdf")  # Default to PyMuPDF as primary
        self.fallback_on_failure = kwargs.get("fallback_on_failure", True)
        self.merge_results = kwargs.get("merge_results", True)
        self.extract_tables = kwargs.get("extract_tables", True)
        
        logger.debug(
            f"Initialized HybridPDFExtractor with primary_method={self.primary_method}, "
            f"fallback_on_failure={self.fallback_on_failure}, "
            f"merge_results={self.merge_results}"
        )
    
    def extract_text_from_pdf(self, file_path: FilePath) -> Tuple[str, Dict]:
        """Extract text and metadata from a PDF file using multiple methods.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple[str, Dict]: Extracted text and metadata
            
        Raises:
            DocumentProcessingError: If all extraction methods fail
        """
        start_time = time.time()
        file_path = Path(file_path)
        logger.info(f"Extracting text from PDF using hybrid approach: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise DocumentProcessingError(f"File not found: {file_path}", str(file_path))
        
        # Dictionary to store results from each method
        results = {}
        errors = {}
        
        # Determine the order of extraction methods
        if self.primary_method == "pymupdf":
            methods = [
                ("pymupdf", self.pymupdf_extractor.extract_text_from_pdf),
                ("pypdf", self.pypdf_extractor.extract_text_from_pdf)
            ]
        else:
            methods = [
                ("pypdf", self.pypdf_extractor.extract_text_from_pdf),
                ("pymupdf", self.pymupdf_extractor.extract_text_from_pdf)
            ]
        
        # Try primary method first
        primary_method, primary_func = methods[0]
        logger.info(f"Trying primary extraction method: {primary_method}")
        
        try:
            text, metadata = primary_func(file_path)
            results[primary_method] = (text, metadata)
            logger.info(f"Primary extraction method {primary_method} succeeded")
            
            # If we're not merging results, return the primary method results
            if not self.merge_results:
                return text, metadata
                
        except Exception as e:
            error_msg = f"Primary extraction method {primary_method} failed: {str(e)}"
            logger.warning(error_msg)
            errors[primary_method] = str(e)
            
            # If fallback is disabled, raise the exception
            if not self.fallback_on_failure:
                raise DocumentProcessingError(
                    f"Primary extraction method failed and fallback is disabled: {str(e)}",
                    str(file_path)
                )
        
        # Try fallback method if primary failed or we're merging results
        if primary_method not in results or self.merge_results:
            fallback_method, fallback_func = methods[1]
            logger.info(f"Trying fallback extraction method: {fallback_method}")
            
            try:
                text, metadata = fallback_func(file_path)
                results[fallback_method] = (text, metadata)
                logger.info(f"Fallback extraction method {fallback_method} succeeded")
                
                # If primary method failed, use fallback results
                if primary_method not in results:
                    return text, metadata
                    
            except Exception as e:
                error_msg = f"Fallback extraction method {fallback_method} failed: {str(e)}"
                logger.warning(error_msg)
                errors[fallback_method] = str(e)
                
                # If both methods failed, raise exception with both errors
                if not results:
                    raise DocumentProcessingError(
                        f"All extraction methods failed. Errors: {errors}",
                        str(file_path)
                    )
        
        # If we're merging results and have both, combine them
        if self.merge_results and len(results) > 1:
            logger.info("Merging results from multiple extraction methods")
            return self._merge_extraction_results(results, file_path)
        
        # Otherwise return the result we have (from primary method)
        logger.info(f"Using results from {list(results.keys())[0]}")
        return results[list(results.keys())[0]]
    
    def _merge_extraction_results(
        self,
        results: Dict[str, Tuple[str, Dict]],
        file_path: FilePath
    ) -> Tuple[str, Dict]:
        """Merge results from multiple extraction methods.
        
        Args:
            results: Dictionary of method name to (text, metadata) tuples
            file_path: Path to the source PDF file
            
        Returns:
            Tuple[str, Dict]: Merged text and metadata
        """
        try:
            # Get results from each method
            pymupdf_results = results.get("pymupdf")
            pypdf_results = results.get("pypdf")
            
            merged_metadata = {}
            merged_text = ""
            
            # Merge metadata (prefer PyMuPDF metadata as it's usually more complete)
            if pymupdf_results:
                merged_metadata = pymupdf_results[1].copy()
                # PyMuPDF usually has better text extraction
                merged_text = pymupdf_results[0]
            elif pypdf_results:
                merged_metadata = pypdf_results[1].copy()
                merged_text = pypdf_results[0]
            
            # Update metadata to indicate hybrid extraction
            merged_metadata["extraction_method"] = "hybrid"
            merged_metadata["extraction_methods_used"] = list(results.keys())
            
            # Compare text lengths and choose the longer one (usually more complete)
            if pymupdf_results and pypdf_results:
                pymupdf_text = pymupdf_results[0]
                pypdf_text = pypdf_results[0]
                
                pymupdf_len = len(pymupdf_text)
                pypdf_len = len(pypdf_text)
                
                logger.info(f"Text length comparison - PyMuPDF: {pymupdf_len}, PyPDF: {pypdf_len}")
                
                # Choose the longer text as it likely has more content
                if pypdf_len > pymupdf_len * 1.25:  # If PyPDF text is significantly longer
                    logger.info("Using PyPDF text as it's significantly longer")
                    merged_text = pypdf_text
                    merged_metadata["primary_text_source"] = "pypdf"
                else:
                    # Otherwise prefer PyMuPDF for better formatting
                    merged_metadata["primary_text_source"] = "pymupdf"
            
            logger.info(f"Successfully merged extraction results from multiple methods")
            return merged_text, merged_metadata
            
        except Exception as e:
            logger.error(f"Error merging extraction results: {str(e)}")
            # Fall back to the first method's results
            method = list(results.keys())[0]
            logger.info(f"Falling back to {method} results due to merge error")
            return results[method]
    
    def create_chunks(self, text: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks.
        
        Args:
            text: Full text to split
            
        Returns:
            List[DocumentChunk]: List of text chunks
        """
        # Use the PyMuPDF extractor's chunking method
        return self.pymupdf_extractor.create_chunks(text)
    
    def process_pdf(self, file_path: FilePath) -> Document:
        """Process a PDF file and return a Document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document: Processed document with metadata and chunks
            
        Raises:
            DocumentProcessingError: If PDF processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        logger.info(f"Processing PDF with hybrid extractor: {file_path}")
        
        # Extract text and metadata
        logger.info("Step 1: Extracting text and metadata")
        extraction_start = time.time()
        text, pdf_metadata = self.extract_text_from_pdf(file_path)
        extraction_time = time.time() - extraction_start
        logger.info(f"Extraction completed in {extraction_time:.2f}s")
        
        # Create document metadata
        logger.info("Step 2: Creating document metadata")
        metadata_start = time.time()
        now = datetime.datetime.now().isoformat()
        file_stat = os.stat(file_path)
        
        metadata = DocumentMetadata(
            title=pdf_metadata.get("title", file_path.stem),
            authors=[pdf_metadata.get("author", "")] if pdf_metadata.get("author") else [],
            date=pdf_metadata.get("creationDate", "") or pdf_metadata.get("creation_date", ""),
            source=str(file_path),
            document_type=DocumentType.PDF,
            pages=pdf_metadata.get("pages", 0) or pdf_metadata.get("page_count", 0),
            file_path=file_path,
            file_size=file_stat.st_size,
            extraction_date=now,
            extraction_method=pdf_metadata.get("extraction_method", "hybrid"),
        )
        
        # Add TOC if available
        if "toc" in pdf_metadata:
            metadata.toc = pdf_metadata["toc"]
        
        metadata_time = time.time() - metadata_start
        logger.debug(f"Metadata creation completed in {metadata_time:.2f}s")
        
        # Create chunks
        logger.info("Step 3: Creating document chunks")
        chunking_start = time.time()
        chunks = self.create_chunks(text)
        chunking_time = time.time() - chunking_start
        logger.info(f"Chunking completed in {chunking_time:.2f}s")
        
        # Create document
        logger.info("Step 4: Assembling final document")
        document = Document(
            metadata=metadata,
            chunks=chunks,
            full_text=text,
        )
        
        total_time = time.time() - start_time
        logger.info(f"PDF processing complete: {len(chunks)} chunks created in {total_time:.2f}s")
        
        # Log document statistics
        text_length = len(text) if text else 0
        avg_chunk_size = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        logger.info(f"Document statistics: {text_length} characters, {metadata.pages} pages, {len(chunks)} chunks, {avg_chunk_size:.1f} avg chunk size")
        
        return document 