"""PDF extraction module using PyMuPDF (fitz)."""

import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF

from ...utils.errors import DocumentProcessingError
from ...utils.logging import get_logger
from ...utils.types import Document, DocumentChunk, DocumentMetadata, DocumentType, FilePath

logger = get_logger(__name__)


class PyMuPDFExtractor:
    """Extract text and metadata from PDF files using PyMuPDF."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        """Initialize PyMuPDF extractor.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            **kwargs: Additional configuration options
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Advanced extraction options
        self.extract_tables = kwargs.get("extract_tables", True)
        self.extract_images = kwargs.get("extract_images", False)
        self.detect_columns = kwargs.get("detect_columns", True)
        self.preserve_layout = kwargs.get("preserve_layout", True)
        
        logger.debug(
            f"Initialized PyMuPDFExtractor with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, extract_tables={self.extract_tables}, "
            f"extract_images={self.extract_images}, detect_columns={self.detect_columns}"
        )
    
    def extract_text_from_pdf(self, file_path: FilePath) -> Tuple[str, Dict]:
        """Extract text and metadata from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple[str, Dict]: Extracted text and metadata
            
        Raises:
            DocumentProcessingError: If PDF extraction fails
        """
        start_time = time.time()
        file_path = Path(file_path)
        logger.info(f"Extracting text from PDF using PyMuPDF: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise DocumentProcessingError(f"File not found: {file_path}", str(file_path))
        
        try:
            # Open the PDF
            doc = fitz.open(file_path)
            
            # Extract text from each page
            full_text = ""
            page_texts = []
            tables = []
            extraction_start = time.time()
            total_pages = len(doc)
            
            logger.info(f"Beginning extraction of {total_pages} pages from PDF")
            for i, page in enumerate(doc):
                page_start = time.time()
                
                # Extract text with layout preservation if enabled
                if self.preserve_layout:
                    page_text = page.get_text("text")
                else:
                    page_text = page.get_text("text")  # Could use different formats like "html"
                
                # Store page text
                page_texts.append(page_text)
                full_text += page_text + "\n\n"
                
                # Extract tables if enabled
                if self.extract_tables:
                    page_tables = self._extract_tables_from_page(page)
                    if page_tables:
                        tables.extend(page_tables)
                        # Add table text to the full text
                        for table in page_tables:
                            if table.get("text"):
                                full_text += f"\nTable: {table.get('text')}\n\n"
                
                page_time = time.time() - page_start
                
                # Log progress periodically
                if (i + 1) % 10 == 0 or total_pages < 5 or i == total_pages - 1:
                    logger.debug(f"Extracted page {i+1}/{total_pages} ({page_time:.2f}s)")
            
            extraction_time = time.time() - extraction_start
            logger.info(f"Text extraction complete: extracted {len(full_text)} characters in {extraction_time:.2f}s")
            
            # Extract metadata
            metadata_start = time.time()
            metadata = self._extract_metadata(doc)
            
            # Add custom extraction metadata
            metadata.update({
                "pages": len(doc),
                "tables_extracted": len(tables),
                "extraction_method": "pymupdf"
            })
            
            if tables:
                metadata["tables"] = tables
            
            metadata_time = time.time() - metadata_start
            logger.debug(f"Metadata extraction complete in {metadata_time:.2f}s")
            
            # Close the document
            doc.close()
            
            total_time = time.time() - start_time
            logger.info(f"PyMuPDF PDF extraction complete in {total_time:.2f}s")
            return full_text, metadata
                
        except Exception as e:
            msg = f"Failed to extract text from PDF using PyMuPDF: {str(e)}"
            logger.error(msg)
            raise DocumentProcessingError(msg, str(file_path))
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """Extract metadata from PDF document.
        
        Args:
            doc: PyMuPDF document
            
        Returns:
            Dict: Extracted metadata
        """
        metadata = {}
        
        # Get standard PDF metadata
        for key, value in doc.metadata.items():
            if key and value:
                metadata[key] = value
        
        # Get additional document information
        metadata["page_count"] = len(doc)
        
        # Extract document structure information
        toc = doc.get_toc()
        if toc:
            metadata["toc"] = toc
            metadata["has_toc"] = True
        else:
            metadata["has_toc"] = False
        
        return metadata
    
    def _extract_tables_from_page(self, page: fitz.Page) -> List[Dict]:
        """Extract tables from a page.
        
        Args:
            page: PyMuPDF page
            
        Returns:
            List[Dict]: List of extracted tables
        """
        tables = []
        
        try:
            # PyMuPDF doesn't have a built-in table extraction feature
            # This is a placeholder for table detection
            # In production, you might use a dedicated table extraction library
            
            # Basic table detection using rectangles and text positioning
            blocks = page.get_text("dict")["blocks"]
            
            # Find potential table blocks (simplified detection)
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    # Check for grid-like pattern in the lines
                    lines = block.get("lines", [])
                    if len(lines) > 2:
                        # Check for horizontal alignment of words
                        columns = self._detect_table_columns(lines)
                        if columns > 2:  # At least 3 columns to be a table
                            # Extract table text
                            table_text = page.get_text("text", clip=fitz.Rect(block["bbox"]))
                            tables.append({
                                "bbox": block["bbox"],
                                "text": table_text,
                                "columns": columns,
                                "rows": len(lines)
                            })
            
            return tables
        
        except Exception as e:
            logger.warning(f"Error extracting tables: {str(e)}")
            return []
    
    def _detect_table_columns(self, lines: List[Dict]) -> int:
        """Detect the number of columns in a potential table.
        
        Args:
            lines: List of text lines
            
        Returns:
            int: Estimated number of columns
        """
        max_spans = 0
        
        for line in lines:
            if "spans" in line:
                spans = line["spans"]
                max_spans = max(max_spans, len(spans))
        
        return max_spans
    
    def create_chunks(self, text: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks.
        
        Args:
            text: Full text to split
            
        Returns:
            List[DocumentChunk]: List of text chunks
        """
        start_time = time.time()
        logger.info(f"Creating chunks from {len(text)} characters of text")
        
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good break point (newline or space)
            if end < len(text):
                # First try to find a double newline (paragraph)
                break_point = text.rfind("\n\n", start, end)
                if break_point == -1 or break_point <= start:
                    # Then try to find a single newline
                    break_point = text.rfind("\n", start, end)
                if break_point == -1 or break_point <= start:
                    # Then try to find a space
                    break_point = text.rfind(" ", start, end)
                if break_point != -1 and break_point > start:
                    end = break_point
            
            # Create chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=f"chunk-{len(chunks)}",
                )
                chunks.append(chunk)
            
            # Move start position for next chunk, with overlap
            start = end
            if start < len(text):
                # Move back by chunk_overlap characters, but not before the previous start
                start = max(start - self.chunk_overlap, 0)
                
                # Try to find a good start point (newline or space)
                new_start = text.find("\n\n", start)
                if new_start == -1 or new_start >= end:
                    new_start = text.find("\n", start)
                if new_start == -1 or new_start >= end:
                    new_start = text.find(" ", start)
                if new_start != -1 and new_start < end:
                    start = new_start + 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"Created {len(chunks)} chunks in {elapsed_time:.2f}s")
        return chunks
    
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
        logger.info(f"Processing PDF with PyMuPDF: {file_path}")
        
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
            date=pdf_metadata.get("creationDate", ""),
            source=str(file_path),
            document_type=DocumentType.PDF,
            pages=pdf_metadata.get("pages", 0),
            file_path=file_path,
            file_size=file_stat.st_size,
            extraction_date=now,
            extraction_method="pymupdf",
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