"""Local PDF extraction module using PyPDF2."""

import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import PyPDF2

from ..utils.errors import DocumentProcessingError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, DocumentMetadata, DocumentType, FilePath

logger = get_logger(__name__)


class PDFExtractor:
    """Extract text and metadata from PDF files using PyPDF2."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize PDF extractor.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(f"Initialized PDFExtractor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
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
        logger.info(f"Extracting text from PDF: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise DocumentProcessingError(f"File not found: {file_path}", str(file_path))
        
        try:
            with open(file_path, "rb") as file:
                logger.debug(f"Loading PDF file: {file_path}")
                load_start = time.time()
                pdf_reader = PyPDF2.PdfReader(file)
                load_time = time.time() - load_start
                logger.debug(f"PDF loaded in {load_time:.2f}s - Pages: {len(pdf_reader.pages)}")
                
                # Extract text from each page
                text = ""
                extraction_start = time.time()
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"Beginning extraction of {total_pages} pages from PDF")
                for i, page in enumerate(pdf_reader.pages):
                    page_start = time.time()
                    page_text = page.extract_text()
                    page_time = time.time() - page_start
                    
                    if page_text:
                        text += page_text + "\n\n"
                    
                    # Log progress periodically or for very large documents
                    if (i + 1) % 10 == 0 or total_pages < 5 or i == total_pages - 1:
                        logger.debug(f"Extracted page {i+1}/{total_pages} ({page_time:.2f}s)")
                
                extraction_time = time.time() - extraction_start
                logger.info(f"Text extraction complete: extracted {len(text)} characters in {extraction_time:.2f}s")
                
                # Extract metadata
                metadata_start = time.time()
                metadata = {
                    "pages": len(pdf_reader.pages),
                }
                
                if pdf_reader.metadata:
                    # Convert PDF metadata to dict
                    for key, value in pdf_reader.metadata.items():
                        if key and value:
                            # Remove the leading slash from PDF metadata keys
                            clean_key = key[1:] if key.startswith("/") else key
                            metadata[clean_key] = value
                
                metadata_time = time.time() - metadata_start
                logger.debug(f"Metadata extraction complete in {metadata_time:.2f}s")
                
                total_time = time.time() - start_time
                logger.info(f"PDF extraction complete in {total_time:.2f}s")
                return text, metadata
                
        except Exception as e:
            msg = f"Failed to extract text from PDF: {str(e)}"
            logger.error(msg)
            raise DocumentProcessingError(msg, str(file_path))
    
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
                # First try to find a double newline
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
        logger.info(f"Processing PDF: {file_path}")
        
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
            title=pdf_metadata.get("Title", file_path.stem),
            authors=[pdf_metadata.get("Author", "")] if pdf_metadata.get("Author") else [],
            date=pdf_metadata.get("CreationDate", ""),
            source=str(file_path),
            document_type=DocumentType.PDF,
            pages=pdf_metadata.get("pages", 0),
            file_path=file_path,
            file_size=file_stat.st_size,
            extraction_date=now,
        )
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