"""Document processor for handling various document types."""

import os
import time
from pathlib import Path
from typing import List, Optional, Union

from ..utils.errors import DocumentProcessingError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentType, FilePath
from .pdf_extractor import PDFExtractor
from .text_cleaner import TextCleaner

logger = get_logger(__name__)


class DocumentProcessor:
    """Process documents from various sources."""
    
    def __init__(
        self,
        pdf_extractor: Optional[PDFExtractor] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        clean_text: bool = True,
    ):
        """Initialize document processor.
        
        Args:
            pdf_extractor: PDF extractor instance
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            clean_text: Whether to clean text during processing
        """
        self.pdf_extractor = pdf_extractor or PDFExtractor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.clean_text = clean_text
        self.text_cleaner = TextCleaner()
        logger.info(f"Document processor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def get_document_type(self, file_path: FilePath) -> DocumentType:
        """Get document type from file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            DocumentType: Document type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        doc_type = DocumentType.from_extension(extension)
        logger.debug(f"Document type for {file_path}: {doc_type.value}")
        return doc_type
    
    def process_file(self, file_path: FilePath) -> Document:
        """Process a file and extract its content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Document: Processed document
            
        Raises:
            DocumentProcessingError: If file processing fails
        """
        process_start_time = time.time()
        file_path = Path(file_path)
        
        logger.info(f"Starting processing of file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise DocumentProcessingError(f"File not found: {file_path}", str(file_path))
        
        # Get file size
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Get document type
        document_type = self.get_document_type(file_path)
        logger.info(f"Processing document of type: {document_type}")
        
        # Process based on type
        try:
            if document_type == DocumentType.PDF:
                logger.info(f"Using PDF extractor for: {file_path}")
                extraction_start = time.time()
                document = self.pdf_extractor.process_pdf(file_path)
                extraction_time = time.time() - extraction_start
                logger.info(f"PDF extraction completed in {extraction_time:.2f}s")
                
                # Apply text cleaning if enabled
                if self.clean_text and document.full_text:
                    cleaning_start = time.time()
                    logger.info("Cleaning extracted text")
                    document.full_text = self.text_cleaner.clean_text(document.full_text)
                    
                    # Clean text in chunks as well
                    for i, chunk in enumerate(document.chunks):
                        document.chunks[i].text = self.text_cleaner.clean_text(chunk.text)
                    
                    cleaning_time = time.time() - cleaning_start
                    logger.info(f"Text cleaning completed in {cleaning_time:.2f}s")
            else:
                logger.error(f"Unsupported document type: {document_type.value}")
                raise DocumentProcessingError(
                    f"Unsupported document type: {document_type.value}",
                    str(file_path)
                )
                
            total_time = time.time() - process_start_time
            logger.info(f"Successfully processed file {file_path} in {total_time:.2f}s")
            logger.info(f"Extracted {len(document.chunks)} chunks from document")
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Error processing file: {str(e)}", str(file_path))
    
    def process_directory(
        self,
        directory_path: FilePath,
        file_types: Optional[List[DocumentType]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """Process all files in a directory.
        
        Args:
            directory_path: Path to directory
            file_types: List of document types to process, or None for all supported types
            recursive: Whether to process subdirectories recursively
            
        Returns:
            List[Document]: List of processed documents
            
        Raises:
            DocumentProcessingError: If directory processing fails
        """
        directory_start_time = time.time()
        directory_path = Path(directory_path)
        
        logger.info(f"Starting directory processing: {directory_path} (recursive={recursive})")
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            raise DocumentProcessingError(f"Directory not found: {directory_path}")
        
        # Default to PDF if not specified
        file_types = file_types or [DocumentType.PDF]
        logger.info(f"Processing file types: {[t.value for t in file_types]}")
        
        # Get file extensions
        extensions = [f".{doc_type.value}" for doc_type in file_types]
        
        documents = []
        
        # Collect files
        files = []
        collection_start = time.time()
        if recursive:
            for root, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    file_path = Path(root) / filename
                    if file_path.suffix.lower() in extensions:
                        files.append(file_path)
        else:
            for file_path in directory_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    files.append(file_path)
        
        collection_time = time.time() - collection_start
        logger.info(f"Found {len(files)} files to process in {collection_time:.2f}s")
        
        # Process files
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(files):
            logger.info(f"Processing file {i+1}/{len(files)}: {file_path}")
            try:
                file_start_time = time.time()
                document = self.process_file(file_path)
                file_time = time.time() - file_start_time
                
                documents.append(document)
                successful += 1
                
                logger.info(f"File {i+1}/{len(files)} processed successfully in {file_time:.2f}s")
            except DocumentProcessingError as e:
                failed += 1
                logger.error(f"Error processing file {file_path}: {str(e)}")
                # Continue processing other files
        
        total_time = time.time() - directory_start_time
        logger.info(
            f"Directory processing complete. Processed {successful} files successfully, "
            f"{failed} files failed. Total time: {total_time:.2f}s"
        )
        
        return documents
    
    def process_sample(self, sample_directory: Optional[FilePath] = None) -> List[Document]:
        """Process sample documents for testing.
        
        Args:
            sample_directory: Path to sample documents directory
            
        Returns:
            List[Document]: List of processed documents
        """
        start_time = time.time()
        
        if sample_directory is None:
            # Use default sample directory
            sample_directory = Path("data/samples")
        else:
            sample_directory = Path(sample_directory)
        
        logger.info(f"Processing sample documents from {sample_directory}")
        
        if not sample_directory.exists():
            logger.warning(f"Sample directory not found: {sample_directory}. Creating it.")
            sample_directory.mkdir(parents=True, exist_ok=True)
            
            # Return empty list if no samples are available
            logger.info("No sample documents available.")
            return []
        
        documents = self.process_directory(sample_directory)
        
        total_time = time.time() - start_time
        logger.info(f"Sample processing complete. Processed {len(documents)} documents in {total_time:.2f}s")
        
        return documents 