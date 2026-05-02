"""Staged document processor for handling complex scientific papers efficiently."""

import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import psutil

from ..utils.errors import DocumentProcessingError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, DocumentMetadata, DocumentType, FilePath
from .document_processor import DocumentProcessor
from .pdf_extractor import PDFExtractor
from .grobid_client import GrobidClient, GrobidTimeoutError, GrobidError
from .text_cleaner import TextCleaner

logger = get_logger(__name__)


class StagedProcessingError(DocumentProcessingError):
    """Error raised during staged document processing."""
    
    def __init__(self, message: str, file_path: str = None, stage: str = None):
        error_msg = f"Staged processing error"
        if stage:
            error_msg += f" at stage '{stage}'"
        error_msg += f": {message}"
        super().__init__(error_msg, file_path)
        self.stage = stage


class ProcessingStageResult:
    """Result of a processing stage."""
    
    def __init__(
        self,
        stage_name: str,
        success: bool,
        data: Any = None,
        error: Optional[Exception] = None,
        processing_time: float = 0.0,
        memory_usage: float = 0.0,
    ):
        """Initialize a processing stage result.
        
        Args:
            stage_name: Name of the processing stage
            success: Whether the stage was successful
            data: The data produced by the stage (if successful)
            error: The error that occurred (if unsuccessful)
            processing_time: Time taken to process the stage in seconds
            memory_usage: Memory used during processing in MB
        """
        self.stage_name = stage_name
        self.success = success
        self.data = data
        self.error = error
        self.processing_time = processing_time
        self.memory_usage = memory_usage
    
    def __str__(self) -> str:
        """Get string representation of the result."""
        status = "SUCCESS" if self.success else "FAILURE"
        error_msg = f": {self.error}" if self.error else ""
        return (
            f"Stage '{self.stage_name}' - {status}{error_msg} "
            f"(Time: {self.processing_time:.2f}s, Memory: {self.memory_usage:.2f}MB)"
        )


class StagedDocumentProcessor:
    """Process complex scientific documents in stages to optimize memory and robustness."""
    
    def __init__(
        self,
        grobid_client: Optional[GrobidClient] = None,
        pdf_extractor: Optional[PDFExtractor] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        clean_text: bool = True,
        header_timeout: int = 30,  # Default 30 seconds for header extraction
        section_timeout: int = 60,  # Default 60 seconds per section
        max_sections: Optional[int] = None,  # Maximum number of sections to process
    ):
        """Initialize staged document processor.
        
        Args:
            grobid_client: Grobid client instance
            pdf_extractor: PDF extractor for fallback processing
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            clean_text: Whether to clean text during processing
            header_timeout: Timeout for header extraction in seconds
            section_timeout: Timeout for section processing in seconds
            max_sections: Maximum number of sections to process (None for all)
        """
        self.grobid_client = grobid_client or GrobidClient(timeout=900)  # 15 minute default timeout
        self.pdf_extractor = pdf_extractor or PDFExtractor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.clean_text = clean_text
        self.text_cleaner = TextCleaner()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_timeout = header_timeout
        self.section_timeout = section_timeout
        self.max_sections = max_sections
        
        # Create fallback processor
        self.fallback_processor = DocumentProcessor(
            pdf_extractor=self.pdf_extractor,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clean_text=clean_text,
        )
        
        logger.info(
            f"Staged document processor initialized with: "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"header_timeout={header_timeout}s, section_timeout={section_timeout}s"
        )
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage.
        
        Returns:
            Tuple[float, float]: A tuple of (used_memory_mb, percent_used)
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        used_memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        percent_used = process.memory_percent()
        
        return used_memory_mb, percent_used
    
    def cleanup_memory(self) -> Tuple[float, float]:
        """Perform garbage collection to free memory.
        
        Returns:
            Tuple[float, float]: A tuple of (freed_memory_mb, current_memory_mb)
        """
        before_mb, _ = self.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        
        after_mb, _ = self.get_memory_usage()
        freed_mb = before_mb - after_mb
        
        logger.debug(f"Memory cleanup: {freed_mb:.2f}MB freed. Before: {before_mb:.2f}MB, After: {after_mb:.2f}MB")
        return freed_mb, after_mb
    
    def process_file(
        self, 
        file_path: FilePath,
        output_dir: Optional[FilePath] = None,
        use_fallback: bool = True,
    ) -> Document:
        """Process a file in stages to optimize memory and robustness.
        
        Args:
            file_path: Path to file
            output_dir: Directory to save intermediate outputs
            use_fallback: Whether to use fallback processing if stages fail
            
        Returns:
            Document: Processed document
            
        Raises:
            StagedProcessingError: If file processing fails
        """
        process_start_time = time.time()
        file_path = Path(file_path)
        
        logger.info(f"Starting staged processing of file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise StagedProcessingError(f"File not found: {file_path}", str(file_path))
        
        # Get file size
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Track processing stages
        stages: List[ProcessingStageResult] = []
        
        # Check document type
        doc_type_result = self._execute_stage(
            "document_type_detection",
            lambda: self._detect_document_type(file_path)
        )
        stages.append(doc_type_result)
        
        if not doc_type_result.success:
            logger.error(f"Failed to detect document type: {doc_type_result.error}")
            raise StagedProcessingError(
                f"Failed to detect document type: {doc_type_result.error}",
                str(file_path),
                "document_type_detection"
            )
        
        document_type = doc_type_result.data
        
        if document_type != DocumentType.PDF:
            logger.error(f"Unsupported document type: {document_type}")
            raise StagedProcessingError(
                f"Unsupported document type: {document_type}",
                str(file_path),
                "document_type_detection"
            )
        
        # STAGE 1: Process header only
        header_result = self._execute_stage(
            "header_extraction",
            lambda: self._process_header(file_path, output_dir, self.header_timeout)
        )
        stages.append(header_result)
        
        if not header_result.success:
            logger.warning(
                f"Header extraction failed: {header_result.error}. "
                f"{'Using fallback processing.' if use_fallback else 'Stopping processing.'}"
            )
            if use_fallback:
                return self._use_fallback_processing(file_path, stages, process_start_time)
        
        # STAGE 2: Process sections
        sections_result = self._execute_stage(
            "section_extraction",
            lambda: self._process_sections(
                file_path, 
                output_dir, 
                self.section_timeout, 
                self.max_sections
            )
        )
        stages.append(sections_result)
        
        if not sections_result.success:
            logger.warning(
                f"Section extraction failed: {sections_result.error}. "
                f"{'Using fallback processing.' if use_fallback else 'Using header data only.'}"
            )
            if use_fallback:
                return self._use_fallback_processing(file_path, stages, process_start_time)
        
        # STAGE 3: Combine results and create document
        document_result = self._execute_stage(
            "document_creation",
            lambda: self._create_document(
                file_path, 
                header_result.data, 
                sections_result.data if sections_result.success else None
            )
        )
        stages.append(document_result)
        
        if not document_result.success:
            logger.error(f"Document creation failed: {document_result.error}")
            if use_fallback:
                return self._use_fallback_processing(file_path, stages, process_start_time)
            else:
                raise StagedProcessingError(
                    f"Document creation failed: {document_result.error}",
                    str(file_path),
                    "document_creation"
                )
        
        document = document_result.data
        
        # Final cleanup
        self.cleanup_memory()
        
        # Log overall processing results
        total_time = time.time() - process_start_time
        successful_stages = sum(1 for stage in stages if stage.success)
        
        logger.info(
            f"Staged processing complete for {file_path}: "
            f"{successful_stages}/{len(stages)} stages successful, "
            f"created {len(document.chunks)} chunks in {total_time:.2f}s"
        )
        
        # Log detailed stage results
        for stage in stages:
            logger.debug(f"Stage result: {stage}")
        
        return document
    
    def _execute_stage(
        self, 
        stage_name: str, 
        stage_function: callable
    ) -> ProcessingStageResult:
        """Execute a processing stage with timing and memory tracking.
        
        Args:
            stage_name: Name of the processing stage
            stage_function: Function to execute
            
        Returns:
            ProcessingStageResult: Result of the processing stage
        """
        logger.info(f"Starting stage: {stage_name}")
        
        start_time = time.time()
        start_memory, _ = self.get_memory_usage()
        
        try:
            result = stage_function()
            
            end_time = time.time()
            end_memory, _ = self.get_memory_usage()
            memory_used = end_memory - start_memory
            processing_time = end_time - start_time
            
            stage_result = ProcessingStageResult(
                stage_name=stage_name,
                success=True,
                data=result,
                processing_time=processing_time,
                memory_usage=memory_used,
            )
            
            logger.info(
                f"Stage '{stage_name}' completed successfully in {processing_time:.2f}s "
                f"(Memory: {memory_used:.2f}MB)"
            )
            
            # Cleanup after stage
            freed_memory, current_memory = self.cleanup_memory()
            logger.debug(
                f"After stage '{stage_name}', freed {freed_memory:.2f}MB, "
                f"current memory usage: {current_memory:.2f}MB"
            )
            
            return stage_result
            
        except Exception as e:
            end_time = time.time()
            end_memory, _ = self.get_memory_usage()
            memory_used = end_memory - start_memory
            processing_time = end_time - start_time
            
            stage_result = ProcessingStageResult(
                stage_name=stage_name,
                success=False,
                error=e,
                processing_time=processing_time,
                memory_usage=memory_used,
            )
            
            logger.error(
                f"Stage '{stage_name}' failed in {processing_time:.2f}s: {str(e)} "
                f"(Memory: {memory_used:.2f}MB)"
            )
            
            # Cleanup after failure
            self.cleanup_memory()
            
            return stage_result
    
    def _detect_document_type(self, file_path: FilePath) -> DocumentType:
        """Detect document type from file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            DocumentType: Detected document type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        doc_type = DocumentType.from_extension(extension)
        logger.debug(f"Document type for {file_path}: {doc_type}")
        return doc_type
    
    def _process_header(
        self, 
        file_path: FilePath, 
        output_dir: Optional[FilePath] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Process document header only.
        
        Args:
            file_path: Path to file
            output_dir: Directory to save output
            timeout: Timeout for header processing in seconds
            
        Returns:
            Dict[str, Any]: Header data
            
        Raises:
            StagedProcessingError: If header processing fails
        """
        logger.info(f"Processing header for {file_path} (timeout: {timeout}s)")
        
        try:
            header_data = self.grobid_client.process_header_only(
                file_path, 
                output_dir=output_dir,
                timeout=timeout
            )
            
            logger.info(f"Successfully extracted header data: {list(header_data.keys())}")
            return header_data
            
        except GrobidTimeoutError as e:
            logger.error(f"Header extraction timed out: {str(e)}")
            raise StagedProcessingError(f"Header extraction timed out: {str(e)}", str(file_path), "header_extraction")
        except GrobidError as e:
            logger.error(f"Header extraction failed: {str(e)}")
            raise StagedProcessingError(f"Header extraction failed: {str(e)}", str(file_path), "header_extraction")
        except Exception as e:
            logger.error(f"Unexpected error in header extraction: {str(e)}")
            raise StagedProcessingError(f"Unexpected error in header extraction: {str(e)}", str(file_path), "header_extraction")
    
    def _process_sections(
        self, 
        file_path: FilePath, 
        output_dir: Optional[FilePath] = None,
        section_timeout: int = 60,
        max_sections: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process document sections.
        
        Args:
            file_path: Path to file
            output_dir: Directory to save output
            section_timeout: Timeout for section processing in seconds
            max_sections: Maximum number of sections to process
            
        Returns:
            Dict[str, Any]: Sections data
            
        Raises:
            StagedProcessingError: If section processing fails
        """
        logger.info(
            f"Processing sections for {file_path} "
            f"(timeout: {section_timeout}s, max_sections: {max_sections or 'all'})"
        )
        
        try:
            sections_data = self.grobid_client.process_sections(
                file_path, 
                output_dir=output_dir,
                section_timeout=section_timeout,
                max_sections=max_sections
            )
            
            processed_count = sections_data.get("processed_sections", 0)
            total_count = sections_data.get("total_sections", 0)
            
            logger.info(f"Successfully processed {processed_count}/{total_count} sections")
            return sections_data
            
        except GrobidTimeoutError as e:
            logger.error(f"Section processing timed out: {str(e)}")
            raise StagedProcessingError(f"Section processing timed out: {str(e)}", str(file_path), "section_extraction")
        except GrobidError as e:
            logger.error(f"Section processing failed: {str(e)}")
            raise StagedProcessingError(f"Section processing failed: {str(e)}", str(file_path), "section_extraction")
        except Exception as e:
            logger.error(f"Unexpected error in section processing: {str(e)}")
            raise StagedProcessingError(f"Unexpected error in section processing: {str(e)}", str(file_path), "section_extraction")
    
    def _create_document(
        self, 
        file_path: FilePath, 
        header_data: Dict[str, Any],
        sections_data: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Create document from header and section data.
        
        Args:
            file_path: Path to file
            header_data: Header data
            sections_data: Sections data
            
        Returns:
            Document: Created document
            
        Raises:
            StagedProcessingError: If document creation fails
        """
        logger.info(f"Creating document from staged data for {file_path}")
        
        try:
            # Create metadata
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            file_stat = os.stat(file_path)
            
            metadata = DocumentMetadata(
                title=header_data.get("title", Path(file_path).stem),
                authors=header_data.get("authors", []),
                date=header_data.get("date", ""),
                source=str(file_path),
                document_type=DocumentType.PDF,
                pages=0,  # Will be updated if available
                file_path=file_path,
                file_size=file_stat.st_size,
                extraction_date=now,
                abstract=header_data.get("abstract", ""),
            )
            
            # Collect text from sections
            all_text = header_data.get("abstract", "")
            chunks = []
            
            if sections_data and "sections" in sections_data:
                for i, section in enumerate(sections_data["sections"]):
                    if "error" in section:
                        logger.warning(f"Section {i+1} had processing error: {section['error']}")
                        continue
                    
                    section_title = section.get("title", f"Section {i+1}")
                    section_content = section.get("content_summary", "")
                    
                    if section_content:
                        section_text = f"{section_title}\n\n{section_content}"
                        all_text += "\n\n" + section_text
            
            # Create chunks if we have text
            if all_text:
                chunks = self._create_chunks(all_text)
            
            # If no chunks created, try to extract basic text from header
            if not chunks:
                logger.warning("No text chunks created from sections, using header data only")
                chunks = [
                    DocumentChunk(
                        text=f"Title: {metadata.title}\n\nAbstract: {header_data.get('abstract', '')}",
                        chunk_id="chunk-0"
                    )
                ]
            
            document = Document(
                metadata=metadata,
                chunks=chunks,
                full_text=all_text,
            )
            
            logger.info(f"Created document with {len(chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Failed to create document: {str(e)}")
            raise StagedProcessingError(f"Failed to create document: {str(e)}", str(file_path), "document_creation")
    
    def _create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create chunks from text.
        
        Args:
            text: Text to chunk
            
        Returns:
            List[DocumentChunk]: Created chunks
        """
        if not text:
            return []
        
        # Clean text if enabled
        if self.clean_text:
            text = self.text_cleaner.clean_text(text)
        
        # Use PDF extractor's chunk creation
        return self.pdf_extractor.create_chunks(text)
    
    def _use_fallback_processing(
        self, 
        file_path: FilePath, 
        stages: List[ProcessingStageResult],
        start_time: float
    ) -> Document:
        """Use fallback processor when staged processing fails.
        
        Args:
            file_path: Path to file
            stages: List of completed stages
            start_time: Start time of processing
            
        Returns:
            Document: Processed document
            
        Raises:
            StagedProcessingError: If fallback processing fails
        """
        logger.warning(f"Using fallback processing for {file_path}")
        
        fallback_result = self._execute_stage(
            "fallback_processing",
            lambda: self.fallback_processor.process_file(file_path)
        )
        
        stages.append(fallback_result)
        
        if not fallback_result.success:
            logger.error(f"Fallback processing failed: {fallback_result.error}")
            raise StagedProcessingError(
                f"Fallback processing failed: {fallback_result.error}",
                str(file_path),
                "fallback_processing"
            )
        
        document = fallback_result.data
        
        total_time = time.time() - start_time
        logger.info(
            f"Fallback processing complete for {file_path}: "
            f"created {len(document.chunks)} chunks in {total_time:.2f}s"
        )
        
        return document
    
    def process_directory(
        self,
        directory_path: FilePath,
        file_types: Optional[List[DocumentType]] = None,
        recursive: bool = True,
        output_dir: Optional[FilePath] = None,
        use_fallback: bool = True,
    ) -> List[Document]:
        """Process all files in a directory using staged processing.
        
        Args:
            directory_path: Path to directory
            file_types: List of document types to process, or None for all supported types
            recursive: Whether to process subdirectories recursively
            output_dir: Directory to save intermediate outputs
            use_fallback: Whether to use fallback processing if stages fail
            
        Returns:
            List[Document]: List of processed documents
            
        Raises:
            StagedProcessingError: If directory processing fails
        """
        directory_start_time = time.time()
        directory_path = Path(directory_path)
        
        logger.info(f"Starting staged directory processing: {directory_path} (recursive={recursive})")
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            raise StagedProcessingError(f"Directory not found: {directory_path}", str(directory_path))
        
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
                document = self.process_file(file_path, output_dir, use_fallback)
                file_time = time.time() - file_start_time
                
                documents.append(document)
                successful += 1
                
                logger.info(f"File {i+1}/{len(files)} processed successfully in {file_time:.2f}s")
                
                # Memory cleanup after each file
                self.cleanup_memory()
                
            except StagedProcessingError as e:
                failed += 1
                logger.error(f"Error processing file {file_path}: {str(e)}")
                # Continue processing other files
            except Exception as e:
                failed += 1
                logger.error(f"Unexpected error processing file {file_path}: {str(e)}")
                # Continue processing other files
        
        total_time = time.time() - directory_start_time
        logger.info(
            f"Directory processing complete. Processed {successful} files successfully, "
            f"{failed} files failed. Total time: {total_time:.2f}s"
        )
        
        return documents 