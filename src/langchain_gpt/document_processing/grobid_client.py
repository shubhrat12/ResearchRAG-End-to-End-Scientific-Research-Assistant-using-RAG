"""Client for interacting with Grobid for PDF processing and structured extraction."""

import gc
import os
import psutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import requests
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import Timeout, ConnectionError, RequestException

from ..config.settings import get_settings
from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, DocumentMetadata, FilePath

logger = get_logger(__name__)


class GrobidError(LangChainGPTError):
    """Error raised by Grobid operations."""
    
    def __init__(self, message: str = "Grobid error"):
        super().__init__(f"Grobid error: {message}")


class GrobidTimeoutError(GrobidError):
    """Error raised when Grobid operations time out."""
    
    def __init__(self, message: str = "Grobid request timed out"):
        super().__init__(f"Timeout error: {message}")


class GrobidClient:
    """Client for interacting with Grobid service."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        timeout: int = 900,  # Default timeout of 15 minutes (900 seconds) for complex papers
        threads: int = None,
    ):
        """Initialize Grobid client.
        
        Args:
            host: Grobid host address
            port: Grobid port number
            timeout: Request timeout in seconds (default: 900 seconds/15 minutes)
            threads: Number of threads for parallel processing
            
        Raises:
            GrobidError: If Grobid client initialization fails
        """
        settings = get_settings()
        
        # Use settings from the grobid section
        self.host = host or settings.grobid.host
        self.port = port or settings.grobid.port
        # Use provided timeout or environment setting, with a fallback of 900 seconds (15 minutes)
        self.timeout = timeout or settings.grobid.timeout or 900
        self.threads = threads or settings.grobid.threads
        
        logger.debug(f"Initializing Grobid client with timeout: {self.timeout}s")
        self.base_url = f"{self.host}:{self.port}/api"
        
        # Configure session with retry capabilities
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        
        # Test connection
        try:
            self._test_connection()
            logger.info(f"Successfully connected to Grobid at {self.base_url}")
        except Timeout:
            logger.error(f"Timeout connecting to Grobid at {self.base_url}")
            raise GrobidTimeoutError(f"Connection to Grobid at {self.base_url} timed out after {self.timeout}s")
        except ConnectionError as e:
            logger.error(f"Connection error to Grobid: {str(e)}")
            raise GrobidError(f"Failed to connect to Grobid at {self.base_url}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to connect to Grobid: {str(e)}")
            raise GrobidError(f"Failed to connect to Grobid at {self.base_url}: {str(e)}")
    
    def _test_connection(self) -> None:
        """Test connection to Grobid service."""
        try:
            response = self.session.get(
                f"{self.base_url}/isalive",
                timeout=self.timeout
            )
            response.raise_for_status()
        except Timeout:
            raise GrobidTimeoutError(f"Connection test to Grobid timed out after {self.timeout}s")
        except RequestException as e:
            raise GrobidError(f"Grobid connection test failed: {str(e)}")
    
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
    
    def cleanup_memory(self) -> None:
        """Perform garbage collection to free memory."""
        before_mb, before_percent = self.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        
        after_mb, after_percent = self.get_memory_usage()
        freed_mb = before_mb - after_mb
        
        logger.debug(f"Memory cleanup: {freed_mb:.2f}MB freed. Before: {before_mb:.2f}MB, After: {after_mb:.2f}MB")
    
    def process_header_only(
        self,
        pdf_path: FilePath,
        output_dir: Optional[FilePath] = None,
        timeout: Optional[int] = 30,  # Default 30 seconds for header extraction
    ) -> Dict[str, Any]:
        """Process only the header/metadata of a PDF and return structured data.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save XML output (optional)
            timeout: Request timeout in seconds (default: 30 seconds)
            
        Returns:
            Dict[str, Any]: Extracted header metadata
            
        Raises:
            GrobidError: If PDF processing fails
            GrobidTimeoutError: If PDF processing times out
        """
        if not os.path.exists(pdf_path):
            raise GrobidError(f"PDF file not found: {pdf_path}")
        
        req_timeout = timeout or 30  # Use 30 seconds as default for header extraction
        
        try:
            logger.info(f"Processing header only for PDF {pdf_path} (timeout: {req_timeout}s)")
            start_time = time.time()
            
            with open(pdf_path, "rb") as pdf_file:
                files = {
                    "input": (os.path.basename(pdf_path), pdf_file, "application/pdf")
                }
                
                # Send request to processHeaderDocument endpoint
                response = self.session.post(
                    f"{self.base_url}/processHeaderDocument",
                    files=files,
                    timeout=req_timeout
                )
                response.raise_for_status()
                
                # Get XML content
                xml_content = response.text
                
                # Save to file if output directory is provided
                if output_dir:
                    output_path = os.path.join(
                        output_dir,
                        f"{os.path.splitext(os.path.basename(pdf_path))[0]}.header.xml"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as xml_file:
                        xml_file.write(xml_content)
                    logger.info(f"Saved header XML to {output_path}")
                
                # Parse basic header info from XML
                # This is a simplified version - in production, use a proper XML parser
                header_data = self._extract_basic_header_info(xml_content)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Successfully processed header for {pdf_path} in {elapsed_time:.2f}s")
                
                # Memory cleanup after processing
                self.cleanup_memory()
                
                return header_data
                
        except Timeout:
            logger.error(f"Grobid header request timed out for {pdf_path} after {req_timeout}s")
            raise GrobidTimeoutError(f"Grobid header processing for {pdf_path} timed out after {req_timeout}s")
        except ConnectionError as e:
            logger.error(f"Grobid connection error for {pdf_path}: {str(e)}")
            raise GrobidError(f"Grobid connection error for {pdf_path}: {str(e)}")
        except RequestException as e:
            logger.error(f"Grobid header request failed for {pdf_path}: {str(e)}")
            raise GrobidError(f"Grobid header processing failed for {pdf_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing PDF header {pdf_path}: {str(e)}")
            raise GrobidError(f"Unexpected error processing PDF header {pdf_path}: {str(e)}")
    
    def _extract_basic_header_info(self, xml_content: str) -> Dict[str, Any]:
        """Extract basic header information from TEI XML.
        
        This is a simple extraction and should be expanded with proper XML parsing.
        
        Args:
            xml_content: TEI XML content from Grobid
            
        Returns:
            Dict[str, Any]: Extracted header metadata
        """
        # This is a simplified extraction - in production use proper XML parsing
        header_data = {
            "title": "",
            "authors": [],
            "abstract": "",
            "date": "",
            "doi": "",
        }
        
        # Very basic extraction - replace with proper XML parsing
        if "<title " in xml_content and "</title>" in xml_content:
            title_start = xml_content.find("<title ", 0)
            title_text_start = xml_content.find(">", title_start) + 1
            title_end = xml_content.find("</title>", title_text_start)
            if title_start > 0 and title_end > title_start:
                header_data["title"] = xml_content[title_text_start:title_end].strip()
        
        # Extract abstract (simplified)
        if "<abstract>" in xml_content and "</abstract>" in xml_content:
            abstract_start = xml_content.find("<abstract>") + len("<abstract>")
            abstract_end = xml_content.find("</abstract>", abstract_start)
            if abstract_end > abstract_start:
                header_data["abstract"] = xml_content[abstract_start:abstract_end].strip()
                # Clean up XML tags
                header_data["abstract"] = header_data["abstract"].replace("<p>", "").replace("</p>", " ")
        
        return header_data
    
    def process_sections(
        self,
        pdf_path: FilePath,
        output_dir: Optional[FilePath] = None,
        section_timeout: Optional[int] = 60,  # Default 60 seconds per section
        max_sections: Optional[int] = None,  # Maximum number of sections to process
    ) -> Dict[str, Any]:
        """Process PDF sections individually and return structured data.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save section XMLs (optional)
            section_timeout: Timeout for each section processing in seconds
            max_sections: Maximum number of sections to process (None for all)
            
        Returns:
            Dict[str, Any]: Processed sections data
            
        Raises:
            GrobidError: If PDF processing fails
            GrobidTimeoutError: If PDF processing times out
        """
        if not os.path.exists(pdf_path):
            raise GrobidError(f"PDF file not found: {pdf_path}")
        
        try:
            logger.info(f"Processing sections for PDF {pdf_path}")
            start_time = time.time()
            
            # First get structure information
            structure_info = self._get_document_structure(pdf_path, timeout=30)
            
            # Extract section information
            sections = structure_info.get("sections", [])
            if max_sections and len(sections) > max_sections:
                logger.info(f"Limiting processing to {max_sections} of {len(sections)} sections")
                sections = sections[:max_sections]
            
            processed_sections = []
            for i, section in enumerate(sections):
                try:
                    logger.info(f"Processing section {i+1}/{len(sections)}: {section.get('title', 'Untitled')}")
                    section_start = time.time()
                    
                    # For each section, we'd extract the relevant pages and process them
                    # This is simplified as direct section extraction isn't available in base Grobid
                    # In a real implementation, you'd either use custom Grobid endpoints or
                    # extract sections from the full document
                    
                    # Simulated section processing
                    section_data = self._process_section(
                        pdf_path, 
                        section, 
                        timeout=section_timeout, 
                        output_dir=output_dir
                    )
                    
                    section_time = time.time() - section_start
                    logger.info(f"Section {i+1} processed in {section_time:.2f}s")
                    
                    processed_sections.append(section_data)
                    
                    # Clean up memory after each section
                    self.cleanup_memory()
                    
                except GrobidTimeoutError as e:
                    logger.warning(f"Timeout processing section {i+1}: {str(e)}")
                    processed_sections.append({"error": str(e), "section_number": i+1})
                except Exception as e:
                    logger.warning(f"Error processing section {i+1}: {str(e)}")
                    processed_sections.append({"error": str(e), "section_number": i+1})
            
            total_time = time.time() - start_time
            logger.info(f"Completed processing {len(processed_sections)} sections in {total_time:.2f}s")
            
            result = {
                "total_sections": len(sections),
                "processed_sections": len(processed_sections),
                "sections": processed_sections,
                "processing_time": total_time,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF sections for {pdf_path}: {str(e)}")
            raise GrobidError(f"Error processing PDF sections for {pdf_path}: {str(e)}")
    
    def _get_document_structure(
        self, 
        pdf_path: FilePath, 
        timeout: Optional[int] = 30
    ) -> Dict[str, Any]:
        """Get document structure information.
        
        Args:
            pdf_path: Path to PDF file
            timeout: Request timeout in seconds
            
        Returns:
            Dict[str, Any]: Document structure information
            
        Raises:
            GrobidError: If structure extraction fails
        """
        # This is a simplified version. In reality, you'd extract structure info
        # from a full document processing or use a specific Grobid API
        # Here we're simulating the structure extraction
        
        try:
            with open(pdf_path, "rb") as pdf_file:
                files = {
                    "input": (os.path.basename(pdf_path), pdf_file, "application/pdf")
                }
                
                # Here we use the full text processing API to get structure
                response = self.session.post(
                    f"{self.base_url}/processFulltextDocument",
                    files=files,
                    timeout=timeout
                )
                response.raise_for_status()
                
                # In a real implementation, parse the XML to extract section structure
                # For this example, we'll create a simplified structure
                xml_content = response.text
                
                # Very simplified section extraction - in production use a proper XML parser
                sections = []
                section_start_pos = 0
                
                while True:
                    section_tag = "<div xmlns=\"http://www.tei-c.org/ns/1.0\">"
                    section_start = xml_content.find(section_tag, section_start_pos)
                    
                    if section_start == -1:
                        break
                        
                    head_start = xml_content.find("<head>", section_start)
                    head_end = xml_content.find("</head>", head_start) if head_start != -1 else -1
                    
                    section_end = xml_content.find("</div>", section_start)
                    
                    if section_start != -1 and section_end != -1:
                        title = "Untitled Section"
                        if head_start != -1 and head_end != -1 and head_start < section_end:
                            title = xml_content[head_start + 6:head_end].strip()
                        
                        sections.append({
                            "title": title,
                            "start_pos": section_start,
                            "end_pos": section_end,
                        })
                        
                        section_start_pos = section_end
                    else:
                        break
                
                # If no sections were found, create a single section for the whole document
                if not sections:
                    sections.append({
                        "title": "Main Content",
                        "start_pos": 0,
                        "end_pos": len(xml_content),
                    })
                
                return {
                    "filename": os.path.basename(pdf_path),
                    "sections": sections,
                }
                
        except Timeout:
            logger.error(f"Timeout getting document structure for {pdf_path}")
            raise GrobidTimeoutError(f"Timeout getting document structure for {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting document structure for {pdf_path}: {str(e)}")
            raise GrobidError(f"Error extracting document structure for {pdf_path}: {str(e)}")
    
    def _process_section(
        self, 
        pdf_path: FilePath, 
        section: Dict[str, Any], 
        timeout: Optional[int] = 60,
        output_dir: Optional[FilePath] = None,
    ) -> Dict[str, Any]:
        """Process a single section of a PDF.
        
        Args:
            pdf_path: Path to PDF file
            section: Section information (title, page range, etc.)
            timeout: Request timeout in seconds
            output_dir: Directory to save section output
            
        Returns:
            Dict[str, Any]: Processed section data
        """
        # This is a simplified implementation since direct section processing
        # isn't available in standard Grobid
        
        # In a real implementation, you might:
        # 1. Extract only the pages for this section
        # 2. Send those pages to Grobid
        # 3. Or use a custom Grobid endpoint for section processing
        
        section_title = section.get("title", "Untitled Section")
        logger.info(f"Processing section: {section_title}")
        
        # For demonstration, we'll return structured info about the section
        result = {
            "title": section_title,
            "content_summary": f"Processed content for section: {section_title}",
            "processing_time": timeout * 0.5,  # Simulated processing time
        }
        
        # In a real implementation, you'd extract the actual content
        
        return result
    
    def process_pdf(
        self,
        pdf_path: FilePath,
        output_dir: Optional[FilePath] = None,
        consolidate_citations: bool = True,
        consolidate_header: bool = True,
        include_raw_citations: bool = True,
        include_raw_affiliations: bool = True,
        segment_sentences: bool = True,
        timeout: Optional[int] = None,
    ) -> str:
        """Process PDF and return TEI XML.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save XML output (optional)
            consolidate_citations: Whether to consolidate citations
            consolidate_header: Whether to consolidate header
            include_raw_citations: Whether to include raw citations
            include_raw_affiliations: Whether to include raw affiliations
            segment_sentences: Whether to segment sentences
            timeout: Request timeout override in seconds (optional)
            
        Returns:
            str: TEI XML string
            
        Raises:
            GrobidError: If PDF processing fails
            GrobidTimeoutError: If PDF processing times out
        """
        if not os.path.exists(pdf_path):
            raise GrobidError(f"PDF file not found: {pdf_path}")
        
        # Use provided timeout or default instance timeout
        req_timeout = timeout or self.timeout
        
        try:
            logger.info(f"Processing PDF {pdf_path} with Grobid (timeout: {req_timeout}s)")
            start_time = time.time()
            memory_before = self.get_memory_usage()[0]
            
            # Prepare parameters
            params = {
                "consolidateCitations": 1 if consolidate_citations else 0,
                "consolidateHeader": 1 if consolidate_header else 0,
                "includeRawCitations": 1 if include_raw_citations else 0,
                "includeRawAffiliations": 1 if include_raw_affiliations else 0,
                "segmentSentences": 1 if segment_sentences else 0,
            }
            
            with open(pdf_path, "rb") as pdf_file:
                files = {
                    "input": (os.path.basename(pdf_path), pdf_file, "application/pdf")
                }
                
                # Send request
                response = self.session.post(
                    f"{self.base_url}/processFulltextDocument",
                    files=files,
                    data=params,
                    timeout=req_timeout
                )
                response.raise_for_status()
                
                # Get XML content
                xml_content = response.text
                
                # Save to file if output directory is provided
                if output_dir:
                    output_path = os.path.join(
                        output_dir,
                        f"{os.path.splitext(os.path.basename(pdf_path))[0]}.tei.xml"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as xml_file:
                        xml_file.write(xml_content)
                    logger.info(f"Saved TEI XML to {output_path}")
                
                elapsed_time = time.time() - start_time
                memory_after = self.get_memory_usage()[0]
                memory_used = memory_after - memory_before
                
                logger.info(
                    f"Successfully processed PDF: {pdf_path} in {elapsed_time:.2f}s "
                    f"(memory: {memory_used:.2f}MB)"
                )
                
                # Cleanup memory after processing
                self.cleanup_memory()
                
                return xml_content
                
        except Timeout:
            logger.error(f"Grobid request timed out for {pdf_path} after {req_timeout}s")
            raise GrobidTimeoutError(f"Grobid processing for {pdf_path} timed out after {req_timeout}s")
        except ConnectionError as e:
            logger.error(f"Grobid connection error for {pdf_path}: {str(e)}")
            raise GrobidError(f"Grobid connection error for {pdf_path}: {str(e)}")
        except RequestException as e:
            logger.error(f"Grobid request failed for {pdf_path}: {str(e)}")
            raise GrobidError(f"Grobid processing failed for {pdf_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing PDF {pdf_path}: {str(e)}")
            raise GrobidError(f"Unexpected error processing PDF {pdf_path}: {str(e)}")
    
    def process_pdf_batch(
        self,
        pdf_paths: List[FilePath],
        output_dir: Optional[FilePath] = None,
        consolidate_citations: bool = True,
        consolidate_header: bool = True,
        include_raw_citations: bool = True,
        include_raw_affiliations: bool = True,
        segment_sentences: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, str]:
        """Process multiple PDFs and return TEI XMLs.
        
        Args:
            pdf_paths: List of paths to PDF files
            output_dir: Directory to save XML outputs (optional)
            consolidate_citations: Whether to consolidate citations
            consolidate_header: Whether to consolidate header
            include_raw_citations: Whether to include raw citations
            include_raw_affiliations: Whether to include raw affiliations
            segment_sentences: Whether to segment sentences
            timeout: Request timeout override in seconds (optional)
            
        Returns:
            Dict[str, str]: Dictionary mapping PDF paths to TEI XML strings
            
        Raises:
            GrobidError: If PDF processing fails
        """
        results = {}
        total_files = len(pdf_paths)
        processed = 0
        
        logger.info(f"Batch processing {total_files} PDF files with Grobid")
        batch_start_time = time.time()
        
        for pdf_path in pdf_paths:
            try:
                # Process with rate limiting
                start_time = time.time()
                xml = self.process_pdf(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    consolidate_citations=consolidate_citations,
                    consolidate_header=consolidate_header,
                    include_raw_citations=include_raw_citations,
                    include_raw_affiliations=include_raw_affiliations,
                    segment_sentences=segment_sentences,
                    timeout=timeout,
                )
                elapsed = time.time() - start_time
                results[pdf_path] = xml
                processed += 1
                
                logger.info(f"Processed {processed}/{total_files} - {pdf_path} in {elapsed:.2f}s")
                
                # Cleanup memory after each file
                self.cleanup_memory()
                
                # Sleep to avoid overwhelming the Grobid server
                time.sleep(1)
                
            except GrobidTimeoutError as e:
                logger.error(f"Timeout processing {pdf_path}: {str(e)}")
                continue
            except GrobidError as e:
                logger.warning(f"Skipping {pdf_path}: {str(e)}")
                continue
        
        total_elapsed = time.time() - batch_start_time
        logger.info(f"Batch processing complete. Processed {processed}/{total_files} files in {total_elapsed:.2f}s")
        return results
    
    def pdf_to_document(
        self,
        pdf_path: FilePath,
        output_dir: Optional[FilePath] = None,
        chunk_size: int = 1000,
        timeout: Optional[int] = None,
    ) -> Document:
        """Convert PDF to Document model using Grobid.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save XML output (optional)
            chunk_size: Size of document chunks in characters
            timeout: Request timeout override in seconds (optional)
            
        Returns:
            Document: Document model
            
        Raises:
            GrobidError: If PDF processing fails
            GrobidTimeoutError: If PDF processing times out
        """
        start_time = time.time()
        logger.info(f"Converting PDF to Document: {pdf_path}")
        
        # Process PDF with Grobid
        xml_content = self.process_pdf(pdf_path, output_dir, timeout=timeout)
        
        # Parse XML and create Document
        try:
            from .grobid_parser import GrobidParser
            parser = GrobidParser()
            document = parser.parse_xml(xml_content, pdf_path, chunk_size)
            
            elapsed = time.time() - start_time
            logger.info(f"PDF to Document conversion completed in {elapsed:.2f}s")
            
            # Memory cleanup after processing
            self.cleanup_memory()
            
            return document
        except ImportError:
            raise GrobidError("GrobidParser not found. Please implement the parser module.")
        except Exception as e:
            logger.error(f"Failed to parse Grobid XML for {pdf_path}: {str(e)}")
            raise GrobidError(f"Failed to parse Grobid XML for {pdf_path}: {str(e)}") 