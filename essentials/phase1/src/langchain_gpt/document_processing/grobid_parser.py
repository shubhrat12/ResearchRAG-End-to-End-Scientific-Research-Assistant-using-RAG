"""Parser for GROBID TEI XML output to Document models."""

import os
import re
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET

from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, DocumentMetadata, FilePath

logger = get_logger(__name__)

# Define namespace for TEI XML
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class GrobidParserError(LangChainGPTError):
    """Error raised by Grobid parser operations."""
    
    def __init__(self, message: str = "Grobid parser error"):
        super().__init__(f"Grobid parser error: {message}")


class GrobidParser:
    """Parser for GROBID TEI XML output."""
    
    def __init__(self):
        """Initialize Grobid parser."""
        pass
    
    def parse_xml(
        self,
        xml_content: str,
        pdf_path: Optional[FilePath] = None,
        chunk_size: int = 1000,
    ) -> Document:
        """Parse TEI XML content into a Document model.
        
        Args:
            xml_content: TEI XML content from Grobid
            pdf_path: Path to the original PDF file (optional)
            chunk_size: Size of document chunks in characters
            
        Returns:
            Document: Document model
            
        Raises:
            GrobidParserError: If XML parsing fails
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract document metadata
            metadata = self._extract_metadata(root, pdf_path)
            
            # Extract document content and create chunks
            chunks = self._extract_content_chunks(root, chunk_size, metadata)
            
            # Create Document model
            document = Document(
                document_id=metadata.document_id,
                metadata=metadata,
                chunks=chunks
            )
            
            logger.info(f"Successfully parsed XML for document: {metadata.title}")
            return document
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            raise GrobidParserError(f"XML parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing XML: {str(e)}")
            raise GrobidParserError(f"Unexpected error parsing XML: {str(e)}")
    
    def _extract_metadata(
        self,
        root: ET.Element,
        pdf_path: Optional[FilePath] = None,
    ) -> DocumentMetadata:
        """Extract document metadata from TEI XML.
        
        Args:
            root: XML root element
            pdf_path: Path to the original PDF file (optional)
            
        Returns:
            DocumentMetadata: Document metadata
        """
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Extract title
        title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled Document"
        
        # Extract authors
        authors = []
        for author_elem in root.findall(".//tei:fileDesc//tei:author", TEI_NS):
            persname = author_elem.find(".//tei:persName", TEI_NS)
            if persname is not None:
                forename = persname.find(".//tei:forename", TEI_NS)
                surname = persname.find(".//tei:surname", TEI_NS)
                
                forename_text = forename.text.strip() if forename is not None and forename.text else ""
                surname_text = surname.text.strip() if surname is not None and surname.text else ""
                
                if forename_text or surname_text:
                    full_name = f"{forename_text} {surname_text}".strip()
                    authors.append(full_name)
        
        # Extract abstract
        abstract_text = ""
        abstract_elem = root.find(".//tei:abstract", TEI_NS)
        if abstract_elem is not None:
            for p in abstract_elem.findall(".//tei:p", TEI_NS):
                if p.text:
                    abstract_text += p.text.strip() + " "
        abstract_text = abstract_text.strip()
        
        # Extract publication date
        pub_date = None
        date_elem = root.find(".//tei:publicationStmt/tei:date", TEI_NS)
        if date_elem is not None and date_elem.get("when"):
            date_str = date_elem.get("when")
            try:
                # Attempt to parse the date - may be in different formats
                if len(date_str) >= 4:  # At least a year
                    year = int(date_str[:4])
                    pub_date = datetime(year, 1, 1).isoformat()
            except (ValueError, TypeError):
                pub_date = None
        
        # Extract DOI
        doi = None
        idno_elem = root.find(".//tei:idno[@type='DOI']", TEI_NS)
        if idno_elem is not None and idno_elem.text:
            doi = idno_elem.text.strip()
            
        # Create document metadata
        source_file = os.path.basename(pdf_path) if pdf_path else None
        file_path = pdf_path if pdf_path else None
        
        # Extract references/bibliography
        references = self._extract_references(root)
        
        # Create metadata object
        metadata = DocumentMetadata(
            document_id=doc_id,
            title=title,
            authors=authors,
            publication_date=pub_date,
            abstract=abstract_text,
            source_file=source_file,
            file_path=file_path,
            document_type="research_paper",
            doi=doi,
            references=references,
            extraction_method="grobid",
        )
        
        return metadata
    
    def _extract_references(self, root: ET.Element) -> List[Dict]:
        """Extract bibliography/references from TEI XML.
        
        Args:
            root: XML root element
            
        Returns:
            List[Dict]: List of reference objects
        """
        references = []
        
        for ref_elem in root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
            try:
                # Initialize reference data
                ref_data = {
                    "title": "",
                    "authors": [],
                    "year": None,
                    "journal": None,
                    "volume": None,
                    "issue": None,
                    "pages": None,
                    "doi": None,
                }
                
                # Extract title
                title_elem = ref_elem.find(".//tei:title[@level='a']", TEI_NS) or ref_elem.find(".//tei:title", TEI_NS)
                if title_elem is not None and title_elem.text:
                    ref_data["title"] = title_elem.text.strip()
                
                # Extract authors
                for author_elem in ref_elem.findall(".//tei:author", TEI_NS):
                    persname = author_elem.find(".//tei:persName", TEI_NS)
                    if persname is not None:
                        forename = persname.find(".//tei:forename", TEI_NS)
                        surname = persname.find(".//tei:surname", TEI_NS)
                        
                        forename_text = forename.text.strip() if forename is not None and forename.text else ""
                        surname_text = surname.text.strip() if surname is not None and surname.text else ""
                        
                        if forename_text or surname_text:
                            full_name = f"{forename_text} {surname_text}".strip()
                            ref_data["authors"].append(full_name)
                
                # Extract date/year
                date_elem = ref_elem.find(".//tei:date", TEI_NS)
                if date_elem is not None and date_elem.get("when"):
                    date_str = date_elem.get("when")
                    if len(date_str) >= 4:  # At least a year
                        try:
                            ref_data["year"] = int(date_str[:4])
                        except ValueError:
                            pass
                
                # Extract journal
                journal_elem = ref_elem.find(".//tei:title[@level='j']", TEI_NS)
                if journal_elem is not None and journal_elem.text:
                    ref_data["journal"] = journal_elem.text.strip()
                
                # Extract volume
                volume_elem = ref_elem.find(".//tei:biblScope[@unit='volume']", TEI_NS)
                if volume_elem is not None and volume_elem.text:
                    ref_data["volume"] = volume_elem.text.strip()
                
                # Extract issue
                issue_elem = ref_elem.find(".//tei:biblScope[@unit='issue']", TEI_NS)
                if issue_elem is not None and issue_elem.text:
                    ref_data["issue"] = issue_elem.text.strip()
                
                # Extract pages
                pages_elem = ref_elem.find(".//tei:biblScope[@unit='page']", TEI_NS)
                if pages_elem is not None:
                    start = pages_elem.get("from", "")
                    end = pages_elem.get("to", "")
                    if start and end:
                        ref_data["pages"] = f"{start}--{end}"
                    elif start:
                        ref_data["pages"] = start
                
                # Extract DOI
                doi_elem = ref_elem.find(".//tei:idno[@type='DOI']", TEI_NS)
                if doi_elem is not None and doi_elem.text:
                    ref_data["doi"] = doi_elem.text.strip()
                
                # Add to references list if we have at least a title
                if ref_data["title"]:
                    references.append(ref_data)
                    
            except Exception as e:
                logger.warning(f"Error extracting reference: {str(e)}")
                continue
        
        return references
    
    def _extract_content_chunks(
        self,
        root: ET.Element,
        chunk_size: int = 1000,
        metadata: DocumentMetadata = None,
    ) -> List[DocumentChunk]:
        """Extract content and split into chunks.
        
        Args:
            root: XML root element
            chunk_size: Size of document chunks in characters
            metadata: Document metadata
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        chunks = []
        
        # Extract full text
        body_elem = root.find(".//tei:body", TEI_NS)
        if body_elem is None:
            logger.warning("No document body found in XML")
            return chunks
        
        # Process document body
        raw_text = self._extract_text_from_element(body_elem)
        
        # Split text into chunks
        text_chunks = self._split_text(raw_text, chunk_size)
        
        # Create DocumentChunks
        doc_id = metadata.document_id if metadata else str(uuid.uuid4())
        
        for i, text in enumerate(text_chunks):
            chunk_id = f"{doc_id}-chunk-{i}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    "document_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document body")
        return chunks
    
    def _extract_text_from_element(self, element: ET.Element) -> str:
        """Recursively extract text from XML element and its children.
        
        Args:
            element: XML element
            
        Returns:
            str: Extracted text
        """
        if element.tag.endswith('note'):
            return ""  # Skip footnotes
        
        text = element.text or ""
        
        for child in element:
            text += self._extract_text_from_element(child)
            
            # Add tail text after processing children
            if child.tail:
                text += child.tail
        
        # Handle paragraphs and headings
        if element.tag.endswith('p') or element.tag.endswith('head'):
            text += "\n\n"
        
        return text
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately equal size.
        
        Args:
            text: Text to split
            chunk_size: Approximate size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or chunk_size <= 0:
            return []
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on paragraph or sentence boundaries
        paragraphs = re.split(r'\n\s*\n|\.\s+', text)
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
                
            # If adding this paragraph would exceed chunk size and we already have content,
            # store the current chunk and start a new one
            if len(current_chunk) + len(p) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = p + ". "
            else:
                current_chunk += p + ". "
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Handle very small text that wasn't split
        if not chunks and text:
            chunks = [text]
            
        return chunks 