"""Unit tests for document processing module."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from PyPDF2 import PdfWriter

from langchain_gpt.document_processing.document_processor import DocumentProcessor
from langchain_gpt.document_processing.pdf_extractor import PDFExtractor
from langchain_gpt.document_processing.text_cleaner import TextCleaner
from langchain_gpt.utils.errors import DocumentProcessingError
from langchain_gpt.utils.types import Document, DocumentMetadata, DocumentType


def create_sample_pdf(path, content="Sample PDF content for testing."):
    """Create a sample PDF file for testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    
    # Add some metadata
    writer.add_metadata({
        "/Title": "Sample PDF",
        "/Author": "Test",
        "/CreationDate": "D:20230101000000",
    })
    
    with open(path, "wb") as f:
        writer.write(f)


class TestPDFExtractor:
    """Tests for PDFExtractor."""
    
    def test_init(self):
        """Test initialization."""
        extractor = PDFExtractor()
        assert extractor.chunk_size == 1000
        assert extractor.chunk_overlap == 200
        
        custom_extractor = PDFExtractor(chunk_size=500, chunk_overlap=100)
        assert custom_extractor.chunk_size == 500
        assert custom_extractor.chunk_overlap == 100
    
    def test_extract_text_from_pdf(self):
        """Test PDF text extraction."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name
        
        try:
            create_sample_pdf(filepath)
            
            extractor = PDFExtractor()
            text, metadata = extractor.extract_text_from_pdf(filepath)
            
            assert isinstance(text, str)
            assert isinstance(metadata, dict)
            assert "pages" in metadata
            assert metadata["pages"] > 0
            
            # Test with non-existent file
            with pytest.raises(DocumentProcessingError):
                extractor.extract_text_from_pdf("non_existent.pdf")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_create_chunks(self):
        """Test text chunking."""
        extractor = PDFExtractor(chunk_size=10, chunk_overlap=2)
        
        # Test with empty text
        chunks = extractor.create_chunks("")
        assert len(chunks) == 0
        
        # Test with text shorter than chunk size
        chunks = extractor.create_chunks("Short text")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
        
        # Test with text longer than chunk size
        chunks = extractor.create_chunks("This is a longer text that should be split into multiple chunks")
        assert len(chunks) > 1
        
        # Test chunk overlap
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = extractor.create_chunks(text)
        assert len(chunks) > 1
        # Check overlap between chunks
        first_chunk_end = chunks[0].text[-1]
        second_chunk_start = chunks[1].text[0]
        assert text.find(second_chunk_start) <= text.find(first_chunk_end) + extractor.chunk_overlap
    
    def test_process_pdf(self):
        """Test PDF processing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name
        
        try:
            create_sample_pdf(filepath)
            
            extractor = PDFExtractor()
            document = extractor.process_pdf(filepath)
            
            assert isinstance(document, Document)
            assert isinstance(document.metadata, DocumentMetadata)
            assert document.metadata.document_type == DocumentType.PDF
            assert document.metadata.file_path == Path(filepath)
            
            # Check if document has chunks
            assert hasattr(document, "chunks")
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestTextCleaner:
    """Tests for TextCleaner."""
    
    def test_remove_extra_whitespace(self):
        """Test removing extra whitespace."""
        text = "  This   has  extra   spaces  "
        cleaned = TextCleaner.remove_extra_whitespace(text)
        assert cleaned == "This has extra spaces"
    
    def test_remove_urls(self):
        """Test removing URLs."""
        text = "Check out https://example.com and www.test.com"
        cleaned = TextCleaner.remove_urls(text)
        assert "https://" not in cleaned
        assert "www." not in cleaned
    
    def test_remove_email_addresses(self):
        """Test removing email addresses."""
        text = "Contact us at test@example.com or support@test.org"
        cleaned = TextCleaner.remove_email_addresses(text)
        assert "@example.com" not in cleaned
        assert "@test.org" not in cleaned
    
    def test_fix_line_breaks(self):
        """Test fixing line breaks."""
        text = "This is a\nbroken line\n\nThis is a new paragraph."
        fixed = TextCleaner.fix_line_breaks(text)
        assert fixed == "This is a broken line\n\nThis is a new paragraph."
    
    def test_clean_pdf_text(self):
        """Test cleaning PDF text."""
        text = "Page header\n1\n\nThis is some-\nwhere text is broken"
        cleaned = TextCleaner.clean_pdf_text(text)
        assert "Page header" in cleaned
        assert "somewhere text is broken" in cleaned


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""
    
    def test_init(self):
        """Test initialization."""
        processor = DocumentProcessor()
        assert processor.pdf_extractor is not None
        assert processor.clean_text is True
    
    def test_get_document_type(self):
        """Test document type detection."""
        processor = DocumentProcessor()
        
        assert processor.get_document_type("document.pdf") == DocumentType.PDF
        assert processor.get_document_type("document.txt") == DocumentType.TEXT
        assert processor.get_document_type("document.md") == DocumentType.MARKDOWN
        assert processor.get_document_type("document.unknown") == DocumentType.UNKNOWN
    
    def test_process_file(self):
        """Test file processing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            filepath = f.name
        
        try:
            create_sample_pdf(filepath)
            
            processor = DocumentProcessor()
            document = processor.process_file(filepath)
            
            assert isinstance(document, Document)
            
            # Test with unsupported file type
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f2:
                unsupported_path = f2.name
            
            try:
                with pytest.raises(DocumentProcessingError):
                    processor.process_file(unsupported_path)
            finally:
                if os.path.exists(unsupported_path):
                    os.remove(unsupported_path)
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_process_directory(self):
        """Test directory processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample files
            pdf_path = os.path.join(temp_dir, "sample.pdf")
            create_sample_pdf(pdf_path)
            
            txt_path = os.path.join(temp_dir, "sample.txt")
            with open(txt_path, "w") as f:
                f.write("Sample text file")
            
            # Create subdirectory with additional file
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir, exist_ok=True)
            subdir_pdf_path = os.path.join(subdir, "subdir_sample.pdf")
            create_sample_pdf(subdir_pdf_path)
            
            processor = DocumentProcessor()
            
            # Test with specific file types
            documents = processor.process_directory(
                temp_dir,
                file_types=[DocumentType.PDF],
                recursive=True,
            )
            assert len(documents) == 2  # Two PDF files
            
            # Test non-recursive
            documents = processor.process_directory(
                temp_dir,
                file_types=[DocumentType.PDF],
                recursive=False,
            )
            assert len(documents) == 1  # Only one PDF in root directory 