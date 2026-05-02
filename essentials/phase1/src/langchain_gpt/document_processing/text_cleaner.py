"""Text cleaning utilities for document processing."""

import re
from typing import List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Text cleaning utilities for document processing."""
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace from text.
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        # Simple URL regex pattern
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_email_addresses(text: str) -> str:
        """Remove email addresses from text.
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def remove_special_characters(text: str, keep_chars: Optional[str] = None) -> str:
        """Remove special characters from text.
        
        Args:
            text: Input text
            keep_chars: Characters to keep (in addition to letters, numbers, spaces)
            
        Returns:
            str: Cleaned text
        """
        if keep_chars:
            pattern = f'[^A-Za-z0-9\\s{re.escape(keep_chars)}]'
        else:
            pattern = r'[^A-Za-z0-9\s]'
        
        return re.sub(pattern, '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        # Replace various whitespace characters with a standard space
        text = re.sub(r'[\n\r\t\f\v]+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def fix_line_breaks(text: str) -> str:
        """Fix line breaks in text extracted from PDF.
        
        Args:
            text: Input text from PDF
            
        Returns:
            str: Text with fixed line breaks
        """
        # Fix hyphenated line breaks: "word-\nbreak" -> "wordbreak"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Fix normal line breaks within sentences
        text = re.sub(r'(?<!\.)(\w)\s*\n\s*(\w)', r'\1 \2', text)
        # Keep paragraph breaks (double line breaks)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Normalize remaining single line breaks
        text = re.sub(r'\n(?!\n)', ' ', text)
        return text
    
    @staticmethod
    def clean_pdf_text(text: str) -> str:
        """Clean text extracted from PDF.
        
        This applies a series of cleaning operations specifically for PDF text.
        
        Args:
            text: Input text from PDF
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Fix line breaks
        text = TextCleaner.fix_line_breaks(text)
        
        # Remove header/footer patterns (like page numbers)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Fix spacing issues
        text = re.sub(r'(\w)- (\w)', r'\1-\2', text)  # Fix "word - break" -> "word-break"
        
        # Normalize whitespace
        text = TextCleaner.normalize_whitespace(text)
        
        return text
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting pattern
        # This is not perfect but works for basic cases
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def clean_text(text: str, remove_urls: bool = True, fix_line_breaks: bool = True) -> str:
        """Apply multiple cleaning operations to text.
        
        Args:
            text: Input text
            remove_urls: Whether to remove URLs
            fix_line_breaks: Whether to fix line breaks
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Apply cleaning operations
        if remove_urls:
            text = TextCleaner.remove_urls(text)
        
        if fix_line_breaks:
            text = TextCleaner.fix_line_breaks(text)
        
        # Remove extra whitespace
        text = TextCleaner.remove_extra_whitespace(text)
        
        return text 