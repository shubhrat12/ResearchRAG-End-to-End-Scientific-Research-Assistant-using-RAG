# test_chunking.py

import unittest
from chunking import chunk_fixed, chunk_by_sentence, chunk_by_paragraph, chunk_by_sections, chunk_document
from models import Chunk, Section

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.text = "This is a sentence. This is another sentence."
        self.paragraphs = "This is a paragraph.\n\nThis is another paragraph."
        self.sections = [Section(title="Introduction", content="This is the introduction."),
                        Section(title="Methodology", content="This is the methodology.")]

    def test_chunk_fixed(self):
        chunks = chunk_fixed(self.text, chunk_size=10, overlap=2)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))

    def test_chunk_by_sentence(self):
        chunks = chunk_by_sentence(self.text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))

    def test_chunk_by_paragraph(self):
        chunks = chunk_by_paragraph(self.paragraphs)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))

    def test_chunk_by_sections(self):
        chunks = chunk_by_sections(self.sections)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, Chunk) for chunk in chunks))

    def test_chunk_document_invalid_strategy(self):
        with self.assertRaises(ValueError):
            chunk_document(self.text, strategy="invalid")

if __name__ == '__main__':
    unittest.main() 