import unittest
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from essentials.phase3_1.chunking import (
    chunk_table_content,
    chunk_mathematical_formulas,
    chunk_with_citations,
    process_special_sections,
    chunk_document
)
from essentials.phase3_1.models import Section

class TestSpecializedChunking(unittest.TestCase):
    
    def test_table_chunking(self):
        print("\nTesting table chunking...")
        # Sample text with a table
        table_text = """This is some text before a table.
        
        +------+-------+--------+
        | Col1 | Col2  | Col3   |
        +------+-------+--------+
        | A    | Data1 | Value1 |
        | B    | Data2 | Value2 |
        +------+-------+--------+
        
        This is text after the table."""
        
        chunks = chunk_table_content(table_text)
        
        # Check if table was detected and chunked
        self.assertGreater(len(chunks), 0)
        print(f"Found {len(chunks)} table chunks")
        self.assertEqual(chunks[0].source, "table")
        self.assertEqual(chunks[0].metadata["content_type"], "table")
    
    def test_formula_chunking(self):
        print("\nTesting formula chunking...")
        # Sample text with mathematical formulas
        formula_text = """This text has an inline formula $E = mc^2$ and 
        a display formula:
        
        $$F = G\\frac{m_1 m_2}{r^2}$$
        
        And also an equation environment:
        
        \\begin{equation}
        \\int_{a}^{b} f(x) dx = F(b) - F(a)
        \\end{equation}
        """
        
        chunks = chunk_mathematical_formulas(formula_text)
        
        # Check if formulas were detected and chunked
        self.assertGreater(len(chunks), 0)
        print(f"Found {len(chunks)} formula chunks")
        self.assertEqual(chunks[0].source, "formula")
        self.assertEqual(chunks[0].metadata["content_type"], "formula")
    
    def test_citation_aware_chunking(self):
        print("\nTesting citation-aware chunking...")
        # Sample text with citations
        citation_text = """This is a paragraph with some citations [Smith, 2020] that should be 
        preserved in chunks. Another reference is mentioned here [Johnson et al., 2019].
        
        Some numeric citations like [1] and [2, 3, 4] should also be handled correctly.
        
        Parenthetical citations (Brown, 2018) are also common in some fields."""
        
        chunks = chunk_with_citations(citation_text, chunk_size=20, overlap=5)
        
        # Check if citations were preserved in chunks
        self.assertGreater(len(chunks), 0)
        print(f"Found {len(chunks)} citation-aware chunks")
        self.assertEqual(chunks[0].source, "citation_aware")
        
        # Check if at least one chunk has citations
        has_citations = any(chunk.metadata["has_citations"] for chunk in chunks)
        self.assertTrue(has_citations)
        
        # Print chunks with citations
        for i, chunk in enumerate(chunks):
            if chunk.metadata["has_citations"]:
                print(f"  Chunk {i} has citations: {chunk.metadata['citations']}")
    
    def test_special_sections(self):
        print("\nTesting special section processing...")
        # Create sections including abstract and conclusion
        sections = [
            Section(title="Abstract", content="This is the abstract of the paper."),
            Section(title="Introduction", content="This is the introduction."),
            Section(title="Methodology", content="This describes the methods."),
            Section(title="Conclusion", content="This is the conclusion.")
        ]
        
        # Test special section processing
        chunks = process_special_sections(sections)
        
        # Check if both abstract and conclusion were identified
        self.assertEqual(len(chunks), 2)
        print(f"Found {len(chunks)} special section chunks")
        self.assertEqual(chunks[0].metadata["section_type"], "abstract")
        self.assertEqual(chunks[1].metadata["section_type"], "conclusion")
        print(f"  Identified sections: {[chunk.metadata['section_type'] for chunk in chunks]}")
    
    def test_chunk_document_strategies(self):
        print("\nTesting chunk_document with different strategies...")
        # Test that all strategies are accessible through chunk_document
        test_text = """Abstract
        This is the abstract with a formula $E=mc^2$.
        
        +------+-------+
        | Col1 | Col2  |
        +------+-------+
        | A    | Data1 |
        +------+-------+
        
        This contains a citation [Smith, 2020]."""
        
        # Test each strategy
        table_chunks = chunk_document(test_text, strategy="table")
        self.assertGreater(len(table_chunks), 0)
        print(f"  Table strategy: {len(table_chunks)} chunks")
        
        formula_chunks = chunk_document(test_text, strategy="formula")
        self.assertGreater(len(formula_chunks), 0)
        print(f"  Formula strategy: {len(formula_chunks)} chunks")
        
        citation_chunks = chunk_document(test_text, strategy="citation_aware")
        self.assertGreater(len(citation_chunks), 0)
        print(f"  Citation-aware strategy: {len(citation_chunks)} chunks")
        
        print("\nAll specialized chunking tests passed!")

if __name__ == "__main__":
    print("Running specialized chunking tests...")
    unittest.main() 