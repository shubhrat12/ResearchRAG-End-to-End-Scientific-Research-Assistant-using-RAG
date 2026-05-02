"""
Test suite for context building components.

This module tests the functionality of context building components including:
- Context assembly
- Deduplication
- Metadata handling
- Prompt template generation
"""

import sys
import os
import unittest
import logging
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from essentials.phase3_1.models import Chunk, Document
from essentials.phase3_4.context_builder import ContextBuilder, trim_text_to_token_limit
from essentials.phase3_4.deduplication_utils import (
    jaccard_similarity, contains_substring, deduplicate_chunks, 
    diversify_chunks, remove_duplicated_sentences
)
from essentials.phase3_4.prompt_templates import (
    PromptTemplate, PromptTemplateLibrary, QueryType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestContextBuilder(unittest.TestCase):
    """Tests for the ContextBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = ContextBuilder(
            max_tokens=500,
            include_metadata=True,
            deduplicate=True,
            diversify=True,
            coherence_check=True
        )
        
        # Create test chunks
        self.chunks = [
            {
                "id": "chunk1",
                "text": "This is the first chunk about vector databases and embeddings.",
                "metadata": {
                    "source": "test_doc_1",
                    "section": "Introduction"
                },
                "score": 0.95
            },
            {
                "id": "chunk2",
                "text": "Vector databases store high-dimensional vectors and enable similarity search.",
                "metadata": {
                    "source": "test_doc_1",
                    "section": "Background"
                },
                "score": 0.85
            },
            {
                "id": "chunk3",
                "text": "This is a very similar chunk about vector databases and embedding models.",
                "metadata": {
                    "source": "test_doc_2",
                    "section": "Introduction"
                },
                "score": 0.80
            },
            {
                "id": "chunk4",
                "text": "Neural networks can generate embeddings that capture semantic meaning.",
                "metadata": {
                    "source": "test_doc_2",
                    "section": "Methods"
                },
                "score": 0.75
            },
            {
                "id": "chunk5",
                "text": "This is completely unrelated content about different topic.",
                "metadata": {
                    "source": "test_doc_3",
                    "section": "Other"
                },
                "score": 0.60
            }
        ]
    
    def test_basic_context_building(self):
        """Test basic context building functionality."""
        result = self.builder.build_context(self.chunks[:2])
        
        self.assertIn("context", result)
        self.assertIn("chunks_used", result)
        self.assertIn("tokens_used", result)
        self.assertIn("citations", result)
        
        self.assertEqual(result["chunks_used"], 2)
        self.assertTrue(result["tokens_used"] > 0)
        
        # Check if metadata is included
        self.assertIn("Source:", result["context"])
        self.assertIn("Section:", result["context"])
    
    def test_deduplication(self):
        """Test that similar chunks are deduplicated."""
        # Turn off diversity and coherence to isolate deduplication
        builder = ContextBuilder(
            max_tokens=500,
            include_metadata=True,
            deduplicate=True,
            diversify=False,
            coherence_check=False
        )
        
        # Use chunks 1 and 3 which are very similar
        result = builder.build_context([self.chunks[0], self.chunks[2]])
        
        # Should only use one of them
        self.assertEqual(result["chunks_used"], 1)
    
    def test_diversification(self):
        """Test that chunks from different sources are diversified."""
        # Turn off deduplication and coherence to isolate diversification
        builder = ContextBuilder(
            max_tokens=500,
            include_metadata=True,
            deduplicate=False,
            diversify=True,
            coherence_check=False
        )
        
        # Create chunks with the same score but different sources
        chunks = [
            {
                "id": "chunk1",
                "text": "Content from source A.",
                "metadata": {"source": "A", "section": "Intro"},
                "score": 0.9
            },
            {
                "id": "chunk2",
                "text": "More content from source A.",
                "metadata": {"source": "A", "section": "Body"},
                "score": 0.9
            },
            {
                "id": "chunk3",
                "text": "Content from source B.",
                "metadata": {"source": "B", "section": "Intro"},
                "score": 0.9
            }
        ]
        
        result = builder.build_context(chunks)
        
        # Check that both sources are represented
        context = result["context"]
        self.assertIn("source: A", context.lower())
        self.assertIn("source: B", context.lower())
    
    def test_token_limit(self):
        """Test that the context respects token limits."""
        # Create a builder with a very small token limit
        builder = ContextBuilder(
            max_tokens=10,  # Very small limit
            include_metadata=True,
            deduplicate=False,
            diversify=False,
            coherence_check=False
        )
        
        result = builder.build_context(self.chunks)
        
        # Should be limited
        self.assertLessEqual(result["tokens_used"], 10)
    
    def test_build_from_chunks(self):
        """Test building context from Chunk objects."""
        # Create Chunk objects
        chunk_objects = [
            Chunk(
                id="chunk1",
                text="This is test chunk one.",
                metadata={"source": "test_doc", "section": "Test"}
            ),
            Chunk(
                id="chunk2",
                text="This is test chunk two.",
                metadata={"source": "test_doc", "section": "Test"}
            )
        ]
        
        scores = [0.9, 0.8]
        
        result = self.builder.build_from_chunks(chunk_objects, scores)
        
        self.assertEqual(result["chunks_used"], 2)
        self.assertIn("This is test chunk one", result["context"])
        self.assertIn("This is test chunk two", result["context"])

    def test_figure_and_section_chunk_merging(self):
        """Test merging figure chunks with section/content chunks for context building."""
        from essentials.pipeline.figure_detector import convert_figures_to_chunks
        # Mock figure detection output (as from detect_figures_in_pdf)
        detections = [
            [
                {"caption": "Figure 1: Example figure caption.", "bbox": [10, 10, 100, 100]},
            ],
            []
        ]
        pdf_path = "mock_paper.pdf"
        figure_chunks = convert_figures_to_chunks(detections, pdf_path)
        # Mock section/content chunk
        section_chunk = {
            "id": "section1",
            "text": "This is the section text explaining Figure 1.",
            "metadata": {"section": "Results", "page": 0, "source": "mock_paper.pdf"},
            "score": 0.9
        }
        # Merge
        all_chunks = figure_chunks + [section_chunk]
        # Build context for a figure query
        result = self.builder.build_context(all_chunks, query="What does Figure 1 show?")
        self.assertIn("Figure 1", result["context"])
        self.assertIn("Example figure caption", result["context"])
        self.assertIn("section text", result["context"].lower())
        self.assertTrue(result.get("figure_found"))


class TestDeduplicationUtils(unittest.TestCase):
    """Tests for deduplication utilities."""
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        text1 = "This is a test sentence with some words"
        text2 = "This is another test with some similar words"
        text3 = "Completely different content here"
        
        # Similar texts should have higher similarity
        sim1_2 = jaccard_similarity(text1, text2)
        sim1_3 = jaccard_similarity(text1, text3)
        
        self.assertGreater(sim1_2, sim1_3)
    
    def test_contains_substring(self):
        """Test substring detection."""
        text1 = "This contains a specific phrase that should be detected"
        text2 = "This text contains a specific phrase that should be detected in longer form"
        text3 = "Completely different content"
        
        self.assertTrue(contains_substring(text1, text2))
        self.assertFalse(contains_substring(text1, text3))
    
    def test_deduplicate_chunks(self):
        """Test chunk deduplication."""
        chunks = [
            {"id": "1", "text": "This is a test document about vector databases.", "score": 0.9},
            {"id": "2", "text": "This is also a test document about vector databases.", "score": 0.8},
            {"id": "3", "text": "Completely different content here.", "score": 0.7}
        ]
        
        result = deduplicate_chunks(chunks, similarity_threshold=0.7)
        
        # Should remove one of the similar chunks
        self.assertEqual(len(result), 2)
    
    def test_diversify_chunks(self):
        """Test chunk diversification."""
        chunks = [
            {"id": "1", "text": "Source A content 1", "metadata": {"source": "A"}, "score": 0.9},
            {"id": "2", "text": "Source A content 2", "metadata": {"source": "A"}, "score": 0.8},
            {"id": "3", "text": "Source B content", "metadata": {"source": "B"}, "score": 0.7},
            {"id": "4", "text": "Source C content", "metadata": {"source": "C"}, "score": 0.6}
        ]
        
        result = diversify_chunks(chunks, max_per_source=1)
        
        # Should have one chunk from each source
        self.assertEqual(len(result), 3)
        
        # Check sources are all represented
        sources = [chunk["metadata"]["source"] for chunk in result]
        self.assertIn("A", sources)
        self.assertIn("B", sources)
        self.assertIn("C", sources)
    
    def test_remove_duplicated_sentences(self):
        """Test duplicate sentence removal."""
        text = "This is a sentence. This is a sentence. This is a different sentence."
        
        result = remove_duplicated_sentences(text)
        
        # Should contain only unique sentences
        self.assertEqual(result, "This is a sentence. This is a different sentence.")


class TestPromptTemplates(unittest.TestCase):
    """Tests for prompt templates."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.template = PromptTemplate(
            system_message="You are a helpful assistant.",
            context_prefix="Context: ",
            query_prefix="Question: ",
            answer_prefix="Answer: ",
            few_shot_examples=[
                {
                    "context": "Paris is the capital of France.",
                    "query": "What is the capital of France?",
                    "answer": "The capital of France is Paris."
                }
            ]
        )
        
        self.library = PromptTemplateLibrary()
    
    def test_basic_prompt_creation(self):
        """Test basic prompt creation."""
        result = self.template.create_prompt(
            query="What is the capital of Italy?",
            context="Rome is the capital of Italy.",
            include_few_shot=True
        )
        
        self.assertIn("system_message", result)
        self.assertIn("prompt", result)
        
        # Check if parts are included
        prompt = result["prompt"]
        self.assertIn("Context: Paris is the capital of France.", prompt)
        self.assertIn("Question: What is the capital of France?", prompt)
        self.assertIn("Answer: The capital of France is Paris.", prompt)
        self.assertIn("Context: Rome is the capital of Italy.", prompt)
        self.assertIn("Question: What is the capital of Italy?", prompt)
    
    def test_library_query_type_detection(self):
        """Test query type detection."""
        methodology_query = "What method was used to analyze the data?"
        results_query = "What were the main findings of the study?"
        comparison_query = "How does method A compare to method B?"
        definition_query = "What is a vector database?"
        
        self.assertEqual(self.library.detect_query_type(methodology_query), QueryType.METHODOLOGY)
        self.assertEqual(self.library.detect_query_type(results_query), QueryType.RESULTS)
        self.assertEqual(self.library.detect_query_type(comparison_query), QueryType.COMPARISON)
        self.assertEqual(self.library.detect_query_type(definition_query), QueryType.DEFINITION)
    
    def test_library_template_selection(self):
        """Test template selection based on query type."""
        # Try different query types
        for query_type in QueryType:
            template = self.library.get_template(query_type)
            self.assertIsInstance(template, PromptTemplate)
            
            # Each template should have a unique system message
            if query_type != QueryType.GENERAL:  # Skip general as it's the default
                other_templates = [self.library.get_template(qt) for qt in QueryType if qt != query_type]
                for other in other_templates:
                    self.assertNotEqual(template.system_message, other.system_message)
    
    def test_context_formatting(self):
        """Test context formatting for different template types."""
        context = "[Source: Test] This is test content. [Section: Introduction] More content."
        
        # Try different context formats
        scientific_template = PromptTemplate(
            system_message="Test",
            context_format="scientific"
        )
        
        compact_template = PromptTemplate(
            system_message="Test",
            context_format="compact"
        )
        
        default_template = PromptTemplate(
            system_message="Test"
        )
        
        scientific_result = scientific_template.format_context(context)
        compact_result = compact_template.format_context(context)
        default_result = default_template.format_context(context)
        
        # Scientific should add newlines
        self.assertIn("\n\nSource:", scientific_result)
        
        # Compact should reduce whitespace
        self.assertLess(compact_result.count("\n"), default_result.count("\n"))
        
        # Default should return as is
        self.assertEqual(default_result, context)


def run_tests():
    """Run the test suite."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    run_tests() 