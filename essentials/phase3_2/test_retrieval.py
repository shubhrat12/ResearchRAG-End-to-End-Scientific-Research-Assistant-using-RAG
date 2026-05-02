import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from essentials.phase3_1.models import Chunk
from essentials.phase3_2.vector_store import VectorStore
from essentials.phase3_2.scientific_embeddings import ScientificEmbedding
from essentials.phase3_2.retrieval import Retriever

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        # Mock the vector store
        self.vector_store = MagicMock(spec=VectorStore)
        
        # Mock the embedding model
        self.embedding_model = MagicMock(spec=ScientificEmbedding)
        
        # Set up the retriever with mocks
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )
        
        # Set up test data
        self.query = "quantum physics applications"
        self.query_embedding = [0.1] * 384  # Mock embedding
        
        # Mock search results
        self.search_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "metadata": {
                    "text": "Applications of quantum physics in computing",
                    "citation_count": 50
                }
            },
            {
                "id": "doc2",
                "score": 0.85,
                "metadata": {
                    "text": "Quantum mechanics and its applications",
                    "citation_count": 20
                }
            },
            {
                "id": "doc3",
                "score": 0.75,
                "metadata": {
                    "text": "Introduction to physics",
                    "citation_count": 5
                }
            }
        ]
        
        # Configure the mocks
        self.embedding_model.embed_text.return_value = self.query_embedding
        self.vector_store.search.return_value = self.search_results
    
    def test_retrieve(self):
        # Test basic retrieval
        results = self.retriever.retrieve(self.query, k=3)
        
        # Check that embed_text was called with the query
        self.embedding_model.embed_text.assert_called_once_with(self.query)
        
        # Check that search was called with the embedding
        self.vector_store.search.assert_called_once_with(self.query_embedding, k=3)
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "doc1")
    
    def test_retrieve_with_filter(self):
        # Test retrieval with metadata filtering
        filter_metadata = {"citation_count": 50}
        
        # Call retrieve with filter
        results = self.retriever.retrieve(self.query, k=3, filter_metadata=filter_metadata)
        
        # Only doc1 should match the filter
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1")
    
    def test_extract_keywords(self):
        # Test keyword extraction
        keywords = self.retriever._extract_keywords("quantum physics applications")
        
        # Check extracted keywords
        self.assertIn("quantum", keywords)
        self.assertIn("physics", keywords)
        self.assertIn("applications", keywords)
        
        # Test with common words
        keywords = self.retriever._extract_keywords("the quantum and physics with applications")
        
        # Common words should be removed
        self.assertNotIn("the", keywords)
        self.assertNotIn("and", keywords)
        self.assertNotIn("with", keywords)
    
    def test_calculate_keyword_score(self):
        # Test keyword scoring
        text = "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles."
        keywords = ["quantum", "physics", "fundamental"]
        
        score = self.retriever._calculate_keyword_score(text, keywords)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with no keywords
        score = self.retriever._calculate_keyword_score(text, [])
        self.assertEqual(score, 0.0)
        
        # Test with no text
        score = self.retriever._calculate_keyword_score("", keywords)
        self.assertEqual(score, 0.0)
    
    def test_hybrid_retrieve(self):
        # Test hybrid retrieval
        with patch.object(self.retriever, '_extract_keywords') as mock_extract:
            with patch.object(self.retriever, '_calculate_keyword_score') as mock_score:
                # Set up mocks
                mock_extract.return_value = ["quantum", "physics", "applications"]
                mock_score.return_value = 0.8  # High keyword score
                
                # Call hybrid retrieve
                results = self.retriever.hybrid_retrieve(self.query, k=2)
                
                # Check results
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0]["id"], "doc1")
                self.assertEqual(results[1]["id"], "doc2")
                
                # Check scores are calculated
                self.assertIn("semantic_score", results[0])
                self.assertIn("keyword_score", results[0])
                self.assertIn("combined_score", results[0])
    
    def test_retrieve_with_reranking(self):
        # Test retrieval with citation-based reranking
        results = self.retriever.retrieve_with_reranking(self.query, k=2, initial_k=3)
        
        # Check results
        self.assertEqual(len(results), 2)
        
        # doc1 should still be first (highest semantic and citation scores)
        self.assertEqual(results[0]["id"], "doc1")
        
        # Check scores are calculated
        self.assertIn("semantic_score", results[0])
        self.assertIn("citation_score", results[0])
        self.assertIn("combined_score", results[0])

if __name__ == '__main__':
    unittest.main() 