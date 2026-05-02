import unittest
import tempfile
import os
import numpy as np
from essentials.phase3_1.models import Chunk
from essentials.phase3_2.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.dimension = 384
        self.chunks = [
            Chunk(id='1', text='This is a test sentence.', source='test', metadata={'index': 0, 'total': 3}),
            Chunk(id='2', text='Another test sentence.', source='test', metadata={'index': 1, 'total': 3}),
            Chunk(id='3', text='A third test sentence.', source='test', metadata={'index': 2, 'total': 3})
        ]
        
        # Create random embeddings for testing
        np.random.seed(42)  # For reproducibility
        self.embeddings = [
            np.random.rand(self.dimension).tolist(),
            np.random.rand(self.dimension).tolist(),
            np.random.rand(self.dimension).tolist()
        ]
        
        # Prepare embedded documents
        self.embedded_docs = []
        for i, chunk in enumerate(self.chunks):
            self.embedded_docs.append({
                'id': chunk.id,
                'embedding': self.embeddings[i],
                'metadata': chunk.metadata
            })
        
        # Create vector store
        self.vector_store = VectorStore(dimension=self.dimension)
    
    def test_add_documents(self):
        # Add documents
        self.vector_store.add_documents(self.embedded_docs)
        
        # Check that documents were added
        self.assertEqual(len(self.vector_store.documents), 3)
    
    def test_search(self):
        # Add documents
        self.vector_store.add_documents(self.embedded_docs)
        
        # Search using the first embedding as query
        results = self.vector_store.search(self.embeddings[0], k=2)
        
        # Check that results are returned
        self.assertEqual(len(results), 2)
        
        # The first result should be the document itself (highest similarity)
        self.assertEqual(results[0]['id'], '1')
    
    def test_save_and_load(self):
        # Add documents
        self.vector_store.add_documents(self.embedded_docs)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save vector store
            self.vector_store.save(temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'index.faiss')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'metadata.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'documents.pkl')))
            
            # Load vector store
            loaded_store = VectorStore.load(temp_dir)
            
            # Check that loaded store has the same documents
            self.assertEqual(len(loaded_store.documents), 3)
            
            # Search with loaded store
            results = loaded_store.search(self.embeddings[0], k=2)
            
            # Check results
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['id'], '1')
    
    def test_cosine_index(self):
        # Create cosine similarity vector store
        cosine_store = VectorStore(dimension=self.dimension, index_type="cosine")
        
        # Add documents
        cosine_store.add_documents(self.embedded_docs)
        
        # Search
        results = cosine_store.search(self.embeddings[0], k=2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], '1')

if __name__ == '__main__':
    unittest.main() 