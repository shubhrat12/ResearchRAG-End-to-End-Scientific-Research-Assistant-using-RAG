import unittest
from essentials.phase3_2.embedding import embed_chunks
from essentials.phase3_1.models import Chunk

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk(id='1', text='This is a test sentence.', source='test', metadata={'index': 0, 'total': 1}),
            Chunk(id='2', text='Another test sentence.', source='test', metadata={'index': 1, 'total': 1})
        ]

    def test_embed_chunks(self):
        embeddings = embed_chunks(self.chunks)
        self.assertIsInstance(embeddings, list)
        self.assertTrue(all(isinstance(e, dict) for e in embeddings))
        self.assertEqual(len(embeddings), len(self.chunks))
        for e in embeddings:
            self.assertIn('id', e)
            self.assertIn('embedding', e)
            self.assertIn('metadata', e)
            self.assertEqual(len(e['embedding']), 384)  # Assuming the model outputs 384-dimensional embeddings

if __name__ == '__main__':
    unittest.main() 