"""Unit tests for knowledge base components."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from langchain_gpt.knowledge_base.vector_store import BaseVectorStore, InMemoryVectorStore, VectorStoreError
from langchain_gpt.knowledge_base.chromadb_vector_store import ChromaDBVectorStore
from langchain_gpt.knowledge_base.embeddings import EmbeddingService
from langchain_gpt.knowledge_base.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from langchain_gpt.utils.types import Document, DocumentChunk, DocumentMetadata, DocumentType


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""
    
    def test_init(self):
        """Test initialization."""
        store = InMemoryVectorStore()
        assert store.embedding_dimension == 384
        assert store.store_name == "in_memory"
        assert store.persist_directory is None
        
        custom_store = InMemoryVectorStore(
            embedding_dimension=768,
            store_name="test_store",
            persist_directory="/tmp/test",
        )
        assert custom_store.embedding_dimension == 768
        assert custom_store.store_name == "test_store"
        assert custom_store.persist_directory == Path("/tmp/test")
    
    def test_add_documents(self):
        """Test adding documents."""
        store = InMemoryVectorStore()
        
        # Create test documents
        doc1 = Document(
            document_id="doc1",
            metadata=DocumentMetadata(
                title="Test Document 1",
                document_type=DocumentType.TEXT,
                file_path=Path("/path/to/doc1.txt"),
            ),
            chunks=[
                DocumentChunk(text="Chunk 1 text", chunk_id="chunk1"),
                DocumentChunk(text="Chunk 2 text", chunk_id="chunk2"),
            ],
        )
        
        doc2 = Document(
            document_id="doc2",
            metadata=DocumentMetadata(
                title="Test Document 2",
                document_type=DocumentType.PDF,
                file_path=Path("/path/to/doc2.pdf"),
            ),
            chunks=[
                DocumentChunk(text="Chunk 3 text", chunk_id="chunk3"),
            ],
        )
        
        # Add documents
        doc_ids = store.add_documents([doc1, doc2])
        
        # Check results
        assert len(doc_ids) == 2
        assert doc_ids[0] == "doc1"
        assert doc_ids[1] == "doc2"
        
        # Check internal state
        assert len(store._document_ids) == 2
        assert len(store._documents) == 3
        assert "chunk1" in store._documents
        assert "chunk2" in store._documents
        assert "chunk3" in store._documents
    
    def test_add_document_chunks(self):
        """Test adding document chunks."""
        store = InMemoryVectorStore()
        
        # Create test chunks
        chunks = [
            DocumentChunk(text="Chunk 1 text", chunk_id="chunk1"),
            DocumentChunk(text="Chunk 2 text", chunk_id="chunk2"),
        ]
        
        # Create mock embeddings
        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
        ]
        
        # Add chunks with embeddings
        chunk_ids = store.add_document_chunks(chunks, embeddings)
        
        # Check results
        assert len(chunk_ids) == 2
        assert chunk_ids[0] == "chunk1"
        assert chunk_ids[1] == "chunk2"
        
        # Check internal state
        assert len(store._documents) == 2
        assert len(store._vectors) == 2
        assert "chunk1" in store._documents
        assert "chunk2" in store._documents
        assert "chunk1" in store._vectors
        assert "chunk2" in store._vectors
    
    def test_search(self):
        """Test vector search."""
        store = InMemoryVectorStore()
        
        # Create test chunks
        chunks = [
            DocumentChunk(text="Machine learning is fascinating", chunk_id="chunk1"),
            DocumentChunk(text="Natural language processing is cool", chunk_id="chunk2"),
            DocumentChunk(text="Deep learning is a subset of machine learning", chunk_id="chunk3"),
        ]
        
        # Create mock embeddings (simplified for testing)
        embeddings = [
            [1.0, 0.0, 0.0] + [0.0] * 381,  # Machine learning vector
            [0.0, 1.0, 0.0] + [0.0] * 381,  # NLP vector
            [0.5, 0.0, 0.5] + [0.0] * 381,  # Deep learning vector
        ]
        
        # Add chunks with embeddings
        store.add_document_chunks(chunks, embeddings)
        
        # Search with a query vector similar to "machine learning"
        query_vector = [0.9, 0.0, 0.1] + [0.0] * 381
        results = store.search(query_vector, k=2)
        
        # Check results
        assert len(results) == 2
        assert results[0][0].chunk_id == "chunk1"  # Most similar to machine learning
        assert results[1][0].chunk_id == "chunk3"  # Contains machine learning
        
        # Check scores
        assert results[0][1] > 0.9  # High similarity
        assert results[1][1] > 0.5  # Medium similarity
    
    def test_delete(self):
        """Test document deletion."""
        store = InMemoryVectorStore()
        
        # Create test documents
        doc = Document(
            document_id="doc1",
            metadata=DocumentMetadata(
                title="Test Document",
                document_type=DocumentType.TEXT,
            ),
            chunks=[
                DocumentChunk(text="Chunk 1 text", chunk_id="chunk1"),
                DocumentChunk(text="Chunk 2 text", chunk_id="chunk2"),
            ],
        )
        
        # Add document
        store.add_documents([doc])
        
        # Delete document
        result = store.delete(["doc1"])
        assert result is True
        
        # Check internal state
        assert "doc1" not in store._document_ids
        assert "chunk1" not in store._documents
        assert "chunk2" not in store._documents


class TestChromaDBVectorStore:
    """Tests for ChromaDBVectorStore."""
    
    @mock.patch("chromadb.PersistentClient")
    def test_init(self, mock_client):
        """Test initialization."""
        # Mock the collection
        mock_collection = mock.MagicMock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBVectorStore(
                embedding_dimension=384,
                store_name="test_store",
                persist_directory=temp_dir,
            )
            
            # Check attributes
            assert store.embedding_dimension == 384
            assert store.store_name == "test_store"
            assert store.persist_directory == Path(temp_dir)
            assert store.collection_name == "test_store"
            
            # Check client initialization
            mock_client.assert_called_once()
            mock_client.return_value.get_collection.assert_called_once()
    
    @mock.patch("chromadb.PersistentClient")
    def test_add_document_chunks(self, mock_client):
        """Test adding document chunks."""
        # Mock the collection
        mock_collection = mock.MagicMock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBVectorStore(
                embedding_dimension=384,
                persist_directory=temp_dir,
            )
            
            # Create test chunks
            chunks = [
                DocumentChunk(text="Chunk 1 text", chunk_id="chunk1"),
                DocumentChunk(text="Chunk 2 text", chunk_id="chunk2"),
            ]
            
            # Create mock embeddings
            embeddings = [
                [0.1] * 384,
                [0.2] * 384,
            ]
            
            # Add chunks with embeddings
            chunk_ids = store.add_document_chunks(chunks, embeddings)
            
            # Check results
            assert len(chunk_ids) == 2
            assert chunk_ids[0] == "chunk1"
            assert chunk_ids[1] == "chunk2"
            
            # Check that collection.add was called
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]
            assert len(call_args["ids"]) == 2
            assert call_args["ids"][0] == "chunk1"
            assert call_args["ids"][1] == "chunk2"
            assert len(call_args["embeddings"]) == 2
            assert len(call_args["documents"]) == 2
            assert call_args["documents"][0] == "Chunk 1 text"
            assert call_args["documents"][1] == "Chunk 2 text"
    
    @mock.patch("chromadb.PersistentClient")
    def test_search(self, mock_client):
        """Test vector search."""
        # Mock the collection and its query response
        mock_collection = mock.MagicMock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        # Mock query response
        mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["Chunk 1 text", "Chunk 2 text"]],
            "metadatas": [[{"key": "value"}, {"key": "value2"}]],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaDBVectorStore(
                embedding_dimension=384,
                persist_directory=temp_dir,
            )
            
            # Create a query vector
            query_vector = [0.1] * 384
            
            # Perform search
            results = store.search(query_vector, k=2)
            
            # Check collection.query was called
            mock_collection.query.assert_called_once()
            call_args = mock_collection.query.call_args[1]
            assert call_args["query_embeddings"] == [query_vector]
            assert call_args["n_results"] == 2
            
            # Check results
            assert len(results) == 2
            assert isinstance(results[0][0], DocumentChunk)
            assert results[0][0].text == "Chunk 1 text"
            assert results[0][1] == 0.9  # 1 - distance
            assert results[1][0].text == "Chunk 2 text"
            assert results[1][1] == 0.8  # 1 - distance


class TestEmbeddingService:
    """Tests for EmbeddingService."""
    
    def test_init(self):
        """Test initialization."""
        service = EmbeddingService()
        assert service.embedding_dimension == 384
        assert service.use_real_embeddings is False
        
        custom_service = EmbeddingService(
            embedding_dimension=768,
            use_real_embeddings=True,
        )
        assert custom_service.embedding_dimension == 768
        assert custom_service.use_real_embeddings is True
    
    def test_embed_text(self):
        """Test text embedding."""
        service = EmbeddingService(embedding_dimension=384)
        
        # Test empty text
        empty_embedding = service.embed_text("")
        assert len(empty_embedding) == 384
        
        # Test normal text
        text_embedding = service.embed_text("This is a test")
        assert len(text_embedding) == 384
        
        # Test determinism (same text should produce same embedding)
        text_embedding2 = service.embed_text("This is a test")
        assert text_embedding == text_embedding2
        
        # Test different text produces different embedding
        different_embedding = service.embed_text("This is different")
        assert text_embedding != different_embedding
    
    def test_embed_texts(self):
        """Test batch text embedding."""
        service = EmbeddingService(embedding_dimension=384)
        
        # Test batch embedding
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = service.embed_texts(texts)
        
        # Check results
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        assert len(embeddings[2]) == 384
        
        # Test empty batch
        empty_embeddings = service.embed_texts([])
        assert empty_embeddings == []


@pytest.mark.skipif(not os.environ.get("RUN_REAL_MODEL_TESTS"), reason="Skipping tests that require real models")
class TestSentenceTransformerEmbeddings:
    """Tests for SentenceTransformerEmbeddings with real models."""
    
    def test_init(self):
        """Test initialization with a real model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                cache_dir=temp_dir,
            )
            
            # Check attributes
            assert embeddings.model_name == "all-MiniLM-L6-v2"
            assert embeddings.device == "cpu"
            assert embeddings.embedding_dimension == 384
    
    def test_embed_text(self):
        """Test embedding a single text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                cache_dir=temp_dir,
            )
            
            # Test embedding
            vector = embeddings.embed_text("This is a test sentence.")
            
            # Check result
            assert len(vector) == 384
            assert isinstance(vector, list)
            assert all(isinstance(x, float) for x in vector)
    
    def test_embed_texts(self):
        """Test embedding multiple texts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                cache_dir=temp_dir,
            )
            
            # Test batch embedding
            texts = [
                "First test sentence.",
                "Second test sentence.",
                "Third test sentence.",
            ]
            vectors = embeddings.embed_texts(texts)
            
            # Check results
            assert len(vectors) == 3
            assert all(len(v) == 384 for v in vectors)
            assert all(isinstance(v, list) for v in vectors)
    
    def test_caching(self):
        """Test embedding caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                cache_dir=temp_dir,
                use_cache=True,
            )
            
            # First embedding call should compute the embedding
            text = "This is a test for caching."
            vector1 = embeddings.embed_text(text)
            
            # Mock the model to detect if it's called again
            original_encode = embeddings.model.encode
            call_count = [0]
            
            def mock_encode(*args, **kwargs):
                call_count[0] += 1
                return original_encode(*args, **kwargs)
            
            embeddings.model.encode = mock_encode
            
            # Second call should use the cache
            vector2 = embeddings.embed_text(text)
            
            # Check that model.encode wasn't called
            assert call_count[0] == 0
            
            # Check that vectors are the same
            assert vector1 == vector2 