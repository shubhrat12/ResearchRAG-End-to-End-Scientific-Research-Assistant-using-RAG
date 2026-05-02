"""Embedding service for converting text to vector representations."""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, EmbeddingVector, FilePath

logger = get_logger(__name__)


class EmbeddingError(LangChainGPTError):
    """Error raised by embedding operations."""
    
    def __init__(self, message: str = "Embedding error"):
        super().__init__(f"Embedding error: {message}")


class EmbeddingService:
    """Service for computing text embeddings.
    
    In Phase 1, this service provides mock embeddings without making real API calls.
    In future phases, it will integrate with models via LangChain or directly.
    """
    
    def __init__(
        self,
        embedding_model: str = "mock",
        embedding_dimension: int = 384,
        cache_dir: Optional[FilePath] = None,
        use_cache: bool = True,
    ):
        """Initialize embedding service.
        
        Args:
            embedding_model: Name of embedding model
            embedding_dimension: Dimension of embedding vectors
            cache_dir: Directory for caching embeddings
            use_cache: Whether to use caching
        """
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embeddings")
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache of text to embedding
        self._cache: Dict[str, EmbeddingVector] = {}
        
        # Load cache from disk if it exists
        self._load_cache()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List[EmbeddingVector]: List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches to simulate efficient processing
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = [self.embed_text(text) for text in batch_texts]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_text(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingVector: Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text:
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension
        
        # Normalize text for consistent caching
        text = self._normalize_text(text)
        
        # Check cache first
        text_hash = self._hash_text(text)
        
        if self.use_cache and text_hash in self._cache:
            return self._cache[text_hash]
        
        # Generate embedding (mock implementation for Phase 1)
        try:
            # In Phase 1, we use a deterministic random embedding based on text hash
            # In future phases, this will call a real embedding model
            embedding = self._generate_mock_embedding(text)
            
            # Cache the result
            if self.use_cache:
                self._cache[text_hash] = embedding
                
                # Periodically save cache to disk
                if len(self._cache) % 100 == 0:
                    self._save_cache()
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    def embed_document(self, document: Document) -> List[EmbeddingVector]:
        """Generate embeddings for all chunks in a document.
        
        Args:
            document: Document to embed
            
        Returns:
            List[EmbeddingVector]: List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not document.chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk.text for chunk in document.chunks]
        
        # Generate embeddings
        return self.embed_texts(texts)
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddingVector]:
        """Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of document chunks to embed
            
        Returns:
            List[EmbeddingVector]: List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        return self.embed_texts(texts)
    
    def _generate_mock_embedding(self, text: str) -> EmbeddingVector:
        """Generate a mock embedding vector.
        
        In Phase 1, this creates deterministic 'fake' embeddings based on text content.
        The embeddings are random but deterministic for the same input text.
        
        Args:
            text: Input text
            
        Returns:
            EmbeddingVector: Mock embedding vector
        """
        # Use hash of text as seed for random generator
        text_hash = self._hash_text(text)
        seed = int(text_hash[:8], 16)  # Use first 8 chars of hash as seed
        np.random.seed(seed)
        
        # Generate random vector
        vector = np.random.uniform(-1, 1, self.embedding_dimension)
        
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent caching.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        # Simple normalization for caching purposes
        return text.strip().lower()
    
    def _hash_text(self, text: str) -> str:
        """Generate hash of text for caching.
        
        Args:
            text: Input text
            
        Returns:
            str: Hash of text
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.use_cache or not self._cache:
            return
        
        try:
            cache_file = self.cache_dir / f"{self.embedding_model}_cache.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{self.embedding_model}_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.debug(f"Loaded {len(self._cache)} embeddings from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
                self._cache = {}


def get_embedding_service(
    model: str = "mock",
    dimension: int = 384,
    cache_dir: Optional[FilePath] = None,
    use_cache: bool = True,
) -> EmbeddingService:
    """Get an embedding service instance.
    
    Args:
        model: Name of embedding model
        dimension: Dimension of embedding vectors
        cache_dir: Directory for caching embeddings
        use_cache: Whether to use caching
        
    Returns:
        EmbeddingService: Embedding service instance
    """
    return EmbeddingService(
        embedding_model=model,
        embedding_dimension=dimension,
        cache_dir=cache_dir,
        use_cache=use_cache,
    ) 