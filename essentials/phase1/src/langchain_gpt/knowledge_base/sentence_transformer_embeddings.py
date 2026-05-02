"""Embedding service using SentenceTransformers for converting text to vector representations."""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config.settings import get_settings
from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, EmbeddingVector, FilePath

logger = get_logger(__name__)


class SentenceTransformerError(LangChainGPTError):
    """Error raised by sentence transformer operations."""
    
    def __init__(self, message: str = "Sentence transformer error"):
        super().__init__(f"Sentence transformer error: {message}")


class SentenceTransformerEmbeddings:
    """Service for computing text embeddings using SentenceTransformers.
    
    This service provides real embeddings using the SentenceTransformers library.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        cache_dir: Optional[FilePath] = None,
        use_cache: bool = True,
        normalize_embeddings: bool = True,
    ):
        """Initialize embedding service.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use for inference ('cpu', 'cuda', etc.)
            cache_dir: Directory for caching embeddings
            use_cache: Whether to use caching
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Raises:
            SentenceTransformerError: If embedding service initialization fails
        """
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or ("cuda" if settings.USE_CUDA else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.EMBEDDING_CACHE_DIR)
        self.use_cache = use_cache
        self.normalize_embeddings = normalize_embeddings
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the model
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded, embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {str(e)}")
            raise SentenceTransformerError(f"Failed to load model: {str(e)}")
        
        # Initialize cache of text to embedding
        self._cache: Dict[str, EmbeddingVector] = {}
        
        # Load cache from disk if it exists
        self._load_cache()
    
    def embed_texts(self, texts: List[str], batch_size: int = None) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List[EmbeddingVector]: List of embedding vectors
            
        Raises:
            SentenceTransformerError: If embedding generation fails
        """
        if not texts:
            return []
        
        settings = get_settings()
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        # Check which texts are already cached
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)
        
        for i, text in enumerate(texts):
            # Normalize text for consistent caching
            normalized_text = self._normalize_text(text)
            text_hash = self._hash_text(normalized_text)
            
            # Check cache first
            if self.use_cache and text_hash in self._cache:
                embeddings[i] = self._cache[text_hash]
            else:
                uncached_texts.append(normalized_text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                start_time = time.time()
                
                # Generate embeddings in batches
                uncached_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=len(uncached_texts) > 100,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                )
                
                elapsed = time.time() - start_time
                logger.info(
                    f"Generated {len(uncached_texts)} embeddings in {elapsed:.2f}s "
                    f"({len(uncached_texts) / elapsed:.2f} texts/s)"
                )
                
                # Store embeddings in result list and cache
                for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                    # Convert to list for JSON serialization
                    embedding_vector = uncached_embeddings[i].tolist()
                    embeddings[idx] = embedding_vector
                    
                    # Cache the result
                    if self.use_cache:
                        text_hash = self._hash_text(text)
                        self._cache[text_hash] = embedding_vector
                
                # Periodically save cache to disk
                if self.use_cache and len(uncached_texts) > 0:
                    self._save_cache()
                    
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise SentenceTransformerError(f"Failed to generate embeddings: {str(e)}")
        
        return embeddings
    
    def embed_text(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingVector: Embedding vector
            
        Raises:
            SentenceTransformerError: If embedding generation fails
        """
        if not text:
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension
        
        # Use the batch method for a single text
        result = self.embed_texts([text])
        return result[0]
    
    def embed_document(self, document: Document) -> List[EmbeddingVector]:
        """Generate embeddings for all chunks in a document.
        
        Args:
            document: Document to embed
            
        Returns:
            List[EmbeddingVector]: List of embedding vectors
            
        Raises:
            SentenceTransformerError: If embedding generation fails
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
            SentenceTransformerError: If embedding generation fails
        """
        if not chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        return self.embed_texts(texts)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model.
        
        Returns:
            Dict[str, str]: Model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": str(self.embedding_dimension),
            "cache_size": str(len(self._cache)) if self._cache else "0",
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent caching.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        # Simple normalization for caching purposes
        return text.strip()
    
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
            cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self._cache, f)
            logger.debug(f"Saved {len(self._cache)} embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} embeddings from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
                self._cache = {} 