"""
Advanced embeddings module with scientific capabilities, caching, and evaluation.

This module provides enhanced embedding capabilities for scientific content
without requiring problematic dependencies like huggingface_hub.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json
import pickle
import time
import hashlib
import numpy as np
import logging
from essentials.phase3_1.models import Chunk, Section
from collections import OrderedDict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingCache:
    """In-memory and disk-based cache for embeddings."""
    
    def __init__(self, cache_dir: str = "data/embedding_cache", max_size: int = 1000):
        """Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store persistent cache
            max_size: Maximum number of entries in the in-memory cache
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache = OrderedDict()  # LRU cache
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load disk cache index
        self.disk_index_path = os.path.join(cache_dir, "index.json")
        self.disk_index = self._load_disk_index()
        
        logger.info(f"Initialized embedding cache with max size {max_size}")
        
    def _load_disk_index(self) -> Dict[str, str]:
        """Load the disk cache index.
        
        Returns:
            Dictionary mapping hashes to filenames
        """
        if os.path.exists(self.disk_index_path):
            try:
                with open(self.disk_index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading disk cache index: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_disk_index(self):
        """Save the disk cache index."""
        try:
            with open(self.disk_index_path, 'w') as f:
                json.dump(self.disk_index, f)
        except Exception as e:
            logger.error(f"Error saving disk cache index: {str(e)}")
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash for a text string.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get an embedding from the cache.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding vector or None if not found
        """
        text_hash = self._hash_text(text)
        
        # Check memory cache first
        if text_hash in self.memory_cache:
            # Move to end of OrderedDict to mark as recently used
            embedding = self.memory_cache.pop(text_hash)
            self.memory_cache[text_hash] = embedding
            return embedding
        
        # Check disk cache
        if text_hash in self.disk_index:
            cache_path = os.path.join(self.cache_dir, self.disk_index[text_hash])
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Add to memory cache
                self.memory_cache[text_hash] = embedding
                
                # Enforce memory cache size limit
                if len(self.memory_cache) > self.max_size:
                    self.memory_cache.popitem(last=False)  # Remove oldest item
                
                return embedding
            except Exception as e:
                logger.error(f"Error loading embedding from disk cache: {str(e)}")
                return None
        
        return None
    
    def set(self, text: str, embedding: List[float]):
        """Add an embedding to the cache.
        
        Args:
            text: Text being embedded
            embedding: Embedding vector
        """
        text_hash = self._hash_text(text)
        
        # Add to memory cache
        self.memory_cache[text_hash] = embedding
        
        # Enforce memory cache size limit
        if len(self.memory_cache) > self.max_size:
            self.memory_cache.popitem(last=False)  # Remove oldest item
        
        # Add to disk cache
        cache_filename = f"{text_hash}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update disk index
            self.disk_index[text_hash] = cache_filename
            self._save_disk_index()
        except Exception as e:
            logger.error(f"Error saving embedding to disk cache: {str(e)}")


class EmbeddingCompressor:
    """Compress embeddings to reduce storage requirements."""
    
    def __init__(self, target_dimensions: int = 100, random_seed: int = 42):
        """Initialize the embedding compressor.
        
        Args:
            target_dimensions: Number of dimensions to compress to
            random_seed: Random seed for reproducibility
        """
        self.target_dimensions = target_dimensions
        self.random_seed = random_seed
        self.projection_matrix = None
        
    def fit(self, embeddings: List[List[float]]):
        """Fit the compressor to a set of embeddings.
        
        Args:
            embeddings: List of embedding vectors
        """
        if not embeddings:
            logger.error("Cannot fit compressor to empty embeddings list")
            return
            
        # Get embedding dimension
        original_dim = len(embeddings[0])
        
        # Initialize random projection matrix
        np.random.seed(self.random_seed)
        self.projection_matrix = np.random.randn(original_dim, self.target_dimensions) / np.sqrt(self.target_dimensions)
        
        logger.info(f"Initialized embedding compressor: {original_dim} → {self.target_dimensions} dimensions")
        
    def compress(self, embedding: List[float]) -> List[float]:
        """Compress an embedding vector.
        
        Args:
            embedding: Original embedding vector
            
        Returns:
            Compressed embedding vector
        """
        if self.projection_matrix is None:
            logger.error("Compressor not fitted. Call fit() first.")
            return embedding
            
        # Convert to numpy array
        embedding_np = np.array(embedding)
        
        # Apply projection
        compressed = np.dot(embedding_np, self.projection_matrix)
        
        return compressed.tolist()
        
    def compress_batch(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Compress a batch of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            List of compressed embedding vectors
        """
        return [self.compress(embedding) for embedding in embeddings]


class EmbeddingEvaluator:
    """Evaluate embedding quality for scientific text."""
    
    def __init__(self):
        """Initialize the embedding evaluator."""
        pass
        
    def evaluate_similarity(self, 
                          embeddings: List[List[float]], 
                          labels: List[int],
                          expected_similarity: Dict[Tuple[int, int], float] = None) -> Dict[str, float]:
        """Evaluate embedding similarity.
        
        Args:
            embeddings: List of embedding vectors
            labels: Cluster labels for each embedding
            expected_similarity: Optional dictionary mapping pairs of indices to expected similarity
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate all pairwise similarities
        n = len(embeddings)
        similarities = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities[i, j] = similarity
                similarities[j, i] = similarity
        
        # Calculate intra-cluster and inter-cluster similarities
        unique_labels = set(labels)
        intra_cluster_sims = []
        inter_cluster_sims = []
        
        for i in range(n):
            for j in range(i+1, n):
                sim = similarities[i, j]
                if labels[i] == labels[j]:
                    intra_cluster_sims.append(sim)
                else:
                    inter_cluster_sims.append(sim)
        
        # Calculate metrics
        avg_intra = np.mean(intra_cluster_sims) if intra_cluster_sims else 0.0
        avg_inter = np.mean(inter_cluster_sims) if inter_cluster_sims else 0.0
        contrast = avg_intra - avg_inter if (intra_cluster_sims and inter_cluster_sims) else 0.0
        
        # Calculate correlation with expected similarities if provided
        correlation = 0.0
        if expected_similarity:
            expected = []
            actual = []
            for (i, j), expected_sim in expected_similarity.items():
                if i < n and j < n:
                    expected.append(expected_sim)
                    actual.append(similarities[i, j])
            
            if expected:
                correlation = np.corrcoef(expected, actual)[0, 1]
        
        return {
            "avg_intra_cluster_similarity": float(avg_intra),
            "avg_inter_cluster_similarity": float(avg_inter),
            "cluster_contrast": float(contrast),
            "correlation_with_expected": float(correlation)
        }
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity
        """
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        
        norm1 = np.linalg.norm(v1_np)
        norm2 = np.linalg.norm(v2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(v1_np, v2_np) / (norm1 * norm2)
    
    def generate_report(self, 
                       model_name: str, 
                       metrics: Dict[str, float], 
                       output_file: Optional[str] = None) -> str:
        """Generate a report of embedding quality.
        
        Args:
            model_name: Name of the embedding model
            metrics: Evaluation metrics
            output_file: Optional file to write the report
            
        Returns:
            Report text
        """
        report = f"Embedding Evaluation Report: {model_name}\n"
        report += "=" * 50 + "\n\n"
        
        report += "Similarity Metrics:\n"
        report += f"  Intra-cluster similarity: {metrics['avg_intra_cluster_similarity']:.4f}\n"
        report += f"  Inter-cluster similarity: {metrics['avg_inter_cluster_similarity']:.4f}\n"
        report += f"  Cluster contrast: {metrics['cluster_contrast']:.4f}\n"
        
        if 'correlation_with_expected' in metrics:
            report += f"  Correlation with expected: {metrics['correlation_with_expected']:.4f}\n"
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
            except Exception as e:
                logger.error(f"Error writing evaluation report: {str(e)}")
        
        return report


class AdvancedEmbedding:
    """Advanced embedding model with caching and compression."""
    
    def __init__(self, 
                model_name: str = "en_core_web_md", 
                use_cache: bool = True,
                cache_dir: str = "data/embedding_cache",
                enable_compression: bool = False,
                target_dimensions: int = 100):
        """Initialize the advanced embedding model.
        
        Args:
            model_name: Name of the spaCy model to use
            use_cache: Whether to use embedding cache
            cache_dir: Cache directory path
            enable_compression: Whether to enable dimension reduction
            target_dimensions: Target dimensions for compression
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.enable_compression = enable_compression
        self.target_dimensions = target_dimensions
        
        # Initialize cache if enabled
        if use_cache:
            self.cache = EmbeddingCache(cache_dir=cache_dir)
        
        # Initialize compressor if enabled
        self.compressor = EmbeddingCompressor(target_dimensions=target_dimensions)
        
        # Load language model
        try:
            import spacy
            self.raw_dimension = 0  # Will be set after loading model
            
            # This will download the model if it's not already available
            logger.info(f"Loading spaCy model: {model_name}")
            self.model = spacy.load(model_name)
            
            # Get the dimension from the model's vectors
            if self.model.vocab.vectors.shape[0] > 0:
                self.raw_dimension = self.model.vocab.vectors.shape[1]
                logger.info(f"Loaded spaCy model with dimension: {self.raw_dimension}")
                
                # Initialize compressor with sample embeddings if compression is enabled
                if enable_compression:
                    # Get sample words for fitting the compressor
                    sample_words = [self.model.vocab[i].text for i in range(min(100, len(self.model.vocab))) 
                                   if self.model.vocab[i].has_vector]
                    if sample_words:
                        sample_embeddings = [self.model.vocab[word].vector.tolist() for word in sample_words 
                                            if word in self.model.vocab and self.model.vocab[word].has_vector]
                        
                        if sample_embeddings:
                            self.compressor.fit(sample_embeddings)
                            logger.info(f"Initialized embedding compressor: {self.model.vocab.vectors.shape[1]} → {target_dimensions} dimensions")
                        
                # Set the final dimension based on compression setting
                if enable_compression:
                    self.dimension = target_dimensions
                else:
                    self.dimension = self.raw_dimension
            else:
                logger.warning(f"Model {model_name} has no word vectors. Using zeros.")
                self.raw_dimension = 300  # Default dimension
                
                # Set the final dimension based on compression setting
                if enable_compression:
                    self.dimension = target_dimensions
                else:
                    self.dimension = self.raw_dimension
                
                # Initialize compressor
                if enable_compression:
                    # Create random embeddings for fitting
                    sample_embeddings = [np.random.rand(self.raw_dimension).tolist() for _ in range(10)]
                    self.compressor.fit(sample_embeddings)
                    logger.info(f"Initialized embedding compressor: {self.raw_dimension} → {target_dimensions} dimensions")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            self.model = None
            self.raw_dimension = 300  # Default dimension
            if enable_compression:
                self.dimension = target_dimensions
            else:
                self.dimension = self.raw_dimension
    
    def _embed_text_internal(self, text: str) -> List[float]:
        """Internal method to embed text without caching or compression.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Get the original dimension for zero vectors
        orig_dim = self.model.vocab.vectors.shape[1]
        
        if not text:
            return [0.0] * orig_dim
        
        try:
            # Process the text with spaCy
            doc = self.model(text)
            
            # If the text is long enough, use the document vector
            if len(doc) > 0:
                # Use the average of word vectors as the document vector
                vector = np.mean([token.vector for token in doc if token.has_vector], axis=0)
                
                # If no tokens had vectors, return zeros
                if np.isnan(vector).any():
                    return [0.0] * orig_dim
                    
                return vector.tolist()
            else:
                return [0.0] * orig_dim
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            return [0.0] * orig_dim
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string with caching and compression if enabled.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first if enabled
        if self.use_cache:
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self._embed_text_internal(text)
        
        # Apply compression if enabled
        if self.enable_compression:
            logger.info(f"Compressing embedding from {len(embedding)} to {self.target_dimensions} dimensions")
            embedding = self.compressor.compress(embedding)
        else:
            logger.info(f"Using full embedding dimensions: {len(embedding)}")
        
        # Cache the result if enabled
        if self.use_cache:
            self.cache.set(text, embedding)
        
        return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed multiple texts with batching, caching, and compression.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Process in batches to handle memory constraints
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = [self.embed_text(text) for text in batch]
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch of texts: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimension for _ in texts]
    
    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 16, debug: bool = False, debug_dump_path: str = None) -> List[Dict]:
        """Embed document chunks with diagnostics and debug mode."""
        total_chunks = len(chunks)
        non_empty_chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
        if total_chunks != len(non_empty_chunks):
            logger.warning(f"Skipped {total_chunks - len(non_empty_chunks)} empty chunks during embedding.")
        chunks = non_empty_chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size)
        all_zero_count = sum(1 for emb in embeddings if np.allclose(emb, 0))
        logger.info(f"Embedding diagnostics: total={total_chunks}, non_empty={len(chunks)}, all_zero={all_zero_count}")
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                "id": chunk.id,
                "embedding": embedding,
                "metadata": chunk.metadata
            })
        if debug and debug_dump_path:
            try:
                with open(debug_dump_path, "w", encoding="utf-8") as f:
                    for i, chunk in enumerate(chunks[:20]):
                        f.write(f"Chunk {i}: {chunk.text[:200]}\n\n")
                logger.info(f"Dumped top-{min(20, len(chunks))} chunk texts to {debug_dump_path}")
            except Exception as e:
                logger.error(f"Failed to dump debug texts: {e}")
        return result
    
    def embed_section_weighted(self, section: Section, 
                              title_weight: float = 2.0,
                              first_para_weight: float = 1.5) -> List[float]:
        """Create a weighted embedding for a document section.
        
        Args:
            section: Document section
            title_weight: Weight for the section title
            first_para_weight: Weight for the first paragraph
            
        Returns:
            Weighted embedding vector
        """
        # Embed title and content separately
        title_embedding = np.array(self.embed_text(section.title))
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
        
        if not paragraphs:
            # Only title available
            return title_embedding.tolist()
        
        # Embed first paragraph
        first_para_embedding = np.array(self.embed_text(paragraphs[0]))
        
        # Embed remaining content if available
        remaining_content = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
        
        if remaining_content:
            content_embedding = np.array(self.embed_text(remaining_content))
            
            # Calculate weighted average
            total_weight = title_weight + first_para_weight + 1.0
            weighted_embedding = (
                (title_weight * title_embedding) + 
                (first_para_weight * first_para_embedding) + 
                content_embedding
            ) / total_weight
        else:
            # Only title and first paragraph
            total_weight = title_weight + first_para_weight
            weighted_embedding = (
                (title_weight * title_embedding) + 
                (first_para_weight * first_para_embedding)
            ) / total_weight
        
        return weighted_embedding.tolist()
    
    def update_embeddings(self, embeddings: List[Dict], new_chunks: List[Chunk]) -> List[Dict]:
        """Update embeddings incrementally with new chunks.
        
        Args:
            embeddings: Existing embeddings
            new_chunks: New chunks to embed
            
        Returns:
            Updated embeddings
        """
        # Create a dictionary of existing embeddings by ID
        embedding_dict = {item["id"]: item for item in embeddings}
        
        # Embed new chunks
        new_embeddings = self.embed_chunks(new_chunks)
        
        # Update dictionary with new embeddings
        for item in new_embeddings:
            embedding_dict[item["id"]] = item
        
        # Convert back to list
        return list(embedding_dict.values())
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # Handle zero vectors
            
        vec1 = vec1 / norm1
        vec2 = vec2 / norm2
        
        # Calculate cosine similarity
        return float(np.dot(vec1, vec2))
    
    def get_evaluator(self) -> EmbeddingEvaluator:
        """Get an embedding evaluator.
        
        Returns:
            Embedding evaluator instance
        """
        return EmbeddingEvaluator() 