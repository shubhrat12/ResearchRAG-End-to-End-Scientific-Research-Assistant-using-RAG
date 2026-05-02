"""
Advanced document retrieval module for phase 3.3.

This module implements various retrieval strategies including:
- Semantic search using vector similarity
- Hybrid search combining keyword + vector retrieval
- Filtering by metadata fields
- Maximum Marginal Relevance (MMR) for diversity in results
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import logging
import numpy as np
import re
from collections import Counter
from essentials.phase3_1.models import Chunk
from essentials.phase3_3.vector_store import ChromaVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import embedding models with priority order and fallbacks
try:
    from essentials.phase3_2.scientific_embeddings import ScientificEmbedding
    logger.info("Successfully imported ScientificEmbedding")
    ScientificEmbeddingAvailable = True
except ImportError as e:
    logger.warning(f"ScientificEmbedding not available: {str(e)}")
    ScientificEmbeddingAvailable = False

try:
    from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding
    logger.info("Successfully imported AdvancedEmbedding")
    AdvancedEmbeddingAvailable = True
except ImportError as e:
    logger.warning(f"AdvancedEmbedding not available: {str(e)}")
    AdvancedEmbeddingAvailable = False

try:
    from essentials.phase3_2.basic_embeddings import BasicEmbedding
    logger.info("Successfully imported BasicEmbedding")
    BasicEmbeddingAvailable = True
except ImportError as e:
    logger.error(f"BasicEmbedding not available: {str(e)}")
    BasicEmbeddingAvailable = False

# Select best available embedding model class
if ScientificEmbeddingAvailable:
    EmbeddingClass = ScientificEmbedding
    logger.info("Using ScientificEmbedding for retrieval")
elif AdvancedEmbeddingAvailable:
    EmbeddingClass = AdvancedEmbedding
    logger.info("Using AdvancedEmbedding for retrieval")
elif BasicEmbeddingAvailable:
    EmbeddingClass = BasicEmbedding
    logger.info("Using BasicEmbedding for retrieval")
else:
    logger.warning("No embedding models available. Will use external embeddings.")
    EmbeddingClass = None

class AdvancedRetriever:
    """Advanced document retriever with multiple retrieval strategies."""
    
    def __init__(
        self, 
        vector_store: ChromaVectorStore,
        embedding_model = None,
        model_name: str = "general",
        use_cache: bool = True,
        enable_compression: bool = False
    ):
        """Initialize the advanced retriever.
        
        Args:
            vector_store: Vector store for embeddings
            embedding_model: Optional embedding model (created if not provided)
            model_name: Model name to use if creating a new embedding model
            use_cache: Whether to use caching for embeddings
            enable_compression: Whether to enable embedding compression
        """
        self.vector_store = vector_store
        
        # If embedding_model is provided, use it; otherwise create a new one
        if embedding_model is not None:
            self.embedding_model = embedding_model
        elif EmbeddingClass is not None:
            # Create embedding model based on available class
            try:
                if EmbeddingClass == ScientificEmbedding:
                    self.embedding_model = ScientificEmbedding(model_name=model_name)
                elif EmbeddingClass == AdvancedEmbedding:
                    self.embedding_model = AdvancedEmbedding(
                        model_name="en_core_web_md" if model_name == "general" else model_name,
                        use_cache=use_cache,
                        enable_compression=enable_compression
                    )
                else:
                    self.embedding_model = BasicEmbedding()
                logger.info(f"Created embedding model: {type(self.embedding_model).__name__}")
            except Exception as e:
                logger.error(f"Failed to create embedding model: {str(e)}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logger.warning("No embedding model available. Queries must provide embeddings.")
    
    def retrieve(
        self, 
        query: str,
        query_embedding: List[float] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Basic semantic retrieval using vector search.
        
        Args:
            query: Query string
            query_embedding: Optional pre-computed query embedding
            k: Number of results to return
            filter_metadata: Optional metadata filters
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of retrieved documents
        """
        try:
            # Get query embedding if not provided
            if query_embedding is None:
                if self.embedding_model is None:
                    raise ValueError("No embedding model available and no query embedding provided")
                query_embedding = self.embedding_model.embed_text(query)
            
            # Search with vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                filter_metadata=filter_metadata,
                k=k,
                include_embeddings=include_embeddings
            )
            
            return results
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            raise
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from a text string.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length
        
        Returns:
            List of keywords
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        words = text.split()
        
        # Filter short words and common stop words
        stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'by', 'with', 
                        'about', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'up', 'down'])
        
        keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
        
        return keywords
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate a keyword matching score.
        
        Args:
            text: Text to search in
            keywords: Keywords to search for
        
        Returns:
            Keyword score (0-1)
        """
        if not keywords:
            return 0.0
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in keywords if keyword in text)
        
        # Calculate score as fraction of keywords matched
        score = keyword_count / len(keywords)
        
        return score
    
    def hybrid_retrieve(
        self,
        query: str,
        query_embedding: List[float] = None,
        k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Query string
            query_embedding: Optional pre-computed query embedding
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            filter_metadata: Optional metadata filters
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of retrieved documents
        """
        try:
            # Get more candidates for re-ranking
            initial_k = min(k * 3, 100)  # Get more results to re-rank
            
            # Get query embedding if not provided
            if query_embedding is None:
                if self.embedding_model is None:
                    raise ValueError("No embedding model available and no query embedding provided")
                query_embedding = self.embedding_model.embed_text(query)
            
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Get semantic search results
            semantic_results = self.vector_store.search(
                query_embedding=query_embedding,
                filter_metadata=filter_metadata,
                k=initial_k,
                include_embeddings=include_embeddings
            )
            
            if not semantic_results:
                return []
            
            # Calculate combined scores
            for result in semantic_results:
                # Semantic score is already in the result
                semantic_score = result["score"]
                
                # Calculate keyword score
                keyword_score = self._calculate_keyword_score(result["text"], keywords)
                
                # Combined score
                combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
                
                # Update result score
                result["semantic_score"] = semantic_score
                result["keyword_score"] = keyword_score
                result["score"] = combined_score
            
            # Sort by combined score
            semantic_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top k results
            return semantic_results[:k]
        
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            raise
    
    def retrieve_with_mmr(
        self,
        query: str,
        query_embedding: List[float] = None,
        k: int = 5,
        initial_k: int = 20,
        lambda_param: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieval with Maximum Marginal Relevance for diversity.
        
        Args:
            query: Query string
            query_embedding: Optional pre-computed query embedding
            k: Number of results to return
            initial_k: Initial number of candidates to consider
            lambda_param: MMR lambda parameter (0-1). Higher values prioritize query relevance,
                         lower values prioritize diversity.
            filter_metadata: Optional metadata filters
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of retrieved documents
        """
        try:
            # Get query embedding if not provided
            if query_embedding is None:
                if self.embedding_model is None:
                    raise ValueError("No embedding model available and no query embedding provided")
                query_embedding = self.embedding_model.embed_text(query)
            
            # Get initial results
            initial_results = self.vector_store.search(
                query_embedding=query_embedding,
                filter_metadata=filter_metadata,
                k=initial_k,
                include_embeddings=True  # We need embeddings for MMR
            )
            
            if not initial_results or len(initial_results) <= 1:
                return initial_results[:k]
            
            # MMR algorithm
            selected_indices = []
            candidates = list(range(len(initial_results)))
            
            # Convert query embedding to numpy array
            query_embedding_np = np.array(query_embedding)
            
            # Convert all document embeddings to numpy arrays
            doc_embeddings = [np.array(doc["embedding"]) for doc in initial_results]
            
            for _ in range(min(k, len(initial_results))):
                if not candidates:
                    break
                    
                # Calculate MMR scores
                mmr_scores = []
                
                for i in candidates:
                    if not selected_indices:  # First selection just uses similarity to query
                        mmr_scores.append((i, cosine_similarity(doc_embeddings[i], query_embedding_np)))
                    else:
                        # Calculate similarity to query
                        sim_query = cosine_similarity(doc_embeddings[i], query_embedding_np)
                        
                        # Calculate maximum similarity to already selected documents
                        max_sim_selected = max(
                            cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                            for j in selected_indices
                        )
                        
                        # Calculate MMR score
                        mmr_score = lambda_param * sim_query - (1 - lambda_param) * max_sim_selected
                        mmr_scores.append((i, mmr_score))
                
                # Select document with highest MMR score
                selected_i, _ = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(selected_i)
                candidates.remove(selected_i)
            
            # Create final results in order of selection
            final_results = [initial_results[i] for i in selected_indices]
            
            # Remove embeddings if not requested
            if not include_embeddings:
                for result in final_results:
                    if "embedding" in result:
                        del result["embedding"]
                        
            return final_results
            
        except Exception as e:
            logger.error(f"Error in MMR retrieval: {str(e)}")
            raise
    
    def filtered_retrieve(
        self,
        query: str,
        query_embedding: List[float] = None,
        field_filters: Dict[str, Any] = None,
        section_type: Optional[str] = None,
        importance: Optional[str] = None,
        content_type: Optional[str] = None,
        k: int = 5,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieval with specific metadata filters.
        
        Args:
            query: Query string
            query_embedding: Optional pre-computed query embedding
            field_filters: Dictionary of field=value pairs for filtering
            section_type: Optional filter for section_type (abstract, conclusion, etc.)
            importance: Optional filter for importance (high, medium, low)
            content_type: Optional filter for content_type (table, formula, etc.)
            k: Number of results to return
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of retrieved documents
        """
        try:
            # Build combined filter
            filter_metadata = field_filters or {}
            
            if section_type:
                filter_metadata["section_type"] = section_type
                
            if importance:
                filter_metadata["importance"] = importance
                
            if content_type:
                filter_metadata["content_type"] = content_type
            
            # Use regular retrieve with the filter
            return self.retrieve(
                query=query,
                query_embedding=query_embedding,
                k=k,
                filter_metadata=filter_metadata,
                include_embeddings=include_embeddings
            )
            
        except Exception as e:
            logger.error(f"Error in filtered retrieval: {str(e)}")
            raise
    
    def contextual_retrieve(
        self,
        query: str,
        context_text: str = None,
        context_chunks: List[Dict] = None,
        query_embedding: List[float] = None,
        context_weight: float = 0.3,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieval considering conversation context to improve results.
        
        Args:
            query: Query string
            context_text: Optional context text (e.g., from conversation history)
            context_chunks: Optional context chunks (e.g., previously retrieved documents)
            query_embedding: Optional pre-computed query embedding
            context_weight: Weight for context influence (0-1)
            k: Number of results to return
            filter_metadata: Optional metadata filters
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of retrieved documents
        """
        try:
            if not context_text and not context_chunks:
                # No context provided, fall back to regular retrieval
                return self.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    k=k,
                    filter_metadata=filter_metadata,
                    include_embeddings=include_embeddings
                )
            
            # Get embedding model
            if self.embedding_model is None:
                raise ValueError("No embedding model available for contextual retrieval")
            
            # Get query embedding
            if query_embedding is None:
                query_embedding = self.embedding_model.embed_text(query)
                
            query_embedding_np = np.array(query_embedding)
            
            # Get context embedding
            if context_text:
                context_embedding = self.embedding_model.embed_text(context_text)
                context_embedding_np = np.array(context_embedding)
            elif context_chunks:
                # Average embeddings from context chunks if they have them
                if all("embedding" in chunk for chunk in context_chunks):
                    context_embeddings = [chunk["embedding"] for chunk in context_chunks]
                    context_embedding_np = np.mean([np.array(emb) for emb in context_embeddings], axis=0)
                else:
                    # Extract text from chunks and embed
                    context_text = " ".join(chunk.get("text", "") for chunk in context_chunks)
                    context_embedding = self.embedding_model.embed_text(context_text)
                    context_embedding_np = np.array(context_embedding)
            
            # Combine query and context embeddings
            combined_embedding = (1 - context_weight) * query_embedding_np + context_weight * context_embedding_np
            
            # Normalize
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            # Convert back to list for the vector store
            combined_embedding_list = combined_embedding.tolist()
            
            # Search with the combined embedding
            return self.vector_store.search(
                query_embedding=combined_embedding_list,
                filter_metadata=filter_metadata,
                k=k,
                include_embeddings=include_embeddings
            )
            
        except Exception as e:
            logger.error(f"Error in contextual retrieval: {str(e)}")
            raise


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    # Ensure vectors are normalized
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    return np.dot(a_norm, b_norm) 