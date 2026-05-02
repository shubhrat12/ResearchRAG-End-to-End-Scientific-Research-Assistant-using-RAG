from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import logging
from essentials.phase3_1.models import Chunk
from essentials.phase3_2.vector_store import VectorStore
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
    raise ImportError("No embedding models available")

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
    raise ImportError("No embedding models available")

class Retriever:
    """Advanced document retriever for scientific RAG."""
    debug_retrieval_dump: bool = False  # Config flag for debug dump
    debug_retrieval_dump_path: str = str(PROJECT_ROOT / "logs" / "retrieval_debug.txt")

    def __init__(
        self, 
        vector_store: VectorStore,
        embedding_model = None,
        model_name: str = "general",
        use_cache: bool = True,
        enable_compression: bool = False,
        debug_retrieval_dump: bool = False
    ):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store for embeddings
            embedding_model: Optional embedding model (created if not provided)
            model_name: Model name to use if creating a new embedding model
            use_cache: Whether to use caching for embeddings
            enable_compression: Whether to enable embedding compression
            debug_retrieval_dump: Enable debug dump of retrieved texts
        """
        self.vector_store = vector_store
        self.debug_retrieval_dump = debug_retrieval_dump
        self.debug_retrieval_dump_path = str(PROJECT_ROOT / "logs" / "retrieval_debug.txt")
        os.makedirs(os.path.dirname(self.debug_retrieval_dump_path), exist_ok=True)
        
        # If embedding_model is provided, use it; otherwise create a new one
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            # Select appropriate embedding model based on availability
            if EmbeddingClass == ScientificEmbedding:
                try:
                    self.embedding_model = ScientificEmbedding(model_name=model_name)
                    logger.info(f"Created ScientificEmbedding with model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to create ScientificEmbedding: {str(e)}")
                    if AdvancedEmbeddingAvailable:
                        logger.info("Falling back to AdvancedEmbedding")
                        self.embedding_model = AdvancedEmbedding(
                            model_name="en_core_web_md",
                            use_cache=use_cache,
                            enable_compression=enable_compression
                        )
                    else:
                        logger.info("Falling back to BasicEmbedding")
                        self.embedding_model = BasicEmbedding()
            elif EmbeddingClass == AdvancedEmbedding:
                try:
                    self.embedding_model = AdvancedEmbedding(
                        model_name="en_core_web_md" if model_name == "general" else model_name,
                        use_cache=use_cache,
                        enable_compression=enable_compression
                    )
                    logger.info(f"Created AdvancedEmbedding with model")
                except Exception as e:
                    logger.warning(f"Failed to create AdvancedEmbedding: {str(e)}")
                    logger.info("Falling back to BasicEmbedding")
                    self.embedding_model = BasicEmbedding()
            else:
                self.embedding_model = BasicEmbedding()
                logger.info("Created BasicEmbedding")
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        debug_dump_path: str = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents based on a query with diagnostics and debug mode."""
        try:
            logger.info(f"Embedding model: {type(self.embedding_model).__name__}")
            logger.info(f"Embedding model expected dimension: {self.embedding_model.dimension}")
            if hasattr(self.embedding_model, 'enable_compression'):
                logger.info(f"Compression enabled: {self.embedding_model.enable_compression}")
            else:
                logger.info("Compression attribute not available for this embedding model.")
            logger.info(f"Embedding query: {query}")
            query_embedding = self.embedding_model.embed_text(query)
            norm = np.linalg.norm(query_embedding)
            logger.info(f"Query embedding norm: {norm:.6f}, shape: {np.shape(query_embedding)}")
            if np.allclose(query_embedding, 0):
                logger.warning("Query embedding is all zeros!")
            elif norm < 1e-3:
                logger.warning(f"Query embedding norm is very small: {norm:.6f}")
            results = self.vector_store.query(query_embeddings=[query_embedding], n_results=k*2)
            if isinstance(results, dict):
                documents = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                # Sort by distance if available
                if distances and len(distances) == len(documents):
                    sorted_results = sorted(zip(documents, metas, distances), key=lambda x: x[2])
                else:
                    sorted_results = list(zip(documents, metas, [None]*len(documents)))
                # Take top-k
                top_results = sorted_results[:k]
                if not top_results:
                    logger.warning("No documents returned.")
                    return []
                logger.info(f"Retrieved {len(top_results)} documents.")
                for doc, meta, dist in top_results:
                    logger.info(f"[Chunk] {doc[:100]}... | Metadata: {meta} | Distance: {dist}")
                return [{"text": doc, "metadata": meta, "distance": dist} for doc, meta, dist in top_results]
            else:
                logger.warning(f"Results is not a dict (type: {type(results)}), cannot parse. Fallback to empty list.")
                return []
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _match_metadata(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria.
        
        Args:
            metadata: Document metadata
            filter_criteria: Filter criteria
            
        Returns:
            True if metadata matches all filter criteria
        """
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            # Handle list values (any match)
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            # Handle exact match
            elif metadata[key] != value:
                return False
        
        return True
    
    def hybrid_retrieve(
        self,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        debug_dump_path: str = None
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining semantic and keyword search with diagnostics and debug mode."""
        total_weight = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        semantic_results = self.retrieve(query, k=k*2, filter_metadata=filter_metadata)
        keywords = self._extract_keywords(query)
        results_with_scores = []
        for result in semantic_results:
            semantic_score = result.get("score", 0)
            # Fallback: use chunk.text if metadata['text'] missing
            doc_text = result.get("metadata", {}).get("text", result.get("text", ""))
            if not doc_text and "source" in result.get("metadata", {}):
                doc_text = result["metadata"]["source"]
            keyword_score = self._calculate_keyword_score(doc_text, keywords)
            combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            results_with_scores.append({
                **result,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score
            })
            if not doc_text:
                logger.warning(f"Chunk ID {result.get('id', 'N/A')} missing or empty metadata['text'] and text.")
        results_with_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        logger.info(f"Hybrid retrieval: Top-{min(k, len(results_with_scores))} IDs and scores:")
        for i, r in enumerate(results_with_scores[:k]):
            logger.info(f"  {i+1}. ID: {r.get('id', 'N/A')}, Combined Score: {r.get('combined_score', 0):.4f}")
        if not results_with_scores:
            logger.warning(f"No hybrid results for query: '{query}' (tokens: {len(query.split())})")
            return [{"id": "fallback", "text": "The document contains no relevant results for this query.", "score": 0.0, "metadata": {}}]
        if debug and debug_dump_path:
            try:
                with open(debug_dump_path, "w", encoding="utf-8") as f:
                    for i, r in enumerate(results_with_scores[:20]):
                        text = r.get('metadata', {}).get('text', r.get('text', ''))
                        f.write(f"Chunk {i}: {text[:200]}\n\n")
                logger.info(f"Dumped top-{min(20, len(results_with_scores))} hybrid retrieved texts to {debug_dump_path}")
            except Exception as e:
                logger.error(f"Failed to dump debug texts: {e}")
        return results_with_scores[:k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query.
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - split by spaces and remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                        "in", "on", "at", "to", "for", "with", "by", "about", "of"}
        
        words = query.lower().split()
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword score.
        
        Args:
            text: Document text
            keywords: List of keywords
            
        Returns:
            Keyword score
        """
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        
        # Count occurrences of each keyword
        keyword_counts = {}
        for keyword in keywords:
            keyword_counts[keyword] = text_lower.count(keyword)
        
        # Calculate score based on keyword occurrences
        total_occurrences = sum(keyword_counts.values())
        unique_keywords_found = sum(1 for count in keyword_counts.values() if count > 0)
        
        # Normalize by text length and number of keywords
        text_length_factor = min(1.0, 1000 / max(100, len(text_lower)))
        keyword_coverage = unique_keywords_found / len(keywords) if keywords else 0
        
        score = (total_occurrences * text_length_factor * keyword_coverage)
        
        # Normalize to 0-1 range
        normalized_score = min(1.0, score / max(1, len(keywords)))
        
        return normalized_score
    
    def retrieve_with_reranking(
        self,
        query: str,
        k: int = 5,
        initial_k: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents with citation-based reranking.
        
        Args:
            query: Query string
            k: Number of results to return
            initial_k: Number of initial results to consider
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        # Get initial results
        initial_results = self.retrieve(query, k=initial_k, filter_metadata=filter_metadata)
        
        if not initial_results:
            return []
        
        # Extract document IDs and citation information
        doc_ids = [result["id"] for result in initial_results]
        
        # Get citation counts from metadata if available
        citation_scores = {}
        for result in initial_results:
            doc_id = result["id"]
            metadata = result.get("metadata", {})
            
            # Look for citation information in metadata
            citation_count = metadata.get("citation_count", 0)
            
            # Normalize citation scores to 0-1 range
            # Add a small value to avoid division by zero
            citation_scores[doc_id] = min(1.0, citation_count / 100)
        
        # Rerank results based on combined relevance and citation scores
        reranked_results = []
        for result in initial_results:
            doc_id = result["id"]
            semantic_score = result["score"]
            citation_score = citation_scores.get(doc_id, 0)
            
            # Calculate combined score (70% semantic, 30% citation)
            combined_score = (0.7 * semantic_score) + (0.3 * citation_score)
            
            reranked_results.append({
                **result,
                "semantic_score": semantic_score,
                "citation_score": citation_score,
                "combined_score": combined_score
            })
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top k results
        return reranked_results[:k]
    
    def retrieve_with_mmr(self, query: str, k: int = 5, lambda_param: float = 0.5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using Maximum Marginal Relevance (MMR) for diversity.
        
        Args:
            query: Query string
            k: Number of results to return
            lambda_param: Trade-off parameter between relevance and diversity (0-1)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        # Embed the query
        query_embedding = self.embedding_model.embed_text(query)
        
        # Perform initial retrieval
        initial_results = self.vector_store.search(query_embedding, k=k*10)
        
        # Initialize selected results
        selected_results = []
        
        # MMR selection process
        while len(selected_results) < k and initial_results:
            # Calculate MMR score for each candidate
            mmr_scores = []
            for candidate in initial_results:
                relevance = candidate['score']
                diversity = max([self._cosine_similarity(candidate['embedding'], selected['embedding']) for selected in selected_results], default=0)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((mmr_score, candidate))
            
            # Select candidate with highest MMR score
            mmr_scores.sort(reverse=True, key=lambda x: x[0])
            best_candidate = mmr_scores[0][1]
            selected_results.append(best_candidate)
            initial_results.remove(best_candidate)
        
        return selected_results

    # Helper function for cosine similarity
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

    def evaluate_retrieval(self, queries: List[str], ground_truth: Dict[str, List[str]], k: int = 5) -> Dict[str, float]:
        """Evaluate retrieval quality using precision, recall, F1-score, and MAP.
        
        Args:
            queries: List of query strings
            ground_truth: Dictionary mapping queries to lists of relevant document IDs
            k: Number of top results to consider for each query
            
        Returns:
            Dictionary with evaluation metrics
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []
        average_precisions = []
        
        for query in queries:
            relevant_docs = set(ground_truth.get(query, []))
            retrieved_docs = set([doc['id'] for doc in self.retrieve(query, k=k)])
            
            # Calculate precision, recall, and F1-score
            true_positives = len(relevant_docs & retrieved_docs)
            precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
            recall = true_positives / len(relevant_docs) if relevant_docs else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
            
            # Calculate average precision
            sorted_retrieved = list(retrieved_docs)[:k]
            average_precision = sum((i + 1) / (rank + 1) for i, rank in enumerate(sorted_retrieved) if rank in relevant_docs) / len(relevant_docs) if relevant_docs else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1_score)
            average_precisions.append(average_precision)
        
        # Calculate mean metrics
        mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        mean_average_precision = sum(average_precisions) / len(average_precisions) if average_precisions else 0
        
        return {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "mean_average_precision": mean_average_precision
        }

    def plot_retrieval_performance(self, evaluation_results: Dict[str, float]):
        """Generate visualizations for retrieval performance.
        
        Args:
            evaluation_results: Dictionary with evaluation metrics
        """
        # Bar chart for mean metrics
        metrics = ['mean_precision', 'mean_recall', 'mean_f1', 'mean_average_precision']
        values = [evaluation_results[metric] for metric in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        plt.title('Retrieval Performance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.ylim(0, 1)
        plt.show()

        # Precision-Recall Curve (if applicable)
        # Note: This requires precision and recall values at different thresholds
        # For simplicity, this example assumes a single precision-recall point
        plt.figure(figsize=(10, 6))
        plt.plot(evaluation_results['mean_recall'], evaluation_results['mean_precision'], 'bo-', label='Precision-Recall')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.show() 