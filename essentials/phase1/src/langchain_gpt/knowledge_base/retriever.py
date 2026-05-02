"""Document retrieval interface for semantic search."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, EmbeddingVector, FilePath
from .embeddings import EmbeddingService, get_embedding_service
from .vector_store import BaseVectorStore, VectorStoreError, get_vector_store

logger = get_logger(__name__)


class RetrieverError(LangChainGPTError):
    """Error raised by retriever operations."""
    
    def __init__(self, message: str = "Retriever error"):
        super().__init__(f"Retriever error: {message}")


class DocumentRetriever:
    """Interface for retrieving documents from vector stores."""
    
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        embedding_dimension: int = 384,
        top_k: int = 5,
    ):
        """Initialize document retriever.
        
        Args:
            vector_store: Vector store for document storage
            embedding_service: Service for computing embeddings
            embedding_dimension: Dimension of embedding vectors
            top_k: Default number of results to retrieve
        """
        self.vector_store = vector_store or get_vector_store(
            embedding_dimension=embedding_dimension
        )
        self.embedding_service = embedding_service or get_embedding_service(
            dimension=embedding_dimension
        )
        self.top_k = top_k
    
    def index_document(self, document: Document) -> List[str]:
        """Index a document in the vector store.
        
        Args:
            document: Document to index
            
        Returns:
            List[str]: List of document IDs
            
        Raises:
            RetrieverError: If indexing fails
        """
        try:
            # Generate embeddings for document chunks
            embeddings = self.embedding_service.embed_document(document)
            
            # Add document to vector store
            return self.vector_store.add_documents([document], embeddings)
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise RetrieverError(f"Failed to index document: {str(e)}")
    
    def index_documents(self, documents: List[Document]) -> List[str]:
        """Index multiple documents in the vector store.
        
        Args:
            documents: Documents to index
            
        Returns:
            List[str]: List of document IDs
            
        Raises:
            RetrieverError: If indexing fails
        """
        if not documents:
            return []
        
        try:
            # Extract all chunks from documents
            all_chunks = []
            doc_map = {}  # Map chunk indices to document indices
            
            for i, doc in enumerate(documents):
                for chunk in doc.chunks:
                    all_chunks.append(chunk)
                    doc_map[len(all_chunks) - 1] = i
            
            # Generate embeddings for all chunks
            all_embeddings = self.embedding_service.embed_chunks(all_chunks)
            
            # Group chunks and embeddings by document
            doc_chunks_map = {}
            doc_embeddings_map = {}
            
            for i, (chunk, embedding) in enumerate(zip(all_chunks, all_embeddings)):
                doc_idx = doc_map[i]
                doc_chunks_map.setdefault(doc_idx, []).append(chunk)
                doc_embeddings_map.setdefault(doc_idx, []).append(embedding)
            
            # Add each document with its embeddings
            doc_ids = []
            for i, doc in enumerate(documents):
                # Set chunks for this document
                doc.chunks = doc_chunks_map.get(i, [])
                
                # Add to vector store
                doc_id = self.vector_store.add_documents(
                    [doc], 
                    [doc_embeddings_map.get(i, [])]
                )
                doc_ids.extend(doc_id)
            
            return doc_ids
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise RetrieverError(f"Failed to index documents: {str(e)}")
    
    def retrieve_similar(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve documents similar to a query.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filters to apply
            
        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, score) tuples
            
        Raises:
            RetrieverError: If retrieval fails
        """
        if not query:
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search vector store
            return self.vector_store.search(
                query_vector=query_embedding,
                k=k or self.top_k,
                filter=filter,
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise RetrieverError(f"Failed to retrieve documents: {str(e)}")
    
    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        keyword_weight: float = 0.3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filters to apply
            keyword_weight: Weight for keyword matching (0-1)
            
        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, score) tuples
            
        Raises:
            RetrieverError: If search fails
        """
        if not query:
            return []
        
        try:
            # Get semantic search results
            semantic_results = self.retrieve_similar(query, k=k or self.top_k, filter=filter)
            
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Re-rank results with keyword matching
            if keywords and keyword_weight > 0:
                reranked_results = []
                
                for chunk, score in semantic_results:
                    # Calculate keyword match score
                    keyword_score = self._calculate_keyword_score(chunk.text, keywords)
                    
                    # Combine scores
                    combined_score = (1 - keyword_weight) * score + keyword_weight * keyword_score
                    
                    reranked_results.append((chunk, combined_score))
                
                # Sort by combined score
                reranked_results.sort(key=lambda x: x[1], reverse=True)
                
                # Limit to top k
                return reranked_results[:k or self.top_k]
            
            return semantic_results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise RetrieverError(f"Failed to perform hybrid search: {str(e)}")
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store.
        
        Args:
            document_ids: IDs of documents to delete
            
        Returns:
            bool: Whether deletion was successful
            
        Raises:
            RetrieverError: If deletion fails
        """
        try:
            return self.vector_store.delete(document_ids)
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise RetrieverError(f"Failed to delete documents: {str(e)}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of keywords
        """
        # Simple keyword extraction (remove common words, keep nouns)
        # In a real implementation, this would use NLP techniques
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words (minimal stop word list)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score for text.
        
        Args:
            text: Text to score
            keywords: List of keywords
            
        Returns:
            float: Keyword match score (0-1)
        """
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # Normalize score (0-1)
        return min(matches / len(keywords), 1.0)


def get_document_retriever(
    vector_store_type: str = "in_memory",
    embedding_model: str = "mock",
    embedding_dimension: int = 384,
    top_k: int = 5,
    persist_directory: Optional[FilePath] = None,
) -> DocumentRetriever:
    """Get a document retriever instance.
    
    Args:
        vector_store_type: Type of vector store to use
        embedding_model: Name of embedding model
        embedding_dimension: Dimension of embedding vectors
        top_k: Default number of results to retrieve
        persist_directory: Directory for persistent storage
        
    Returns:
        DocumentRetriever: Document retriever instance
    """
    # Create vector store
    vector_store = get_vector_store(
        store_type=vector_store_type,
        embedding_dimension=embedding_dimension,
        persist_directory=persist_directory,
    )
    
    # Create embedding service
    embedding_service = get_embedding_service(
        model=embedding_model,
        dimension=embedding_dimension,
    )
    
    # Create retriever
    return DocumentRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        embedding_dimension=embedding_dimension,
        top_k=top_k,
    ) 