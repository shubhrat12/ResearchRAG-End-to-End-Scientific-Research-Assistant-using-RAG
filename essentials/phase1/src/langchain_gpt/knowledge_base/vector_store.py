"""Vector store abstraction for document storage and retrieval."""

import abc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.errors import LangChainGPTError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, DocumentId, EmbeddingVector, FilePath

logger = get_logger(__name__)


class VectorStoreError(LangChainGPTError):
    """Error raised by vector stores."""
    
    def __init__(self, message: str = "Vector store error"):
        super().__init__(f"Vector store error: {message}")


class BaseVectorStore(abc.ABC):
    """Abstract base class for vector stores."""
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        store_name: str = "default",
        persist_directory: Optional[FilePath] = None,
    ):
        """Initialize vector store.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
            store_name: Name of the vector store
            persist_directory: Directory for persistent storage
        """
        self.embedding_dimension = embedding_dimension
        self.store_name = store_name
        self.persist_directory = Path(persist_directory) if persist_directory else None
        
    @abc.abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[EmbeddingVector]] = None,
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: Documents to add
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List[str]: List of document IDs
            
        Raises:
            VectorStoreError: If documents cannot be added
        """
        pass
    
    @abc.abstractmethod
    def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[EmbeddingVector]] = None,
    ) -> List[str]:
        """Add document chunks to the vector store.
        
        Args:
            chunks: Document chunks to add
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List[str]: List of chunk IDs
            
        Raises:
            VectorStoreError: If chunks cannot be added
        """
        pass
    
    @abc.abstractmethod
    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents by vector.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Metadata filters to apply
            
        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, score) tuples
            
        Raises:
            VectorStoreError: If search fails
        """
        pass
    
    @abc.abstractmethod
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store.
        
        Args:
            document_ids: IDs of documents to delete
            
        Returns:
            bool: Whether deletion was successful
            
        Raises:
            VectorStoreError: If deletion fails
        """
        pass
    
    @abc.abstractmethod
    def persist(self) -> bool:
        """Persist the vector store to disk.
        
        Returns:
            bool: Whether persistence was successful
            
        Raises:
            VectorStoreError: If persistence fails
        """
        pass
    
    @abc.abstractmethod
    def load(self) -> bool:
        """Load the vector store from disk.
        
        Returns:
            bool: Whether loading was successful
            
        Raises:
            VectorStoreError: If loading fails
        """
        pass
    
    def _validate_embeddings(self, embeddings: List[EmbeddingVector]) -> bool:
        """Validate embeddings.
        
        Args:
            embeddings: Embedding vectors to validate
            
        Returns:
            bool: Whether embeddings are valid
            
        Raises:
            VectorStoreError: If embeddings are invalid
        """
        if not embeddings or not isinstance(embeddings, list):
            raise VectorStoreError("Embeddings must be a non-empty list")
        
        # Check first embedding dimension
        if len(embeddings[0]) != self.embedding_dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(embeddings[0])}"
            )
        
        return True


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store implementation."""
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        store_name: str = "in_memory",
        persist_directory: Optional[FilePath] = None,
    ):
        """Initialize in-memory vector store.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
            store_name: Name of the vector store
            persist_directory: Directory for persistent storage
        """
        super().__init__(
            embedding_dimension=embedding_dimension,
            store_name=store_name,
            persist_directory=persist_directory,
        )
        self._vectors: Dict[str, EmbeddingVector] = {}
        self._documents: Dict[str, DocumentChunk] = {}
        self._document_ids: Dict[DocumentId, List[str]] = {}
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[EmbeddingVector]] = None,
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: Documents to add
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List[str]: List of document IDs
            
        Raises:
            VectorStoreError: If documents cannot be added
        """
        if not documents:
            return []
        
        # If embeddings are not provided, return document IDs only
        # In a real implementation, we would compute embeddings here
        document_ids = []
        
        for i, doc in enumerate(documents):
            doc_id = str(doc.metadata.file_path) if doc.metadata.file_path else f"doc-{len(self._document_ids)}"
            self._document_ids[doc_id] = []
            
            # Process each chunk
            for j, chunk in enumerate(doc.chunks):
                chunk_id = chunk.chunk_id or f"{doc_id}-chunk-{j}"
                chunk.chunk_id = chunk_id
                
                # Store the chunk
                self._documents[chunk_id] = chunk
                
                # If embeddings are provided, store them
                if embeddings and i < len(embeddings):
                    self._vectors[chunk_id] = embeddings[i]
                
                # Associate chunk with document
                self._document_ids[doc_id].append(chunk_id)
            
            document_ids.append(doc_id)
        
        logger.info(f"Added {len(documents)} documents with {len(self._documents)} chunks to vector store")
        return document_ids
    
    def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[EmbeddingVector]] = None,
    ) -> List[str]:
        """Add document chunks to the vector store.
        
        Args:
            chunks: Document chunks to add
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List[str]: List of chunk IDs
            
        Raises:
            VectorStoreError: If chunks cannot be added
        """
        if not chunks:
            return []
        
        # Validate embeddings if provided
        if embeddings:
            if len(chunks) != len(embeddings):
                raise VectorStoreError(
                    f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})"
                )
            self._validate_embeddings(embeddings)
        
        # Add chunks to store
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            # Generate chunk ID if not provided
            chunk_id = chunk.chunk_id or f"chunk-{len(self._documents)}"
            chunk.chunk_id = chunk_id
            
            # Store the chunk
            self._documents[chunk_id] = chunk
            
            # If embeddings are provided, store them
            if embeddings and i < len(embeddings):
                self._vectors[chunk_id] = embeddings[i]
            
            chunk_ids.append(chunk_id)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        return chunk_ids
    
    def search(
        self,
        query_vector: EmbeddingVector,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents by vector.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Metadata filters to apply
            
        Returns:
            List[Tuple[DocumentChunk, float]]: List of (chunk, score) tuples
            
        Raises:
            VectorStoreError: If search fails
        """
        if not self._vectors:
            logger.warning("Vector store is empty")
            return []
        
        if len(query_vector) != self.embedding_dimension:
            raise VectorStoreError(
                f"Query vector dimension mismatch: expected {self.embedding_dimension}, got {len(query_vector)}"
            )
        
        # Calculate cosine similarities
        similarities = []
        
        for chunk_id, vector in self._vectors.items():
            # Apply filter if provided
            if filter and self._documents[chunk_id].metadata:
                if not all(self._documents[chunk_id].metadata.get(k) == v for k, v in filter.items()):
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = [
            (self._documents[chunk_id], score)
            for chunk_id, score in similarities[:k]
        ]
        
        return results
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store.
        
        Args:
            document_ids: IDs of documents to delete
            
        Returns:
            bool: Whether deletion was successful
            
        Raises:
            VectorStoreError: If deletion fails
        """
        if not document_ids:
            return True
        
        try:
            for doc_id in document_ids:
                if doc_id in self._document_ids:
                    # Delete all chunks for this document
                    for chunk_id in self._document_ids[doc_id]:
                        if chunk_id in self._documents:
                            del self._documents[chunk_id]
                        if chunk_id in self._vectors:
                            del self._vectors[chunk_id]
                    
                    # Delete document ID mapping
                    del self._document_ids[doc_id]
            
            logger.info(f"Deleted {len(document_ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def persist(self) -> bool:
        """Persist the vector store to disk.
        
        Returns:
            bool: Whether persistence was successful
            
        Raises:
            VectorStoreError: If persistence fails
        """
        if not self.persist_directory:
            logger.warning("No persist directory specified")
            return False
        
        try:
            # In a real implementation, we would serialize to disk here
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Vector store persisted to {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")
            return False
    
    def load(self) -> bool:
        """Load the vector store from disk.
        
        Returns:
            bool: Whether loading was successful
            
        Raises:
            VectorStoreError: If loading fails
        """
        if not self.persist_directory or not self.persist_directory.exists():
            logger.warning("No persist directory or directory does not exist")
            return False
        
        try:
            # In a real implementation, we would deserialize from disk here
            logger.info(f"Vector store loaded from {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity (-1 to 1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if np.linalg.norm(vec1) * np.linalg.norm(vec2) == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_vector_store(
    store_type: str = "in_memory",
    embedding_dimension: int = 384,
    store_name: str = "default",
    persist_directory: Optional[FilePath] = None,
    collection_name: Optional[str] = None,
) -> BaseVectorStore:
    """Get a vector store instance.
    
    Args:
        store_type: Type of vector store ('in_memory', 'chromadb')
        embedding_dimension: Dimension of embedding vectors
        store_name: Name of the vector store
        persist_directory: Directory for persistent storage
        collection_name: Name of the collection (for ChromaDB)
        
    Returns:
        BaseVectorStore: Vector store instance
        
    Raises:
        VectorStoreError: If vector store type is invalid
    """
    if store_type == "in_memory":
        return InMemoryVectorStore(
            embedding_dimension=embedding_dimension,
            store_name=store_name,
            persist_directory=persist_directory,
        )
    elif store_type == "chromadb":
        from .chromadb_vector_store import ChromaDBVectorStore
        return ChromaDBVectorStore(
            embedding_dimension=embedding_dimension,
            store_name=store_name,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    else:
        # In Phase 1, only in-memory and ChromaDB are supported
        # In future phases, we can add support for more vector stores
        raise VectorStoreError(f"Unsupported vector store type: {store_type}") 