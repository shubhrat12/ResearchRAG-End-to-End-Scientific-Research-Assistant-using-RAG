"""ChromaDB vector store implementation for persistent document storage and retrieval."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.config import Settings
import numpy as np

from .vector_store import BaseVectorStore, VectorStoreError
from ..utils.logging import get_logger
from ..utils.types import Document, DocumentChunk, DocumentId, EmbeddingVector, FilePath

logger = get_logger(__name__)


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation with persistence."""
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        store_name: str = "documents",
        persist_directory: Optional[FilePath] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize ChromaDB vector store.
        
        Args:
            embedding_dimension: Dimension of embedding vectors
            store_name: Name of the vector store
            persist_directory: Directory for persistent storage (required)
            collection_name: Name of the ChromaDB collection
            
        Raises:
            VectorStoreError: If initialization fails
        """
        super().__init__(
            embedding_dimension=embedding_dimension,
            store_name=store_name,
            persist_directory=persist_directory,
        )
        
        # ChromaDB requires a persistent directory
        if not self.persist_directory:
            raise VectorStoreError("ChromaDBVectorStore requires a persist_directory")
        
        # Ensure the directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Set collection name
        self.collection_name = collection_name or store_name
        
        # Initialize client and collection
        try:
            # Initialize the ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get or create collection
            try:
                # Try to get existing collection
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None,  # We'll provide embeddings explicitly
                )
                logger.info(f"Using existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Create a new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # We'll provide embeddings explicitly
                    metadata={"dimension": embedding_dimension}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {str(e)}")
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")
    
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
        # Verify we have documents to add
        if not documents:
            return []
        
        # Process each document
        document_ids = []
        
        for doc in documents:
            # Extract document ID
            doc_id = doc.document_id
            document_ids.append(doc_id)
            
            # Add all chunks from the document
            try:
                # First, add document metadata
                metadata = {
                    "document_id": doc_id,
                    "title": doc.metadata.title if doc.metadata else "Untitled",
                    "source": doc.metadata.source if doc.metadata else None,
                    "is_doc_metadata": True,
                }
                
                # Then add all chunks
                for i, chunk in enumerate(doc.chunks):
                    # If embeddings are provided, use them; otherwise will need to compute externally
                    self.add_document_chunks([chunk], None if embeddings is None else [embeddings[i]])
            
            except Exception as e:
                logger.error(f"Error adding document {doc_id}: {str(e)}")
                raise VectorStoreError(f"Error adding document {doc_id}: {str(e)}")
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return document_ids
    
    def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[EmbeddingVector]] = None,
    ) -> List[str]:
        """Add document chunks to the vector store.
        
        Args:
            chunks: Document chunks to add
            embeddings: Pre-computed embeddings (must be provided)
            
        Returns:
            List[str]: List of chunk IDs
            
        Raises:
            VectorStoreError: If chunks cannot be added
        """
        # ChromaDB requires embeddings
        if not embeddings or len(chunks) != len(embeddings):
            raise VectorStoreError("ChromaDB requires embeddings for all chunks")
        
        # Validate embeddings
        self._validate_embeddings(embeddings)
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embedding_vectors = []
        
        for i, chunk in enumerate(chunks):
            # Generate ID if not provided
            chunk_id = chunk.chunk_id or f"chunk-{i}-{hash(chunk.text[:50])}"
            ids.append(chunk_id)
            
            # Extract text and metadata
            texts.append(chunk.text)
            
            # Prepare chunk metadata
            metadata = {}
            if chunk.metadata:
                # Copy metadata to avoid modifying the original
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
            
            metadatas.append(metadata)
            
            # Add embedding
            embedding_vectors.append(embeddings[i])
        
        try:
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                embeddings=embedding_vectors,
                documents=texts,
                metadatas=metadatas,
            )
            
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"ChromaDB add error: {str(e)}")
            raise VectorStoreError(f"Failed to add chunks to ChromaDB: {str(e)}")
    
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
        # Validate query vector
        if len(query_vector) != self.embedding_dimension:
            raise VectorStoreError(
                f"Query vector dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_vector)}"
            )
        
        try:
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract results
            documents = results["documents"][0]  # First query
            metadatas = results["metadatas"][0]  # First query
            distances = results["distances"][0]  # First query
            
            # Convert to similarity scores (1 - distance)
            # ChromaDB uses cosine distance by default, which is 1 - cosine similarity
            similarities = [1 - dist for dist in distances]
            
            # Convert to DocumentChunks
            chunks_with_scores = []
            for i in range(len(documents)):
                # Create DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=results["ids"][0][i],
                    text=documents[i],
                    metadata=metadatas[i],
                )
                
                # Add to results with similarity score
                chunks_with_scores.append((chunk, similarities[i]))
            
            logger.info(f"Found {len(chunks_with_scores)} chunks for query")
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"ChromaDB search error: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
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
            # In ChromaDB, we need to find all chunks for these documents
            # and delete them individually
            for doc_id in document_ids:
                # Find all chunks for this document
                results = self.collection.get(
                    where={"document_id": doc_id}
                )
                
                chunk_ids = results["ids"]
                if chunk_ids:
                    # Delete the chunks
                    self.collection.delete(ids=chunk_ids)
                    logger.info(f"Deleted {len(chunk_ids)} chunks for document {doc_id}")
                else:
                    logger.warning(f"No chunks found for document {doc_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB delete error: {str(e)}")
            raise VectorStoreError(f"Deletion failed: {str(e)}")
    
    def persist(self) -> bool:
        """Persist the vector store to disk.
        
        ChromaDB automatically persists data, so this is a no-op.
        
        Returns:
            bool: Whether persistence was successful
            
        Raises:
            VectorStoreError: If persistence fails
        """
        # ChromaDB automatically persists data, but we can trigger
        # a sync to ensure everything is saved
        try:
            # For Chroma, persistence is automatic
            # We're already using PersistentClient
            logger.debug("ChromaDB automatically persists data")
            return True
        except Exception as e:
            logger.error(f"ChromaDB persistence error: {str(e)}")
            raise VectorStoreError(f"Persistence failed: {str(e)}")
    
    def load(self) -> bool:
        """Load the vector store from disk.
        
        ChromaDB automatically loads data, so this is a no-op.
        
        Returns:
            bool: Whether loading was successful
            
        Raises:
            VectorStoreError: If loading fails
        """
        # ChromaDB automatically loads data on initialization
        # We already did this in the constructor
        logger.debug("ChromaDB automatically loads data on initialization")
        return True
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
            
        Raises:
            VectorStoreError: If stats cannot be retrieved
        """
        try:
            # Get all collection items to count
            results = self.collection.get()
            
            # Count items
            count = len(results["ids"]) if "ids" in results else 0
            
            return {
                "name": self.collection_name,
                "count": count,
                "dimension": self.embedding_dimension,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            logger.error(f"ChromaDB stats error: {str(e)}")
            raise VectorStoreError(f"Failed to get collection stats: {str(e)}")
    
    def reset_collection(self) -> bool:
        """Reset the collection by deleting all items.
        
        Returns:
            bool: Whether reset was successful
            
        Raises:
            VectorStoreError: If reset fails
        """
        try:
            # Delete collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,  # We'll provide embeddings explicitly
                metadata={"dimension": self.embedding_dimension}
            )
            
            logger.info(f"Reset ChromaDB collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB reset error: {str(e)}")
            raise VectorStoreError(f"Reset failed: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """List all collections in the ChromaDB instance.
        
        Returns:
            List[str]: Collection names
            
        Raises:
            VectorStoreError: If listing fails
        """
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"ChromaDB list collections error: {str(e)}")
            raise VectorStoreError(f"Failed to list collections: {str(e)}") 