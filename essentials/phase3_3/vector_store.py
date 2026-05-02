"""
ChromaDB Vector Store implementation for phase 3.3.

This module provides a persistent vector store implementation using ChromaDB with support for:
- Collection management (create, reset, delete)
- Document chunking conversion and storage
- Metadata filtering
- Vector similarity search
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import os
import logging
import json
import uuid
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from essentials.phase3_1.models import Chunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Vector store implementation using ChromaDB with persistent storage."""
    
    def __init__(
        self, 
        persist_directory: str = "data/chroma_db",
        collection_name: str = "default",
        embedding_function = None,
        embedding_dimension: int = 384,
    ):
        """Initialize the ChromaDB vector store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection to use
            embedding_function: Optional custom embedding function
            embedding_dimension: Dimension of the embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Set up embedding function - default to None which means embeddings will be provided directly
        self.embedding_function = embedding_function
        
        # Initialize or get collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"dimension": embedding_dimension}
            )
            logger.info(f"Initialized collection '{collection_name}' with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of dictionaries with id, embedding, text, and metadata.
                      Each document must have at least id, text, and metadata.
                      If embedding is not provided, embedding_function must be set.
        
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        # Check if we have embeddings or an embedding function
        has_embeddings = "embedding" in documents[0]
        
        if not has_embeddings and self.embedding_function is None:
            raise ValueError("Either document embeddings or an embedding function must be provided")
        
        # Extract document components
        ids = [str(doc.get("id", str(uuid.uuid4()))) for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Use embeddings if provided
        embeddings = None
        if has_embeddings:
            embeddings = [doc["embedding"] for doc in documents]
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]] = None) -> List[str]:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Optional list of embeddings (must match chunks length)
        
        Returns:
            List of document IDs
        """
        # Convert chunks to document dictionaries
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata or {}
            }
            
            # Add embedding if provided
            if embeddings and i < len(embeddings):
                doc["embedding"] = embeddings[i]
            
            documents.append(doc)
        
        return self.add_documents(documents)
    
    def search(
        self, 
        query_embedding: List[float] = None, 
        query_text: str = None,
        filter_metadata: Dict[str, Any] = None,
        k: int = 5,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector (optional if query_text and embedding_function provided)
            query_text: Query text (optional if query_embedding provided)
            filter_metadata: Optional metadata filters
            k: Number of results to return
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of results with document ID, score, text, and metadata
        """
        if query_embedding is None and query_text is None:
            raise ValueError("Either query_embedding or query_text must be provided")
        
        if query_embedding is None and self.embedding_function is None:
            raise ValueError("If query_embedding is not provided, embedding_function must be set")
        
        try:
            # Prepare search parameters
            where = filter_metadata
            
            # Execute search
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    where=where,
                    n_results=k,
                    include=["documents", "metadatas", "distances", "embeddings"] if include_embeddings else ["documents", "metadatas", "distances"]
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    where=where,
                    n_results=k,
                    include=["documents", "metadatas", "distances", "embeddings"] if include_embeddings else ["documents", "metadatas", "distances"]
                )
            
            # Format results
            formatted_results = []
            
            # ChromaDB returns results in a dictionary with lists
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, doc_id in enumerate(results["ids"][0]):
                    result = {
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i]  # Convert distance to similarity score
                    }
                    
                    # Add embedding if requested
                    if include_embeddings and "embeddings" in results:
                        result["embedding"] = results["embeddings"][0][i]
                    
                    formatted_results.append(result)
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": self.embedding_dimension,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise
    
    def delete_collection(self) -> bool:
        """Delete the collection.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def reset_collection(self) -> bool:
        """Reset the collection by deleting and recreating it.
        
        Returns:
            True if successful
        """
        try:
            # Delete collection if it exists
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
            except:
                pass
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"dimension": self.embedding_dimension}
            )
            logger.info(f"Recreated collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections in the Chroma database.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
    
    def get_or_create_collection(self, collection_name: str) -> "ChromaVectorStore":
        """Get or create a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            New ChromaVectorStore instance with the specified collection
        """
        return ChromaVectorStore(
            persist_directory=self.persist_directory,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            embedding_dimension=self.embedding_dimension
        )
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from the collection.
        
        Args:
            document_ids: List of document IDs to delete
        
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def get(self, document_ids: List[str], include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Get documents by ID.
        
        Args:
            document_ids: List of document IDs
            include_embeddings: Whether to include embeddings in results
        
        Returns:
            List of documents
        """
        try:
            results = self.collection.get(
                ids=document_ids,
                include=["documents", "metadatas", "embeddings"] if include_embeddings else ["documents", "metadatas"]
            )
            
            # Format results
            formatted_results = []
            
            for i, doc_id in enumerate(results["ids"]):
                result = {
                    "id": doc_id,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                
                # Add embedding if requested
                if include_embeddings and "embeddings" in results:
                    result["embedding"] = results["embeddings"][i]
                
                formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            raise 