from typing import List, Dict, Any, Optional, Union
import os
import json
import pickle
import numpy as np
import faiss
from essentials.phase3_1.models import Chunk
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for embeddings using FAISS."""
    
    def __init__(self, dimension: int = 384, index_type: str = "L2"):
        """Initialize vector store.
        
        Args:
            dimension: Dimension of the embeddings
            index_type: Type of FAISS index (L2 or cosine)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "cosine":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.normalize = True
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Storage for document metadata
        self.documents = {}
    
    def _adapt_embedding_dimension(self, embedding: List[float], target_dim: int) -> List[float]:
        """Adapt embedding to target dimension by truncation or zero-padding.
        
        Args:
            embedding: Original embedding
            target_dim: Target dimension
            
        Returns:
            Adapted embedding
        """
        original_dim = len(embedding)
        
        if original_dim == target_dim:
            return embedding
        
        logger.warning(f"Adapting embedding from dimension {original_dim} to {target_dim}")
        
        if original_dim > target_dim:
            # Truncate
            return embedding[:target_dim]
        else:
            # Pad with zeros
            return embedding + [0.0] * (target_dim - original_dim)
        
    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[Dict]) -> None:
        """Add documents to the vector store.
        
        Args:
            ids: List of document IDs
            embeddings: List of document embeddings
            documents: List of document metadata
        """
        if len(ids) != len(embeddings) or len(ids) != len(documents):
            raise ValueError("Length of ids, embeddings, and documents must be the same")
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype("float32")
        
        # Check dimensions
        embedding_dim = embeddings_np.shape[1]
        if embedding_dim != self.dimension:
            logger.warning(f"Embedding dimension mismatch: got {embedding_dim}, expected {self.dimension}")
            
            # Handle dimension mismatch by recreating the index with the new dimension
            if len(self.documents) == 0:  # Only do this if the index is empty
                logger.info(f"Recreating index with dimension {embedding_dim}")
                self.dimension = embedding_dim
                
                # Recreate FAISS index
                if self.index_type == "L2":
                    self.index = faiss.IndexFlatL2(self.dimension)
                elif self.index_type == "cosine":
                    self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
            else:
                raise ValueError(f"Cannot add embeddings with dimension {embedding_dim} to index with dimension {self.dimension}")
        
        # Normalize if using cosine similarity
        if self.index_type == "cosine":
            faiss.normalize_L2(embeddings_np)
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        
        # Store document metadata
        for i, doc_id in enumerate(ids):
            self.documents[doc_id] = documents[i]
    
    def add_documents(self, embedded_documents: List[Dict]) -> None:
        """Add documents from embedded document dictionaries.
        
        Args:
            embedded_documents: List of dictionaries with id, embedding, and metadata
        """
        ids = [doc["id"] for doc in embedded_documents]
        embeddings = [doc["embedding"] for doc in embedded_documents]
        documents = [{"id": doc["id"], "metadata": doc["metadata"]} for doc in embedded_documents]
        self.add(ids, embeddings, documents)
        logger.info(f"VectorStore now contains {len(self.documents)} entries with dimension {self.dimension}.")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of results with document ID, score, and metadata
        """
        try:
            # Check if there are documents in the index
            if len(self.documents) == 0:
                logger.warning("Search called on empty vector store")
                return []
            
            # Check dimensions match
            query_dim = len(query_embedding)
            if query_dim != self.dimension:
                logger.warning(f"Query dimension mismatch: got {query_dim}, expected {self.dimension}")
                
                # Try to adapt the embedding to the right dimension
                try:
                    query_embedding = self._adapt_embedding_dimension(query_embedding, self.dimension)
                    query_dim = len(query_embedding)
                    logger.info(f"Adapted query embedding to dimension {query_dim}")
                except Exception as e:
                    logger.error(f"Failed to adapt embedding: {str(e)}")
                    raise ValueError(f"Query dimension mismatch: got {query_dim}, expected {self.dimension}")
            
            # Convert query to numpy array
            query_np = np.array([query_embedding]).astype("float32")
            
            # Normalize if using cosine similarity
            if self.index_type == "cosine":
                faiss.normalize_L2(query_np)
            
            logger.info(f"Searching with k={k}, index_type={self.index_type}, dimension={self.dimension}")
            
            # Search
            scores, indices = self.index.search(query_np, k)
            
            # Debug information about search results
            valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(self.documents)]
            logger.info(f"FAISS returned {len(valid_indices)} valid indices out of {k} requested")
            
            # Convert to list of results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.documents):  # FAISS may return -1 for not enough results
                    continue
                
                doc_id = list(self.documents.keys())[idx]
                results.append({
                    "id": doc_id,
                    "score": float(scores[0][i]),
                    "metadata": self.documents[doc_id]["metadata"]
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during vector store search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def save(self, directory: str) -> None:
        """Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save metadata
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump({
                "dimension": self.dimension,
                "index_type": self.index_type,
                "document_count": len(self.documents)
            }, f)
        
        # Save documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load vector store from disk.
        
        Args:
            directory: Directory containing the vector store
            
        Returns:
            Loaded vector store
        """
        # Load metadata
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(dimension=metadata["dimension"], index_type=metadata["index_type"])
        
        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            instance.documents = pickle.load(f)
        
        return instance

    def get_entry_count_and_dim(self):
        """Return the number of entries and embedding dimension."""
        return len(self.documents), self.dimension 