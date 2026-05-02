"""Knowledge base module for document storage and retrieval."""

from .vector_store import BaseVectorStore, InMemoryVectorStore, VectorStoreError, get_vector_store
from .embeddings import EmbeddingError, EmbeddingService
from .chromadb_vector_store import ChromaDBVectorStore
from .sentence_transformer_embeddings import SentenceTransformerEmbeddings, SentenceTransformerError
from .retriever import get_retriever, SearchResult 