"""
Test script for ChromaDB vector store and advanced retrieval functionality.

This script demonstrates how to:
1. Initialize and use ChromaDB vector store
2. Convert and store document chunks
3. Perform various types of semantic search
4. Evaluate retrieval quality
"""

import sys
import logging
import os
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from essentials.phase3_1.models import Chunk
from essentials.phase3_1.chunking import chunk_document, chunk_fixed, chunk_by_paragraph
from essentials.phase3_3.vector_store import ChromaVectorStore
from essentials.phase3_3.retriever import AdvancedRetriever
from essentials.phase3_3.retrieval_evaluation import RetrievalEvaluator

# Try importing embedding models with fallbacks
try:
    from essentials.phase3_2.scientific_embeddings import ScientificEmbedding as EmbeddingModel
    print("Using ScientificEmbedding model")
except ImportError:
    try:
        from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding as EmbeddingModel
        print("Using AdvancedEmbedding model")
    except ImportError:
        try:
            from essentials.phase3_2.basic_embeddings import BasicEmbedding as EmbeddingModel
            print("Using BasicEmbedding model")
        except ImportError:
            print("No embedding models found, using placeholder")
            
            # Simple placeholder embedding model for demonstration
            class BasicEmbeddingPlaceholder:
                def __init__(self):
                    self.dimension = 384
                
                def embed_text(self, text):
                    # Generate placeholder embedding
                    import hashlib
                    hash_obj = hashlib.md5(text.encode('utf-8'))
                    seed = int(hash_obj.hexdigest(), 16) % (2**32)
                    np.random.seed(seed)
                    return np.random.normal(0, 1, self.dimension).tolist()
                
                def embed_texts(self, texts, batch_size=16):
                    return [self.embed_text(text) for text in texts]
                
                def embed_chunks(self, chunks, batch_size=16):
                    embedded_chunks = []
                    for chunk in chunks:
                        embedding = self.embed_text(chunk.text)
                        embedded_chunks.append({
                            "id": chunk.id,
                            "embedding": embedding,
                            "metadata": chunk.metadata or {}
                        })
                    return embedded_chunks
            
            EmbeddingModel = BasicEmbeddingPlaceholder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample text for testing
SAMPLE_TEXT = """
# Introduction to Vector Databases

Vector databases are specialized database systems designed to store, manage, and search high-dimensional vectors efficiently. These vectors typically represent embeddings of text, images, audio, or other data.

## Key Features

Vector databases offer several important features:

1. **Similarity Search**: Find vectors that are similar to a query vector.
2. **Metadata Filtering**: Filter search results based on metadata attributes.
3. **Scalability**: Handle millions or billions of vectors efficiently.
4. **Persistence**: Store vectors durably for later retrieval.

## Applications

Vector databases have numerous applications, including:

- Semantic search engines
- Recommendation systems
- Image similarity search
- Anomaly detection
- Natural language processing

# Technical Details

## Vector Embeddings

Vectors in these databases are numerical representations of data in high-dimensional space. For text, these embeddings capture semantic meaning, allowing similar concepts to be close in vector space.

## Indexing Methods

Efficient similarity search relies on indexing methods such as:

- Approximate Nearest Neighbors (ANN)
- Hierarchical Navigable Small World (HNSW)
- Inverted File Index (IVF)

These methods create data structures that significantly speed up similarity searches.

## Distance Metrics

Common distance metrics for measuring similarity include:

- Euclidean distance
- Cosine similarity
- Dot product
- Jaccard similarity

The choice of metric depends on the specific application and embedding characteristics.
"""

def create_and_populate_vector_store():
    """Create and populate a vector store with sample data."""
    # Create data directory if it doesn't exist
    data_dir = Path("data/chroma_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    
    # Initialize vector store
    vector_store = ChromaVectorStore(
        persist_directory=str(data_dir),
        collection_name="test_collection",
        embedding_dimension=embedding_model.dimension
    )
    
    # Generate chunks from sample text
    chunks_fixed = chunk_fixed(SAMPLE_TEXT, chunk_size=50, overlap=5)
    chunks_paragraph = chunk_by_paragraph(SAMPLE_TEXT)
    all_chunks = chunks_fixed + chunks_paragraph
    
    # Add metadata to chunks
    for i, chunk in enumerate(chunks_fixed):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["chunk_method"] = "fixed"
        chunk.metadata["section"] = "section_" + str(i // 3)
        
    for i, chunk in enumerate(chunks_paragraph):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["chunk_method"] = "paragraph"
        chunk.metadata["section"] = "section_" + str(i)
        if "# " in chunk.text or "## " in chunk.text:
            chunk.metadata["contains_heading"] = True
            
    # Create embeddings
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embedded_chunks = embedding_model.embed_chunks(all_chunks)
    
    # Add to vector store
    print("Adding chunks to vector store...")
    vector_store.add_chunks(all_chunks, [ec["embedding"] for ec in embedded_chunks])
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    return vector_store, embedding_model, all_chunks

def test_basic_retrieval(vector_store, embedding_model):
    """Test basic retrieval functionality."""
    print("\n=== Basic Retrieval Test ===")
    
    # Initialize retriever
    retriever = AdvancedRetriever(vector_store, embedding_model)
    
    # Test queries
    test_queries = [
        "What are vector databases?",
        "Explain similarity search methods",
        "What distance metrics are used in vector databases?",
        "Applications of vector embeddings"
    ]
    
    # Run queries
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=3)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. [Score: {result['score']:.4f}] {result['text'][:100]}...")

def test_filtered_retrieval(vector_store, embedding_model):
    """Test filtered retrieval functionality."""
    print("\n=== Filtered Retrieval Test ===")
    
    # Initialize retriever
    retriever = AdvancedRetriever(vector_store, embedding_model)
    
    # Test query with filters
    query = "Explain indexing methods"
    
    # Filter by metadata
    print(f"\nQuery: {query} (filtered to paragraph chunks only)")
    results = retriever.filtered_retrieve(
        query,
        field_filters={"chunk_method": "paragraph"},
        k=3
    )
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}, Method: {result['metadata'].get('chunk_method')}] {result['text'][:100]}...")

def test_hybrid_retrieval(vector_store, embedding_model):
    """Test hybrid retrieval functionality."""
    print("\n=== Hybrid Retrieval Test ===")
    
    # Initialize retriever
    retriever = AdvancedRetriever(vector_store, embedding_model)
    
    # Test hybrid query
    query = "ANN and HNSW indexing methods"
    
    print(f"\nQuery: {query}")
    print("Standard Retrieval:")
    std_results = retriever.retrieve(query, k=3)
    
    for i, result in enumerate(std_results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['text'][:100]}...")
    
    print("\nHybrid Retrieval (semantic + keyword):")
    hybrid_results = retriever.hybrid_retrieve(
        query,
        semantic_weight=0.7,
        keyword_weight=0.3,
        k=3
    )
    
    for i, result in enumerate(hybrid_results):
        print(f"{i+1}. [Score: {result['score']:.4f}, Semantic: {result.get('semantic_score', 0):.4f}, Keyword: {result.get('keyword_score', 0):.4f}] {result['text'][:100]}...")

def test_mmr_retrieval(vector_store, embedding_model):
    """Test MMR retrieval for diversity."""
    print("\n=== MMR Retrieval Test (Diversity) ===")
    
    # Initialize retriever
    retriever = AdvancedRetriever(vector_store, embedding_model)
    
    # Test query
    query = "Applications of vector databases"
    
    print(f"\nQuery: {query}")
    print("Standard Retrieval:")
    std_results = retriever.retrieve(query, k=4)
    
    for i, result in enumerate(std_results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['text'][:100]}...")
    
    print("\nMMR Retrieval (for diversity):")
    mmr_results = retriever.retrieve_with_mmr(
        query,
        lambda_param=0.5,  # Balance between relevance and diversity
        k=4,
        initial_k=10,
        include_embeddings=False
    )
    
    for i, result in enumerate(mmr_results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['text'][:100]}...")

def test_retrieval_evaluation(vector_store, embedding_model, chunks):
    """Test retrieval evaluation functionality."""
    print("\n=== Retrieval Evaluation Test ===")
    
    # Initialize retriever and evaluator
    retriever = AdvancedRetriever(vector_store, embedding_model)
    evaluator = RetrievalEvaluator(retriever, vector_store)
    
    # Create test queries and "ground truth" (for demonstration)
    # In a real scenario, you'd have human-labeled ground truth
    test_queries = [
        "What are vector databases?",
        "Explain indexing methods",
        "Distance metrics for similarity search",
        "Applications of vector search"
    ]
    
    # Mock ground truth for demo (just using IDs of chunks containing certain keywords)
    ground_truth = []
    for query in test_queries:
        keywords = query.lower().split()
        relevant_ids = []
        
        for chunk in chunks:
            if any(keyword in chunk.text.lower() for keyword in keywords):
                relevant_ids.append(chunk.id)
        
        ground_truth.append(relevant_ids)
    
    # Define retrieval functions to evaluate
    def standard_retrieval(query):
        return retriever.retrieve(query, k=5)
    
    def hybrid_retrieval(query):
        return retriever.hybrid_retrieve(query, k=5)
    
    def mmr_retrieval(query):
        return retriever.retrieve_with_mmr(query, k=5)
    
    # Evaluate different retrieval methods
    print("\nEvaluating standard retrieval...")
    std_results = evaluator.evaluate_retrieval(
        test_queries,
        ground_truth,
        standard_retrieval,
        k_values=[1, 3, 5],
        run_name="standard_retrieval"
    )
    
    print("\nEvaluating hybrid retrieval...")
    hybrid_results = evaluator.evaluate_retrieval(
        test_queries,
        ground_truth,
        hybrid_retrieval,
        k_values=[1, 3, 5],
        run_name="hybrid_retrieval"
    )
    
    print("\nEvaluating MMR retrieval...")
    mmr_results = evaluator.evaluate_retrieval(
        test_queries,
        ground_truth,
        mmr_retrieval,
        k_values=[1, 3, 5],
        run_name="mmr_retrieval"
    )
    
    # Compare results
    comparison = evaluator.compare_runs(
        ["standard_retrieval", "hybrid_retrieval", "mmr_retrieval"]
    )
    
    print("\nComparison of retrieval methods:")
    for run_name, metrics in comparison.items():
        print(f"\n{run_name}:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {metric}@{k}: {v:.4f}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Uncomment to visualize (requires matplotlib)
    # evaluator.visualize_comparison(comparison)
    
    return evaluator

def main():
    """Main test function."""
    print("=== ChromaDB Vector Store and Advanced Retrieval Test ===")
    
    try:
        # Create and populate vector store
        vector_store, embedding_model, chunks = create_and_populate_vector_store()
        
        # Run tests
        test_basic_retrieval(vector_store, embedding_model)
        test_filtered_retrieval(vector_store, embedding_model)
        test_hybrid_retrieval(vector_store, embedding_model)
        test_mmr_retrieval(vector_store, embedding_model)
        evaluator = test_retrieval_evaluation(vector_store, embedding_model, chunks)
        
        print("\n=== All tests completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 