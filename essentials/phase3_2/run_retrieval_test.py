"""
Test script for Phase 3.2 retrieval functionality
"""

import os
import json
import logging
import time
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variable to handle OpenMP duplicate lib issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import required modules
from essentials.phase3_1.models import Chunk
from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding
from essentials.phase3_2.vector_store import VectorStore
from essentials.phase3_2.retrieval import Retriever

def load_sample_data(filepath: str = "data/sample_chunks.json") -> List[Chunk]:
    """Load sample data from JSON file."""
    logger.info(f"Loading data from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        chunks = []
        for item in data:
            chunks.append(Chunk(
                id=item['id'],
                text=item['text'],
                source=item['source'],
                metadata=item['metadata']
            ))
        
        logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return []

def create_vector_store(chunks: List[Chunk]) -> VectorStore:
    """Create and populate a vector store with sample chunks."""
    print("\n=== Creating Vector Store ===")
    
    # Create embedding model
    model = AdvancedEmbedding(
        model_name="en_core_web_md",
        use_cache=True,
        enable_compression=False
    )
    
    # Create vector store with cosine similarity
    vector_store = VectorStore(dimension=model.dimension, index_type="cosine")
    print(f"Created vector store with dimension {model.dimension}")
    
    # Embed chunks
    print(f"Embedding {len(chunks)} chunks...")
    embedded_docs = model.embed_chunks(chunks)
    
    # Add to vector store
    vector_store.add_documents(embedded_docs)
    print(f"Added {len(embedded_docs)} documents to vector store")
    
    return vector_store, model

def test_basic_retrieval(vector_store: VectorStore, model: AdvancedEmbedding):
    """Test basic retrieval."""
    print("\n=== Testing Basic Retrieval ===")
    
    # Create retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=model
    )
    
    # Perform retrieval
    query = "quantum entanglement applications"
    print(f"Query: '{query}'")
    
    start_time = time.time()
    results = retriever.retrieve(query, k=3)
    print(f"Retrieved {len(results)} results in {time.time() - start_time:.4f}s")
    
    # Print results
    print("\nRESULTS:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] ID: {result['id']}")
        print(f"   Source: {result['metadata'].get('source', 'unknown')}")
        print(f"   Text: {result['metadata'].get('text', '')[:100]}...")
    
    return retriever

def test_hybrid_retrieval(retriever: Retriever):
    """Test hybrid retrieval."""
    print("\n=== Testing Hybrid Retrieval ===")
    
    # Perform retrieval
    query = "quantum physics applications"
    print(f"Query: '{query}'")
    
    start_time = time.time()
    results = retriever.hybrid_retrieve(
        query, 
        k=3,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    print(f"Retrieved {len(results)} results in {time.time() - start_time:.4f}s")
    
    # Print results
    print("\nRESULTS:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f}]")
        print(f"   ID: {result['id']}")
        print(f"   Source: {result['metadata'].get('source', 'unknown')}")

def test_reranking(retriever: Retriever):
    """Test citation-based reranking."""
    print("\n=== Testing Citation-Based Reranking ===")
    
    # Perform retrieval with reranking
    query = "quantum mechanics innovations"
    print(f"Query: '{query}'")
    
    start_time = time.time()
    results = retriever.retrieve_with_reranking(
        query, 
        k=3,
        initial_k=5
    )
    print(f"Retrieved {len(results)} results in {time.time() - start_time:.4f}s")
    
    # Print results
    print("\nRESULTS:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Citation: {result['citation_score']:.4f}]")
        print(f"   ID: {result['id']}")
        print(f"   Source: {result['metadata'].get('source', 'unknown')}")
        print(f"   Citation Score: {result['citation_score']:.4f}")

def test_dimension_adaptation():
    """Test dimension adaptation between different embedding sizes."""
    print("\n=== Testing Dimension Adaptation ===")
    
    # Create chunks
    chunks = load_sample_data()
    if not chunks:
        print("Failed to load sample data for dimension adaptation test")
        return
    
    # Create compressed model (dimension 100)
    compressed_model = AdvancedEmbedding(
        model_name="en_core_web_md",
        use_cache=False,
        enable_compression=True,
        target_dimensions=100
    )
    print(f"Created compressed model with dimension {compressed_model.dimension}")
    
    # Create standard model (dimension 300)
    standard_model = AdvancedEmbedding(
        model_name="en_core_web_md",
        use_cache=False,
        enable_compression=False
    )
    print(f"Created standard model with dimension {standard_model.dimension}")
    
    # Create vector store with standard dimension
    vector_store = VectorStore(dimension=standard_model.dimension, index_type="cosine")
    
    # Embed chunks with standard model
    embedded_docs = standard_model.embed_chunks(chunks)
    
    # Add to vector store
    vector_store.add_documents(embedded_docs)
    print(f"Added {len(embedded_docs)} documents with dimension {standard_model.dimension}")
    
    # Test query with compressed model
    query = "quantum physics"
    query_embedding = compressed_model.embed_text(query)
    print(f"Created query embedding with dimension {len(query_embedding)}")
    
    # Search vector store - this should adapt dimensions automatically
    try:
        results = vector_store.search(query_embedding, k=2)
        print(f"Successfully adapted dimensions and retrieved {len(results)} results")
        print("Dimension adaptation test passed!")
    except Exception as e:
        print(f"Dimension adaptation test failed: {str(e)}")

def main():
    print("\n*** PHASE 3.2 RETRIEVAL TEST ***\n")
    
    # Load sample data
    chunks = load_sample_data()
    if not chunks:
        print("Failed to load sample data")
        return
    
    # Create vector store
    vector_store, model = create_vector_store(chunks)
    
    # Test basic retrieval
    retriever = test_basic_retrieval(vector_store, model)
    
    # Test hybrid retrieval
    test_hybrid_retrieval(retriever)
    
    # Test citation-based reranking
    test_reranking(retriever)
    
    # Test dimension adaptation
    test_dimension_adaptation()
    
    print("\nRetrieval tests completed successfully!")

if __name__ == "__main__":
    main() 