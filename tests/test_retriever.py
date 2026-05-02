"""
Test for AdvancedRetriever functionality.
"""

import os
import sys
import tempfile
import numpy as np

print("Starting retriever test script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("Importing required classes...")
    from essentials.phase3_3.vector_store import ChromaVectorStore
    from essentials.phase3_3.retriever import AdvancedRetriever
    print("Import successful!")
except Exception as e:
    print(f"Import error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a temporary directory for ChromaDB
temp_dir = tempfile.mkdtemp()
print(f"Using temporary directory: {temp_dir}")

try:
    # Initialize vector store
    print("Initializing ChromaVectorStore...")
    vector_store = ChromaVectorStore(
        persist_directory=temp_dir,
        collection_name="test_collection",
        embedding_dimension=384
    )
    print("Successfully initialized ChromaVectorStore")
    
    # Create some test documents with embeddings
    print("Creating test documents...")
    docs = [
        {
            "id": "doc1",
            "text": "This is a test document about ChromaDB vector storage",
            "embedding": np.random.rand(384).tolist(),  # Random embedding
            "metadata": {"source": "test", "category": "vector_store"}
        },
        {
            "id": "doc2",
            "text": "Vector databases are useful for semantic search",
            "embedding": np.random.rand(384).tolist(),
            "metadata": {"source": "test", "category": "search"}
        },
        {
            "id": "doc3",
            "text": "ChromaDB provides persistent storage for embeddings",
            "embedding": np.random.rand(384).tolist(),
            "metadata": {"source": "test", "category": "vector_store"}
        },
        {
            "id": "doc4",
            "text": "Semantic search helps find similar content based on meaning",
            "embedding": np.random.rand(384).tolist(),
            "metadata": {"source": "test", "category": "search"}
        },
        {
            "id": "doc5",
            "text": "Hybrid search combines keyword matching with vector similarity",
            "embedding": np.random.rand(384).tolist(),
            "metadata": {"source": "test", "category": "search"}
        }
    ]
    
    # Add documents
    print("Adding documents...")
    vector_store.add_documents(docs)
    print(f"Added {len(docs)} documents")
    
    # Initialize retriever
    print("Initializing AdvancedRetriever...")
    retriever = AdvancedRetriever(vector_store)
    print("Successfully initialized AdvancedRetriever")
    
    # Test basic retrieval
    print("\n--- Testing basic retrieval ---")
    query = "vector database for search"
    results = retriever.retrieve(
        query=query,
        query_embedding=np.random.rand(384).tolist(),  # Random embedding for demo
        k=2
    )
    
    print(f"Basic retrieval returned {len(results)} results")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['text']} (score: {result['score']:.4f})")
    
    # Test hybrid retrieval
    print("\n--- Testing hybrid retrieval ---")
    hybrid_results = retriever.hybrid_retrieve(
        query=query,
        query_embedding=np.random.rand(384).tolist(),  # Random embedding for demo
        semantic_weight=0.7,
        keyword_weight=0.3,
        k=2
    )
    
    print(f"Hybrid retrieval returned {len(hybrid_results)} results")
    for i, result in enumerate(hybrid_results):
        print(f"Result {i+1}: {result['text']} (score: {result['score']:.4f})")
        print(f"  Semantic score: {result.get('semantic_score', 0):.4f}, Keyword score: {result.get('keyword_score', 0):.4f}")
    
    # Test filtered retrieval
    print("\n--- Testing filtered retrieval ---")
    filtered_results = retriever.filtered_retrieve(
        query=query,
        query_embedding=np.random.rand(384).tolist(),  # Random embedding for demo
        field_filters={"category": "vector_store"},
        k=2
    )
    
    print(f"Filtered retrieval returned {len(filtered_results)} results")
    for i, result in enumerate(filtered_results):
        print(f"Result {i+1}: {result['text']} (score: {result['score']:.4f}, category: {result['metadata']['category']})")
    
    print("\nRetriever test completed successfully!")
    
except Exception as e:
    print(f"Error during test: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up temporary directory
    import shutil
    try:
        print("Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Failed to clean up temporary directory: {temp_dir}")
        print(f"Error: {str(e)}") 