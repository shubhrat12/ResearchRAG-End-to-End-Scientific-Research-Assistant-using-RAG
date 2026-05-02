"""
Simple test for ChromaVectorStore and AdvancedRetriever.
"""

import os
import sys
import tempfile
import numpy as np

print("Starting test script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in essentials/phase3_3: {os.listdir('essentials/phase3_3')}")

try:
    print("Importing ChromaVectorStore...")
    from essentials.phase3_3.vector_store import ChromaVectorStore
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
        }
    ]
    
    # Add documents
    print("Adding documents...")
    vector_store.add_documents(docs)
    print(f"Added {len(docs)} documents")
    
    # Get collection stats
    print("Getting collection stats...")
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Try a simple search
    print("Performing search...")
    query_embedding = np.random.rand(384).tolist()
    results = vector_store.search(
        query_embedding=query_embedding,
        k=2
    )
    
    print(f"Search returned {len(results)} results")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['text']} (score: {result['score']:.4f})")
    
    # Try a filtered search
    print("Performing filtered search...")
    filtered_results = vector_store.search(
        query_embedding=query_embedding,
        filter_metadata={"category": "vector_store"},
        k=2
    )
    
    print(f"\nFiltered search returned {len(filtered_results)} results")
    for i, result in enumerate(filtered_results):
        print(f"Result {i+1}: {result['text']} (score: {result['score']:.4f}, category: {result['metadata']['category']})")
    
    print("\nTest completed successfully!")
    
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