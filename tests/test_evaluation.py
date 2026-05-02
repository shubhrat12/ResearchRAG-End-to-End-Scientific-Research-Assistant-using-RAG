"""
Test for RetrievalEvaluator functionality.
"""

import os
import sys
import tempfile
import numpy as np

print("Starting evaluation test script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("Importing required classes...")
    from essentials.phase3_3.vector_store import ChromaVectorStore
    from essentials.phase3_3.retriever import AdvancedRetriever
    from essentials.phase3_3.retrieval_evaluation import RetrievalEvaluator
    print("Import successful!")
except Exception as e:
    print(f"Import error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a temporary directory for ChromaDB
temp_dir = tempfile.mkdtemp()
eval_dir = tempfile.mkdtemp()
print(f"Using temporary directory for ChromaDB: {temp_dir}")
print(f"Using temporary directory for evaluation results: {eval_dir}")

try:
    # Initialize vector store
    print("Initializing ChromaVectorStore...")
    vector_store = ChromaVectorStore(
        persist_directory=temp_dir,
        collection_name="test_collection",
        embedding_dimension=384
    )
    print("Successfully initialized ChromaVectorStore")
    
    # Create test documents with embeddings
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
    
    # Initialize evaluator
    print("Initializing RetrievalEvaluator...")
    evaluator = RetrievalEvaluator(
        retriever=retriever,
        vector_store=vector_store,
        results_dir=eval_dir
    )
    print("Successfully initialized RetrievalEvaluator")
    
    # Set up test queries and ground truth
    print("Setting up test queries and ground truth...")
    test_queries = [
        "vector database for search",
        "persistent storage for embeddings"
    ]
    
    # Mock ground truth - in a real scenario this would be human-labeled
    ground_truth = [
        ["doc2", "doc4", "doc5"],  # Relevant docs for query 1
        ["doc1", "doc3"]           # Relevant docs for query 2
    ]
    
    # Define retrieval functions to evaluate
    def standard_retrieval(query):
        return retriever.retrieve(
            query=query,
            query_embedding=np.random.rand(384).tolist(),  # Random embedding for demo
            k=3
        )
    
    def hybrid_retrieval(query):
        return retriever.hybrid_retrieve(
            query=query,
            query_embedding=np.random.rand(384).tolist(),  # Random embedding for demo
            semantic_weight=0.7,
            keyword_weight=0.3,
            k=3
        )
    
    # Evaluate standard retrieval
    print("\n--- Evaluating standard retrieval ---")
    std_results = evaluator.evaluate_retrieval(
        queries=test_queries,
        ground_truth=ground_truth,
        retrieval_fn=standard_retrieval,
        k_values=[1, 2, 3],
        run_name="standard_retrieval"
    )
    
    # Print metrics
    print("\nStandard Retrieval Metrics:")
    for k in [1, 2, 3]:
        print(f"  Precision@{k}: {std_results['precision'][k]:.4f}")
        print(f"  Recall@{k}: {std_results['recall'][k]:.4f}")
        print(f"  F1@{k}: {std_results['f1'][k]:.4f}")
    print(f"  MAP: {std_results['map']:.4f}")
    print(f"  MRR: {std_results['mrr']:.4f}")
    
    # Evaluate hybrid retrieval
    print("\n--- Evaluating hybrid retrieval ---")
    hybrid_results = evaluator.evaluate_retrieval(
        queries=test_queries,
        ground_truth=ground_truth,
        retrieval_fn=hybrid_retrieval,
        k_values=[1, 2, 3],
        run_name="hybrid_retrieval"
    )
    
    # Print metrics
    print("\nHybrid Retrieval Metrics:")
    for k in [1, 2, 3]:
        print(f"  Precision@{k}: {hybrid_results['precision'][k]:.4f}")
        print(f"  Recall@{k}: {hybrid_results['recall'][k]:.4f}")
        print(f"  F1@{k}: {hybrid_results['f1'][k]:.4f}")
    print(f"  MAP: {hybrid_results['map']:.4f}")
    print(f"  MRR: {hybrid_results['mrr']:.4f}")
    
    # Compare runs
    print("\n--- Comparing retrieval methods ---")
    comparison = evaluator.compare_runs(
        run_names=["standard_retrieval", "hybrid_retrieval"]
    )
    
    print("\nComparison:")
    for run_name, metrics in comparison.items():
        print(f"\n{run_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    print("\n--- Saving evaluation results ---")
    evaluator.save_results("standard_retrieval")
    evaluator.save_results("hybrid_retrieval")
    
    print("\nEvaluation test completed successfully!")
    
except Exception as e:
    print(f"Error during test: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up temporary directories
    import shutil
    try:
        print("Cleaning up temporary directories...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(eval_dir, ignore_errors=True)
        print(f"Cleaned up temporary directories")
    except Exception as e:
        print(f"Failed to clean up temporary directories")
        print(f"Error: {str(e)}") 