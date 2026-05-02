"""
Advanced Scientific Retriever Demo

This script demonstrates the scientific retrieval capabilities of the RAG system
using advanced embeddings with caching and compression.
"""

import os
import json
import argparse
import sys
import time
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting advanced scientific retriever demo...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

try:
    from essentials.phase3_1.models import Chunk
    logger.info("Successfully imported Chunk")
    from essentials.phase3_2.vector_store import VectorStore
    logger.info("Successfully imported VectorStore")
    from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding
    logger.info("Successfully imported AdvancedEmbedding")
    from essentials.phase3_2.retrieval import Retriever
    logger.info("Successfully imported Retriever")
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    raise

def load_sample_data(filepath: str) -> List[Chunk]:
    """Load sample data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Loading sample data from {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Sample data file not found: {filepath}")
        raise FileNotFoundError(f"Sample data file not found: {filepath}")
        
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
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        raise

def create_vector_store(chunks: List[Chunk], model: AdvancedEmbedding, output_dir: str) -> VectorStore:
    """Create a vector store from chunks.
    
    Args:
        chunks: List of chunks
        model: Embedding model
        output_dir: Directory to save the vector store
        
    Returns:
        Vector store
    """
    # Embed chunks
    logger.info("Embedding chunks...")
    try:
        # Measure time for embedding
        start_time = time.time()
        embedded_docs = model.embed_chunks(chunks)
        embedding_time = time.time() - start_time
        logger.info(f"Embedding completed in {embedding_time:.2f} seconds")
        
        # Create vector store
        logger.info(f"Creating vector store with dimension {model.dimension}")
        vector_store = VectorStore(dimension=model.dimension, index_type="cosine")
        
        # Add documents
        vector_store.add_documents(embedded_docs)
        
        # Save vector store
        os.makedirs(output_dir, exist_ok=True)
        vector_store.save(output_dir)
        
        logger.info(f"Vector store created and saved to {output_dir}")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def run_retrieval_demo(query: str, vector_store: VectorStore, model: AdvancedEmbedding, use_cache: bool, verbose: bool = False):
    """Run retrieval demo.
    
    Args:
        query: Query string
        vector_store: Vector store
        model: Embedding model
        use_cache: Whether to use caching for embeddings
        verbose: Whether to display full text of results
    """
    try:
        # Load sample data to get the original text
        sample_data = {}
        if os.path.exists("data/sample_chunks.json"):
            with open("data/sample_chunks.json", "r") as f:
                data = json.load(f)
                for item in data:
                    sample_data[item["id"]] = item["text"]
        
        # Create retriever
        logger.info("Initializing retriever...")
        retriever = Retriever(
            vector_store=vector_store, 
            embedding_model=model,
            use_cache=use_cache
        )
        
        print(f"\n\n{'=' * 50}")
        print(f"QUERY: {query}")
        print(f"{'=' * 50}\n")
        
        # Basic retrieval
        logger.info("Running basic retrieval...")
        
        try:
            start_time = time.time()
            # Debug: first check the query embedding
            debug_query_embedding = model.embed_text(query)
            logger.info(f"Query embedding generated successfully: dimension={len(debug_query_embedding)}")
            
            # Run retrieval
            results = retriever.retrieve(query, k=3)
            query_time = time.time() - start_time
            logger.info(f"Basic retrieval completed in {query_time:.4f} seconds")
            
            print(f"\n--- BASIC RETRIEVAL (Time: {query_time:.4f}s) ---")
            for i, result in enumerate(results):
                doc_id = result["id"]
                original_text = sample_data.get(doc_id, "Text not found")
                print(f"{i+1}. [Score: {result['score']:.4f}] {original_text[:100]}...")
                if verbose:
                    print(f"   ID: {doc_id}, Source: {result['metadata'].get('source', 'unknown')}")
                    print(f"   Paper ID: {result['metadata'].get('paper_id', 'unknown')}")
                    print(f"   Citation Count: {result['metadata'].get('citation_count', 'unknown')}")
                    print(f"   Full Text: {original_text}")
                    print()
        except Exception as e:
            logger.error(f"Error during basic retrieval: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"Basic retrieval error: {str(e)}")
            return
        
        # Run query again to test caching effect
        if use_cache:
            logger.info("Running basic retrieval again (with cache)...")
            try:
                start_time = time.time()
                cached_results = retriever.retrieve(query, k=3)
                cached_query_time = time.time() - start_time
                cache_speedup = (query_time / cached_query_time) if cached_query_time > 0 else float('inf')
                logger.info(f"Cached retrieval completed in {cached_query_time:.4f} seconds (speedup: {cache_speedup:.2f}x)")
                print(f"\n--- CACHED RETRIEVAL (Time: {cached_query_time:.4f}s, Speedup: {cache_speedup:.2f}x) ---")
                for i, result in enumerate(cached_results):
                    doc_id = result["id"]
                    original_text = sample_data.get(doc_id, "Text not found")
                    print(f"{i+1}. [Score: {result['score']:.4f}] {original_text[:100]}...")
            except Exception as e:
                logger.error(f"Error during cached retrieval: {str(e)}")
                print(f"Cached retrieval error: {str(e)}")
        
        # Hybrid retrieval
        logger.info("Running hybrid retrieval...")
        try:
            start_time = time.time()
            hybrid_results = retriever.hybrid_retrieve(query, k=3)
            hybrid_time = time.time() - start_time
            logger.info(f"Hybrid retrieval completed in {hybrid_time:.4f} seconds")
            
            print(f"\n\n--- HYBRID RETRIEVAL (Time: {hybrid_time:.4f}s) ---")
            for i, result in enumerate(hybrid_results):
                doc_id = result["id"]
                original_text = sample_data.get(doc_id, "Text not found")
                print(f"{i+1}. [Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f}]")
                print(f"   {original_text[:100]}...")
                if verbose:
                    print(f"   ID: {doc_id}, Source: {result['metadata'].get('source', 'unknown')}")
                    print(f"   Paper ID: {result['metadata'].get('paper_id', 'unknown')}")
                    print(f"   Citation Count: {result['metadata'].get('citation_count', 'unknown')}")
                    print(f"   Full Text: {original_text}")
                    print()
        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {str(e)}")
            print(f"Hybrid retrieval error: {str(e)}")
        
        # Citation-based reranking
        logger.info("Running citation-based reranking...")
        try:
            start_time = time.time()
            reranked_results = retriever.retrieve_with_reranking(query, k=3)
            reranking_time = time.time() - start_time
            logger.info(f"Citation-based reranking completed in {reranking_time:.4f} seconds")
            
            print(f"\n\n--- CITATION-BASED RERANKING (Time: {reranking_time:.4f}s) ---")
            for i, result in enumerate(reranked_results):
                doc_id = result["id"]
                original_text = sample_data.get(doc_id, "Text not found")
                print(f"{i+1}. [Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Citation: {result['citation_score']:.4f}]")
                print(f"   {original_text[:100]}...")
                if verbose:
                    print(f"   ID: {doc_id}, Source: {result['metadata'].get('source', 'unknown')}")
                    print(f"   Paper ID: {result['metadata'].get('paper_id', 'unknown')}")
                    print(f"   Citation Count: {result['metadata'].get('citation_count', 'unknown')}")
                    print(f"   Full Text: {original_text}")
                    print()
                    
            # Summary
            print(f"\n\n--- PERFORMANCE SUMMARY ---")
            print(f"Basic retrieval:       {query_time:.4f}s")
            if use_cache:
                print(f"Cached retrieval:      {cached_query_time:.4f}s (Speedup: {cache_speedup:.2f}x)")
            print(f"Hybrid retrieval:      {hybrid_time:.4f}s")
            print(f"Citation-based:        {reranking_time:.4f}s")
        except Exception as e:
            logger.error(f"Error during citation-based reranking: {str(e)}")
            print(f"Citation-based reranking error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during retrieval demo: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nError during retrieval: {str(e)}")

def main():
    """Main function."""
    try:
        logger.info("Parsing arguments...")
        parser = argparse.ArgumentParser(description="Advanced Scientific Retriever Demo")
        parser.add_argument("--data", type=str, default="data/sample_chunks.json", help="Path to sample data JSON file")
        parser.add_argument("--model", type=str, default="en_core_web_md", help="spaCy model to use")
        parser.add_argument("--vector-store", type=str, default="data/vector_store_advanced", help="Path to vector store directory")
        parser.add_argument("--query", type=str, default="quantum entanglement applications", help="Query string")
        parser.add_argument("--use-cache", action="store_true", help="Use embedding cache")
        parser.add_argument("--enable-compression", action="store_true", help="Enable embedding compression")
        parser.add_argument("--target-dimensions", type=int, default=100, help="Target dimensions for compression")
        parser.add_argument("--verbose", action="store_true", help="Display verbose output with full text")
        args = parser.parse_args()
        
        logger.info(f"Arguments: data={args.data}, model={args.model}, vector_store={args.vector_store}")
        logger.info(f"Query: {args.query}, use_cache: {args.use_cache}, compression: {args.enable_compression}, dimensions: {args.target_dimensions}")
        
        # Create embedding model
        logger.info(f"Creating advanced embedding model: {args.model}")
        model = AdvancedEmbedding(
            model_name=args.model,
            use_cache=args.use_cache,
            enable_compression=args.enable_compression,
            target_dimensions=args.target_dimensions
        )
        
        # Check if vector store exists
        if os.path.exists(args.vector_store) and os.path.isdir(args.vector_store) and \
           os.path.exists(os.path.join(args.vector_store, "index.faiss")):
            logger.info(f"Loading existing vector store from {args.vector_store}")
            try:
                vector_store = VectorStore.load(args.vector_store)
                logger.info(f"Loaded vector store with {len(vector_store.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                logger.info("Creating new vector store instead")
                # Load sample data
                chunks = load_sample_data(args.data)
                # Create vector store
                vector_store = create_vector_store(chunks, model, args.vector_store)
        else:
            logger.info(f"Creating new vector store using data from {args.data}")
            # Load sample data
            chunks = load_sample_data(args.data)
            # Create vector store
            vector_store = create_vector_store(chunks, model, args.vector_store)
        
        # Run retrieval demo
        run_retrieval_demo(args.query, vector_store, model, args.use_cache, args.verbose)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Script started")
    try:
        main()
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        print(f"Unhandled error: {str(e)}")
        import traceback
        traceback.print_exc() 