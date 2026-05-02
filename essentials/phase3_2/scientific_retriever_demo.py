"""
Scientific Retriever Demo

This script demonstrates the scientific retrieval capabilities of the RAG system.
"""

import os
import json
import argparse
import sys
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting scientific retriever demo...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

try:
    from essentials.phase3_1.models import Chunk
    logger.info("Successfully imported Chunk")
    from essentials.phase3_2.vector_store import VectorStore
    logger.info("Successfully imported VectorStore")
    from essentials.phase3_2.scientific_embeddings import ScientificEmbedding
    logger.info("Successfully imported ScientificEmbedding")
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

def create_vector_store(chunks: List[Chunk], model: ScientificEmbedding, output_dir: str) -> VectorStore:
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
        embedded_docs = model.embed_chunks(chunks)
        
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

def run_retrieval_demo(query: str, vector_store: VectorStore, model: ScientificEmbedding):
    """Run retrieval demo.
    
    Args:
        query: Query string
        vector_store: Vector store
        model: Embedding model
    """
    try:
        # Create retriever
        logger.info("Initializing retriever...")
        retriever = Retriever(vector_store=vector_store, embedding_model=model)
        
        print(f"\n\n{'=' * 50}")
        print(f"QUERY: {query}")
        print(f"{'=' * 50}\n")
        
        # Basic retrieval
        logger.info("Running basic retrieval...")
        print("\n--- BASIC RETRIEVAL ---")
        results = retriever.retrieve(query, k=3)
        for i, result in enumerate(results):
            print(f"{i+1}. [Score: {result['score']:.4f}] {result['metadata'].get('text', '')[:100]}...")
        
        # Hybrid retrieval
        logger.info("Running hybrid retrieval...")
        print("\n\n--- HYBRID RETRIEVAL (Semantic + Keyword) ---")
        hybrid_results = retriever.hybrid_retrieve(query, k=3)
        for i, result in enumerate(hybrid_results):
            print(f"{i+1}. [Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f}]")
            print(f"   {result['metadata'].get('text', '')[:100]}...")
        
        # Citation-based reranking
        logger.info("Running citation-based reranking...")
        print("\n\n--- CITATION-BASED RERANKING ---")
        reranked_results = retriever.retrieve_with_reranking(query, k=3)
        for i, result in enumerate(reranked_results):
            print(f"{i+1}. [Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Citation: {result['citation_score']:.4f}]")
            print(f"   {result['metadata'].get('text', '')[:100]}...")
    except Exception as e:
        logger.error(f"Error during retrieval demo: {str(e)}")
        print(f"\nError during retrieval: {str(e)}")

def main():
    """Main function."""
    try:
        logger.info("Parsing arguments...")
        parser = argparse.ArgumentParser(description="Scientific Retriever Demo")
        parser.add_argument("--data", type=str, default="data/sample_chunks.json", help="Path to sample data JSON file")
        parser.add_argument("--model", type=str, default="general", help="Embedding model to use")
        parser.add_argument("--vector-store", type=str, default="data/vector_store", help="Path to vector store directory")
        parser.add_argument("--query", type=str, default="quantum entanglement applications", help="Query string")
        parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache models")
        args = parser.parse_args()
        
        logger.info(f"Arguments: data={args.data}, model={args.model}, vector_store={args.vector_store}, query={args.query}")
        
        # Create embedding model
        logger.info(f"Loading embedding model: {args.model}")
        model = ScientificEmbedding(model_name=args.model, cache_dir=args.cache_dir)
        
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
        run_retrieval_demo(args.query, vector_store, model)
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