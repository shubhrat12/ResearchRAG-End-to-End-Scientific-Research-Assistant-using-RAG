"""
Simple diagnostic test for embeddings and vector store.
"""

import os
import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from essentials.phase3_1.models import Chunk
    from essentials.phase3_2.vector_store import VectorStore
    from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding
    logger.info("Successfully imported required modules")
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    raise

def main():
    # Load sample data
    data_path = "data/sample_chunks.json"
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} chunks")
    
    # Convert to Chunk objects
    chunks = []
    for item in data:
        chunks.append(Chunk(
            id=item['id'],
            text=item['text'],
            source=item['source'],
            metadata=item['metadata']
        ))
    
    # Create embedding model
    logger.info("Creating embedding model")
    model = AdvancedEmbedding(
        model_name="en_core_web_md",
        use_cache=False,  # Disable cache for diagnostic
        enable_compression=False  # Disable compression for diagnostic
    )
    
    # Test embedding
    logger.info("Testing single embedding")
    start_time = time.time()
    sample_text = "This is a sample text for testing embeddings"
    embedding = model.embed_text(sample_text)
    logger.info(f"Embedded text in {time.time() - start_time:.4f}s, dimension={len(embedding)}")
    
    # Test embedding chunks
    logger.info("Testing chunk embedding")
    start_time = time.time()
    embedded_docs = model.embed_chunks(chunks[:2])  # Just try a couple
    logger.info(f"Embedded {len(embedded_docs)} chunks in {time.time() - start_time:.4f}s")
    
    # Test vector store
    logger.info("Creating vector store")
    vector_store = VectorStore(dimension=model.dimension, index_type="cosine")
    
    # Add documents
    logger.info("Adding documents to vector store")
    vector_store.add_documents(embedded_docs)
    logger.info(f"Added {len(embedded_docs)} documents")
    
    # Test search
    logger.info("Testing search")
    query = "quantum entanglement"
    query_embedding = model.embed_text(query)
    logger.info(f"Query embedding dimension: {len(query_embedding)}")
    
    results = vector_store.search(query_embedding, k=2)
    logger.info(f"Search returned {len(results)} results")
    
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: ID={result['id']}, Score={result['score']:.4f}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}") 