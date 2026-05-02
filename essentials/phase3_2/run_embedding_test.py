"""
Test script for Phase 3.2 embedding functionality
"""

import os
import json
import logging
import time
from typing import List

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
from essentials.phase3_2.basic_embeddings import BasicEmbedding
from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding

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

def test_basic_embedding():
    """Test BasicEmbedding functionality."""
    print("\n=== Testing BasicEmbedding ===")
    
    # Create embedding model
    model = BasicEmbedding()
    print(f"BasicEmbedding dimension: {model.dimension}")
    
    # Test text embedding
    sample_text = "This is a sample text for testing basic embeddings"
    embedding = model.embed_text(sample_text)
    print(f"Embedded text dimension: {len(embedding)}")
    print(f"First few values: {embedding[:3]}")
    
    return model

def test_advanced_embedding(use_cache=True, enable_compression=False):
    """Test AdvancedEmbedding functionality."""
    print(f"\n=== Testing AdvancedEmbedding (cache={use_cache}, compression={enable_compression}) ===")
    
    # Create embedding model
    model = AdvancedEmbedding(
        model_name="en_core_web_md",
        use_cache=use_cache,
        enable_compression=enable_compression,
        target_dimensions=100 if enable_compression else 300
    )
    print(f"AdvancedEmbedding dimension: {model.dimension}")
    
    # Test text embedding
    sample_text = "This is a sample text for testing advanced embeddings with scientific terminology like quantum entanglement and neural networks"
    
    # First embedding (measure time)
    start_time = time.time()
    embedding = model.embed_text(sample_text)
    first_time = time.time() - start_time
    print(f"First embedding time: {first_time:.4f}s")
    print(f"Dimension: {len(embedding)}")
    print(f"First few values: {embedding[:3]}")
    
    # Second embedding (test caching)
    if use_cache:
        start_time = time.time()
        embedding2 = model.embed_text(sample_text)
        second_time = time.time() - start_time
        print(f"Second embedding time (should be faster with cache): {second_time:.4f}s")
        print(f"Cache speedup: {first_time/max(second_time, 0.000001):.2f}x")
    
    return model

def main():
    print("\n*** PHASE 3.2 EMBEDDING TEST ***\n")
    
    # Test BasicEmbedding
    basic_model = test_basic_embedding()
    
    # Test AdvancedEmbedding with different configurations
    # 1. Standard with cache
    advanced_model = test_advanced_embedding(use_cache=True, enable_compression=False)
    
    # 2. With compression enabled
    compressed_model = test_advanced_embedding(use_cache=True, enable_compression=True)
    
    # Test embedding similarity
    print("\n=== Testing Embedding Similarity ===")
    
    # Create two related and one unrelated text sample
    text1 = "Quantum physics describes the behavior of matter at the atomic scale."
    text2 = "Quantum mechanics revolutionized our understanding of physics."
    text3 = "Machine learning algorithms learn patterns from data."
    
    # Get embeddings using advanced model
    emb1 = advanced_model.embed_text(text1)
    emb2 = advanced_model.embed_text(text2)
    emb3 = advanced_model.embed_text(text3)
    
    # Calculate similarities
    sim12 = advanced_model.similarity(emb1, emb2)
    sim13 = advanced_model.similarity(emb1, emb3)
    sim23 = advanced_model.similarity(emb2, emb3)
    
    print(f"Similarity between related texts: {sim12:.4f}")
    print(f"Similarity between unrelated texts 1-3: {sim13:.4f}")
    print(f"Similarity between unrelated texts 2-3: {sim23:.4f}")
    
    print("\nEmbedding tests completed successfully!")

if __name__ == "__main__":
    main() 