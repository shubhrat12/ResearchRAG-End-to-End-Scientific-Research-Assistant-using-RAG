"""
Complete test script for Phase 3.2 - Embedding and Retrieval

This script tests all major components of Phase 3.2:
1. Basic, Advanced and Scientific embeddings
2. Vector store operations
3. Different retrieval strategies
4. Optimizations: caching and compression
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variable to handle OpenMP duplicate lib issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    # Import models and components
    from essentials.phase3_1.models import Chunk, Section
    from essentials.phase3_2.vector_store import VectorStore
    from essentials.phase3_2.retrieval import Retriever
    
    # Try to import embedding models with fallbacks
    try:
        from essentials.phase3_2.scientific_embeddings import ScientificEmbedding
        scientific_available = True
        logger.info("ScientificEmbedding is available")
    except ImportError as e:
        scientific_available = False
        logger.warning(f"ScientificEmbedding not available: {str(e)}")
    
    from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding
    from essentials.phase3_2.basic_embeddings import BasicEmbedding
    
    logger.info("Successfully imported all required modules")
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    raise

def load_sample_data(filepath: str = "data/sample_chunks.json") -> List[Chunk]:
    """Load sample data from JSON file."""
    logger.info(f"Loading data from {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        sample_data = generate_sample_data()
        save_sample_data(sample_data, filepath)
        return sample_data
    
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

def generate_sample_data() -> List[Chunk]:
    """Generate sample data if none exists."""
    logger.info("Generating sample scientific text chunks")
    
    sample_texts = [
        "Abstract: Deep learning has revolutionized machine learning by enabling the automatic extraction of hierarchical features from raw data. This paper explores recent advancements in transformer architectures for scientific text analysis.",
        "Introduction: The concept of quantum entanglement was first proposed by Einstein, Podolsky, and Rosen in their famous 1935 paper. It describes a phenomenon where quantum states of particles are connected regardless of distance.",
        "Methods: We employed a mixed-methods approach combining both qualitative and quantitative data analysis. Statistical significance was determined using Student's t-test with p < 0.05 considered significant.",
        "Results: The model achieved 95% accuracy on the test dataset, significantly outperforming previous approaches (p < 0.001). Table 1 summarizes the comparative performance metrics across all evaluated models.",
        "Discussion: The observed quantum advantage in our algorithm scales quadratically with the problem size, consistent with theoretical predictions by Bennett et al. (2019).",
        "Conclusion: Quantum entanglement remains one of the most powerful resources in quantum information science, enabling capabilities impossible in classical systems.",
        "References: 1. Smith J, et al. (2020) Advances in quantum algorithms. Nature Physics 16(2), 113-119. 2. Johnson M, et al. (2019) Quantum supremacy using a programmable superconducting processor. Nature 574, 505-510.",
        "Appendix A: The mathematical formulation of entanglement entropy is given by S = -Tr(ρ log ρ), where ρ is the density matrix of the quantum system under consideration.",
        "Figure 1: Schematic representation of the proposed quantum circuit implementation, showing the arrangement of Hadamard and CNOT gates required for entanglement generation.",
        "Table 1: Performance comparison between classical and quantum approaches, showing runtime (ms) and memory requirements (MB) for problem sizes ranging from n=10 to n=1000."
    ]
    
    chunks = []
    for i, text in enumerate(sample_texts):
        chunk_id = f"chunk{i+1}"
        source = f"sample_paper_{i//2 + 1}.pdf"
        
        # Create varied metadata
        metadata = {
            "paper_id": f"paper{i//2 + 1}",
            "section": text.split(":")[0],
            "page": i + 1,
        }
        
        # Add citation counts to some documents
        if i % 3 == 0:
            metadata["citation_count"] = 120 - (i * 10)  # Higher citations for earlier chunks
        
        chunks.append(Chunk(
            id=chunk_id,
            text=text,
            source=source,
            metadata=metadata
        ))
    
    logger.info(f"Generated {len(chunks)} sample chunks")
    return chunks

def save_sample_data(chunks: List[Chunk], filepath: str):
    """Save sample data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = []
    for chunk in chunks:
        data.append({
            "id": chunk.id,
            "text": chunk.text,
            "source": chunk.source,
            "metadata": chunk.metadata
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to {filepath}")

def test_basic_embedding(chunks: List[Chunk]):
    """Test basic embedding functionality."""
    logger.info("=== Testing BasicEmbedding ===")
    
    model = BasicEmbedding()
    logger.info(f"Created BasicEmbedding with dimension: {model.dimension}")
    
    # Test single text embedding
    start_time = time.time()
    sample_text = "This is a sample text for testing embeddings"
    embedding = model.embed_text(sample_text)
    logger.info(f"Embedded single text in {time.time() - start_time:.4f}s, dimension={len(embedding)}")
    
    # Test chunk embedding
    start_time = time.time()
    embedded_docs = model.embed_chunks(chunks[:2])
    logger.info(f"Embedded {len(embedded_docs)} chunks in {time.time() - start_time:.4f}s")
    
    return model, embedded_docs

def test_advanced_embedding(chunks: List[Chunk], use_cache: bool = True, enable_compression: bool = False):
    """Test advanced embedding functionality."""
    logger.info("=== Testing AdvancedEmbedding ===")
    logger.info(f"Parameters: use_cache={use_cache}, enable_compression={enable_compression}")
    
    # Create with caching and optional compression
    model = AdvancedEmbedding(
        model_name="en_core_web_md",
        use_cache=use_cache,
        enable_compression=enable_compression,
        target_dimensions=100 if enable_compression else 300
    )
    
    logger.info(f"Created AdvancedEmbedding with dimension: {model.dimension}")
    
    # Test single text embedding (first run)
    query = "quantum entanglement applications"
    logger.info(f"Embedding query: '{query}'")
    
    start_time = time.time()
    embedding = model.embed_text(query)
    first_time = time.time() - start_time
    logger.info(f"First embedding took {first_time:.4f}s, dimension={len(embedding)}")
    
    # Test caching with second run
    if use_cache:
        start_time = time.time()
        embedding2 = model.embed_text(query)
        second_time = time.time() - start_time
        logger.info(f"Second embedding (cached) took {second_time:.4f}s")
        
        speedup = first_time / max(second_time, 0.0001)
        logger.info(f"Cache speedup: {speedup:.2f}x")
    
    # Test chunk embedding
    start_time = time.time()
    embedded_docs = model.embed_chunks(chunks[:3])
    logger.info(f"Embedded {len(embedded_docs)} chunks in {time.time() - start_time:.4f}s")
    
    # Test weighted section embedding
    if len(chunks) >= 2:
        section = Section(
            id="section1",
            title=chunks[0].text.split(":")[1].strip(),
            content=chunks[1].text.split(":")[1].strip()
        )
        
        weighted_embedding = model.embed_section_weighted(
            section, 
            title_weight=2.0,
            first_para_weight=1.5
        )
        
        logger.info(f"Created weighted section embedding with dimension: {len(weighted_embedding)}")
    
    return model, embedded_docs

def test_scientific_embedding(chunks: List[Chunk]):
    """Test scientific embedding if available."""
    if not scientific_available:
        logger.warning("Skipping ScientificEmbedding test - not available")
        return None, []
    
    logger.info("=== Testing ScientificEmbedding ===")
    
    try:
        model = ScientificEmbedding()
        logger.info(f"Created ScientificEmbedding with dimension: {model.dimension}")
        
        # Test single text embedding
        start_time = time.time()
        sample_text = "The quantum entanglement phenomenon demonstrates non-local correlations."
        embedding = model.embed_text(sample_text)
        logger.info(f"Embedded scientific text in {time.time() - start_time:.4f}s, dimension={len(embedding)}")
        
        # Test chunk embedding
        start_time = time.time()
        embedded_docs = model.embed_chunks(chunks[:2])
        logger.info(f"Embedded {len(embedded_docs)} chunks in {time.time() - start_time:.4f}s")
        
        return model, embedded_docs
        
    except Exception as e:
        logger.error(f"Error testing ScientificEmbedding: {str(e)}")
        return None, []

def test_vector_store(embedded_docs, dimension, similarity_type="cosine"):
    """Test vector store functionality."""
    logger.info(f"=== Testing VectorStore with {similarity_type} similarity ===")
    
    # Create vector store
    vector_store = VectorStore(dimension=dimension, index_type=similarity_type)
    logger.info(f"Created vector store with dimension {dimension}")
    
    # Add documents
    vector_store.add_documents(embedded_docs)
    logger.info(f"Added {len(embedded_docs)} documents to vector store")
    
    # Save and reload 
    store_dir = "data/test_vector_store"
    os.makedirs(store_dir, exist_ok=True)
    
    vector_store.save(store_dir)
    logger.info(f"Saved vector store to {store_dir}")
    
    loaded_store = VectorStore.load(store_dir)
    logger.info(f"Loaded vector store with {len(loaded_store.documents)} documents")
    
    return loaded_store

def test_retrieval_strategies(chunks, embedding_model, vector_store):
    """Test different retrieval strategies."""
    logger.info("=== Testing Retrieval Strategies ===")
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model
    )
    
    # Basic retrieval
    query = "quantum entanglement applications"
    logger.info(f"Running basic retrieval for query: '{query}'")
    
    try:
        start_time = time.time()
        results = retriever.retrieve(query, k=3)
        logger.info(f"Basic retrieval completed in {time.time() - start_time:.4f}s")
        
        print("\n--- BASIC RETRIEVAL RESULTS ---")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.4f}, ID: {result['id']}")
            print(f"   Source: {result['metadata'].get('source', 'unknown')}")
            
        # Hybrid retrieval
        logger.info(f"Running hybrid retrieval for query: '{query}'")
        start_time = time.time()
        hybrid_results = retriever.hybrid_retrieve(query, k=3)
        logger.info(f"Hybrid retrieval completed in {time.time() - start_time:.4f}s")
        
        print("\n--- HYBRID RETRIEVAL RESULTS ---")
        for i, result in enumerate(hybrid_results):
            print(f"{i+1}. Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f}")
            print(f"   ID: {result['id']}")
            
        # Citation-based reranking
        logger.info(f"Running citation-based reranking for query: '{query}'")
        start_time = time.time()
        reranked_results = retriever.retrieve_with_reranking(query, k=3)
        logger.info(f"Citation-based reranking completed in {time.time() - start_time:.4f}s")
        
        print("\n--- CITATION-BASED RERANKING RESULTS ---")
        for i, result in enumerate(reranked_results):
            print(f"{i+1}. Combined: {result['combined_score']:.4f}, Semantic: {result['semantic_score']:.4f}, Citation: {result['citation_score']:.4f}")
            print(f"   ID: {result['id']}")
            
        return True
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_evaluation(embedding_model):
    """Test embedding evaluation functionality."""
    logger.info("=== Testing Embedding Evaluation ===")
    
    try:
        # Generate sample embeddings with known clusters
        texts = [
            "Quantum physics describes the behavior of matter at the atomic scale.",
            "Quantum mechanics revolutionized our understanding of physics.",
            "Quantum entanglement is a phenomenon with no classical analogue.",
            "Machine learning algorithms learn patterns from data.",
            "Deep learning has revolutionized computer vision and NLP.",
            "Neural networks are the foundation of modern AI systems."
        ]
        
        # Clusters: 0=quantum, 1=ML/AI
        labels = [0, 0, 0, 1, 1, 1]
        
        # Embed texts
        embeddings = [embedding_model.embed_text(text) for text in texts]
        
        # Get evaluator and run evaluation
        evaluator = embedding_model.get_evaluator()
        metrics = evaluator.evaluate_similarity(embeddings, labels)
        
        # Generate report
        report = evaluator.generate_report(
            model_name=embedding_model.__class__.__name__,
            metrics=metrics
        )
        
        print("\n--- EMBEDDING EVALUATION REPORT ---")
        print(report)
        
        return metrics
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return {}

def main():
    """Main test function."""
    print("\n\n========== COMPLETE TEST OF PHASE 3.2 ==========\n")
    print("Testing embedding and retrieval components...")
    
    # Load or generate sample data
    chunks = load_sample_data()
    if not chunks:
        logger.error("Failed to load or generate sample data")
        return
    
    # 1. Test embedding models
    print("\n\n----- TESTING EMBEDDING MODELS -----\n")
    
    # Basic embedding
    basic_model, basic_docs = test_basic_embedding(chunks)
    
    # Advanced embedding with different configurations
    advanced_model_standard, advanced_docs = test_advanced_embedding(
        chunks, use_cache=True, enable_compression=False
    )
    
    advanced_model_compressed, compressed_docs = test_advanced_embedding(
        chunks, use_cache=True, enable_compression=True
    )
    
    # Scientific embedding (if available)
    scientific_model, scientific_docs = test_scientific_embedding(chunks)
    
    # 2. Test vector store with different similarity types
    print("\n\n----- TESTING VECTOR STORE -----\n")
    
    vector_store_cosine = test_vector_store(
        advanced_docs, 
        dimension=advanced_model_standard.dimension,
        similarity_type="cosine"
    )
    
    vector_store_l2 = test_vector_store(
        advanced_docs, 
        dimension=advanced_model_standard.dimension,
        similarity_type="L2"
    )
    
    # 3. Test retrieval strategies
    print("\n\n----- TESTING RETRIEVAL STRATEGIES -----\n")
    retrieval_success = test_retrieval_strategies(
        chunks,
        advanced_model_standard, 
        vector_store_cosine
    )
    
    # 4. Test dimension adaptation
    if advanced_model_compressed:
        print("\n\n----- TESTING DIMENSION ADAPTATION -----\n")
        logger.info("Testing retrieval with compressed embeddings and dimension adaptation")
        
        try:
            adapter_success = test_retrieval_strategies(
                chunks,
                advanced_model_compressed,
                vector_store_cosine
            )
            if adapter_success:
                logger.info("Successfully adapted dimensions during retrieval")
        except Exception as e:
            logger.error(f"Error during dimension adaptation test: {str(e)}")
    
    # 5. Test embedding evaluation
    print("\n\n----- TESTING EMBEDDING EVALUATION -----\n")
    eval_metrics = test_evaluation(advanced_model_standard)
    
    # Summary
    print("\n\n========== TEST SUMMARY ==========\n")
    print(f"Basic Embedding: {'✓' if basic_docs else '✗'}")
    print(f"Advanced Embedding: {'✓' if advanced_docs else '✗'}")
    print(f"Advanced Embedding (Compressed): {'✓' if compressed_docs else '✗'}")
    print(f"Scientific Embedding: {'✓' if scientific_model and scientific_docs else 'Not Available'}")
    print(f"Vector Store: {'✓' if vector_store_cosine and vector_store_l2 else '✗'}")
    print(f"Retrieval Strategies: {'✓' if retrieval_success else '✗'}")
    print(f"Embedding Evaluation: {'✓' if eval_metrics else '✗'}")
    
    print("\nPhase 3.2 testing complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled error in main: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nTest failed with error: {str(e)}")
        sys.exit(1) 