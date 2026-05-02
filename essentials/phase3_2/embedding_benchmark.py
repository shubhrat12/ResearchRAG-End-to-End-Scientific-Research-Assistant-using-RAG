"""
Embedding Benchmark

This script benchmarks different embedding models on scientific text to evaluate
their performance in similarity tasks and retrieval quality.
"""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models
try:
    from essentials.phase3_1.models import Chunk
    from essentials.phase3_2.basic_embeddings import BasicEmbedding
    from essentials.phase3_2.advanced_embeddings import AdvancedEmbedding, EmbeddingEvaluator
    
    # Try to import ScientificEmbedding, but don't fail if it's not available
    try:
        from essentials.phase3_2.scientific_embeddings import ScientificEmbedding
        has_scientific_embeddings = True
    except ImportError:
        logger.warning("ScientificEmbedding not available - skipping this model")
        has_scientific_embeddings = False
        
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    raise

def load_test_data(filepath: str = "data/sample_chunks.json") -> Tuple[List[Chunk], List[int]]:
    """Load test data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Tuple of (list of chunks, list of cluster labels)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data file not found: {filepath}")
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        chunks = []
        # Assign clusters based on paper_id
        paper_ids = {}
        cluster_labels = []
        
        for item in data:
            chunks.append(Chunk(
                id=item['id'],
                text=item['text'],
                source=item['source'],
                metadata=item['metadata']
            ))
            
            # Assign cluster based on paper_id
            paper_id = item.get('metadata', {}).get('paper_id', 'unknown')
            if paper_id not in paper_ids:
                paper_ids[paper_id] = len(paper_ids)
            
            cluster_labels.append(paper_ids[paper_id])
            
        logger.info(f"Loaded {len(chunks)} chunks with {len(paper_ids)} clusters")
        return chunks, cluster_labels
    
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def benchmark_embedding_models(chunks: List[Chunk], 
                              cluster_labels: List[int],
                              models: Dict[str, Any],
                              output_dir: str = "data/benchmark_results"):
    """Benchmark embedding models on scientific text.
    
    Args:
        chunks: List of test chunks
        cluster_labels: List of cluster labels for evaluation
        models: Dictionary mapping model names to model instances
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Benchmarking model: {model_name}")
        
        # Measure embedding time
        start_time = time.time()
        embedded_docs = model.embed_chunks(chunks)
        embedding_time = time.time() - start_time
        
        embeddings = [doc["embedding"] for doc in embedded_docs]
        
        # Evaluate embedding quality
        evaluator = EmbeddingEvaluator()
        metrics = evaluator.evaluate_similarity(embeddings, cluster_labels)
        
        # Generate evaluation report
        report = evaluator.generate_report(
            model_name, 
            metrics,
            output_file=os.path.join(output_dir, f"{model_name}_report.txt")
        )
        
        # Store results
        results[model_name] = {
            "embedding_time": embedding_time,
            "embedding_dimension": len(embeddings[0]),
            "metrics": metrics,
        }
        
        logger.info(f"Results for {model_name}:")
        logger.info(f"  Embedding time: {embedding_time:.2f} seconds")
        logger.info(f"  Intra-cluster similarity: {metrics['avg_intra_cluster_similarity']:.4f}")
        logger.info(f"  Inter-cluster similarity: {metrics['avg_inter_cluster_similarity']:.4f}")
        logger.info(f"  Cluster contrast: {metrics['cluster_contrast']:.4f}")
    
    # Save overall results
    with open(os.path.join(output_dir, "benchmark_results.json"), 'w') as f:
        # Convert values that aren't JSON serializable
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "embedding_time": model_results["embedding_time"],
                "embedding_dimension": model_results["embedding_dimension"],
                "metrics": {
                    k: float(v) for k, v in model_results["metrics"].items()
                }
            }
        json.dump(serializable_results, f, indent=2)
    
    # Generate visualization
    plot_benchmark_results(results, output_dir)
    
    return results

def plot_benchmark_results(results: Dict[str, Dict], output_dir: str):
    """Create visualizations of benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        output_dir: Directory to save visualizations
    """
    try:
        # Prepare data for plotting
        model_names = list(results.keys())
        contrasts = [results[name]["metrics"]["cluster_contrast"] for name in model_names]
        times = [results[name]["embedding_time"] for name in model_names]
        dimensions = [results[name]["embedding_dimension"] for name in model_names]
        
        # Plot cluster contrast (higher is better)
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, contrasts)
        plt.title("Cluster Contrast by Model (Higher is Better)")
        plt.ylabel("Cluster Contrast")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cluster_contrast.png"))
        
        # Plot embedding time (lower is better)
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, times)
        plt.title("Embedding Time by Model (Lower is Better)")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "embedding_time.png"))
        
        # Plot dimension vs. quality
        plt.figure(figsize=(10, 6))
        plt.scatter(dimensions, contrasts, s=100)
        for i, name in enumerate(model_names):
            plt.annotate(name, (dimensions[i], contrasts[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        plt.title("Embedding Dimension vs. Quality")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Cluster Contrast")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dimension_vs_quality.png"))
        
        logger.info(f"Saved visualizations to {output_dir}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--data", type=str, default="data/sample_chunks.json", 
                       help="Path to test data JSON file")
    parser.add_argument("--output-dir", type=str, default="data/benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--enable-cache", action="store_true", 
                       help="Enable embedding cache for advanced model")
    parser.add_argument("--enable-compression", action="store_true",
                       help="Enable embedding compression for advanced model")
    parser.add_argument("--target-dimensions", type=int, default=100,
                       help="Target dimensions for compression")
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    chunks, cluster_labels = load_test_data(args.data)
    
    # Initialize models
    models = {
        "basic_small": BasicEmbedding(model_name="en_core_web_sm"),
        "basic_medium": BasicEmbedding(model_name="en_core_web_md"),
        "advanced": AdvancedEmbedding(
            model_name="en_core_web_md",
            use_cache=args.enable_cache,
            enable_compression=args.enable_compression,
            target_dimensions=args.target_dimensions
        )
    }
    
    # Add scientific model if available
    if has_scientific_embeddings:
        try:
            models["scientific"] = ScientificEmbedding(model_name="general")
        except Exception as e:
            logger.warning(f"Error initializing ScientificEmbedding: {str(e)}")
    
    # Try to add a scientific spaCy model if available
    try:
        import spacy
        # Check if scispacy model is installed
        try:
            spacy.load("en_core_sci_sm")
            models["scispacy_small"] = AdvancedEmbedding(model_name="en_core_sci_sm")
            logger.info("Added scispacy small model")
        except OSError:
            logger.info("scispacy model not available")
    except ImportError:
        logger.info("spaCy or scispacy not installed")
    
    # Run benchmark
    logger.info(f"Benchmarking {len(models)} models")
    results = benchmark_embedding_models(
        chunks, 
        cluster_labels, 
        models,
        output_dir=args.output_dir
    )
    
    logger.info("Benchmark completed")
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    for model_name, model_results in results.items():
        logger.info(f"Model: {model_name}")
        logger.info(f"  Cluster contrast: {model_results['metrics']['cluster_contrast']:.4f}")
        logger.info(f"  Embedding time: {model_results['embedding_time']:.2f} seconds")
        logger.info(f"  Dimension: {model_results['embedding_dimension']}")
        logger.info("")

if __name__ == "__main__":
    main() 