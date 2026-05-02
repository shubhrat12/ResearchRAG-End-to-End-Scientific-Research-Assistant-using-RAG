"""
Retrieval evaluation module for phase 3.3.

This module provides tools for evaluating the quality of retrieval systems, including:
- Precision, recall, and F1 metrics
- Mean Average Precision (MAP) calculation
- Mean Reciprocal Rank (MRR) measurement
- Support for ground truth comparison
- Visualization options for result analysis
- Iterative feedback-based improvement
"""

from typing import List, Dict, Any, Optional, Union, Callable, Set, Tuple
import logging
import json
import os
import csv
import math
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from essentials.phase3_3.vector_store import ChromaVectorStore
from essentials.phase3_3.retriever import AdvancedRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Evaluation tools for retrieval systems."""
    
    def __init__(
        self,
        retriever: Optional[AdvancedRetriever] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        results_dir: str = "data/evaluation_results"
    ):
        """Initialize the evaluator.
        
        Args:
            retriever: Optional retriever to evaluate
            vector_store: Optional vector store for direct access
            results_dir: Directory to store evaluation results
        """
        self.retriever = retriever
        self.vector_store = vector_store
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results storage
        self.evaluation_runs = {}
        self.feedback_history = defaultdict(list)
    
    def precision_at_k(self, retrieved: List, relevant: List, k: int) -> float:
        """Calculate precision@k.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: k value
            
        Returns:
            Precision@k value (0-1)
        """
        if not retrieved or k <= 0:
            return 0.0
        
        # Convert to sets for efficient intersection
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # Calculate precision@k
        relevant_retrieved = len(retrieved_set.intersection(relevant_set))
        precision = relevant_retrieved / min(k, len(retrieved))
        
        return precision
    
    def recall_at_k(self, retrieved: List, relevant: List, k: int) -> float:
        """Calculate recall@k.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: k value
            
        Returns:
            Recall@k value (0-1)
        """
        if not retrieved or not relevant or k <= 0:
            return 0.0
        
        # Convert to sets for efficient intersection
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # Calculate recall@k
        relevant_retrieved = len(retrieved_set.intersection(relevant_set))
        recall = relevant_retrieved / len(relevant_set)
        
        return recall
    
    def f1_at_k(self, retrieved: List, relevant: List, k: int) -> float:
        """Calculate F1@k.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: k value
            
        Returns:
            F1@k value (0-1)
        """
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def average_precision(self, retrieved: List, relevant: List) -> float:
        """Calculate average precision (AP).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            
        Returns:
            Average precision value (0-1)
        """
        if not retrieved or not relevant:
            return 0.0
        
        # Convert relevant to set for efficient lookup
        relevant_set = set(relevant)
        
        # Calculate average precision
        precision_sum = 0.0
        num_relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved):
            # Check if this document is relevant
            if doc_id in relevant_set:
                num_relevant_retrieved += 1
                precision_at_i = num_relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        if num_relevant_retrieved == 0:
            return 0.0
        
        ap = precision_sum / len(relevant_set)
        
        return ap
    
    def mean_average_precision(self, retrieved_lists: List[List], relevant_lists: List[List]) -> float:
        """Calculate mean average precision (MAP).
        
        Args:
            retrieved_lists: List of retrieved document ID lists, one per query
            relevant_lists: List of relevant document ID lists, one per query
            
        Returns:
            MAP value (0-1)
        """
        if not retrieved_lists or not relevant_lists or len(retrieved_lists) != len(relevant_lists):
            return 0.0
        
        # Calculate average precision for each query
        aps = [self.average_precision(retrieved, relevant) 
               for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
        
        # Calculate mean
        map_value = sum(aps) / len(aps)
        
        return map_value
    
    def mean_reciprocal_rank(self, retrieved_lists: List[List], relevant_lists: List[List]) -> float:
        """Calculate mean reciprocal rank (MRR).
        
        Args:
            retrieved_lists: List of retrieved document ID lists, one per query
            relevant_lists: List of relevant document ID lists, one per query
            
        Returns:
            MRR value (0-1)
        """
        if not retrieved_lists or not relevant_lists or len(retrieved_lists) != len(relevant_lists):
            return 0.0
        
        # Calculate reciprocal rank for each query
        rrs = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            
            # Find the first relevant document position
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    # Reciprocal rank is 1 / (position + 1)
                    rrs.append(1.0 / (i + 1))
                    break
            else:
                # No relevant documents found
                rrs.append(0.0)
        
        # Calculate mean
        mrr_value = sum(rrs) / len(rrs)
        
        return mrr_value
    
    def normalized_discounted_cumulative_gain(
        self, 
        retrieved: List, 
        relevant_scores: Dict[str, float],
        k: int
    ) -> float:
        """Calculate normalized discounted cumulative gain (nDCG@k).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant_scores: Dictionary mapping relevant document IDs to relevance scores
            k: k value
            
        Returns:
            nDCG@k value (0-1)
        """
        if not retrieved or not relevant_scores or k <= 0:
            return 0.0
        
        # Calculate DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_scores:
                # DCG formula: rel_i / log2(i+2)
                relevance = relevant_scores[doc_id]
                position = i + 1  # 1-indexed position
                dcg += relevance / math.log2(position + 1)
        
        # Calculate ideal DCG@k
        ideal_ranking = sorted(relevant_scores.items(), key=lambda x: x[1], reverse=True)
        idcg = 0.0
        for i, (_, relevance) in enumerate(ideal_ranking[:k]):
            position = i + 1  # 1-indexed position
            idcg += relevance / math.log2(position + 1)
        
        # Calculate nDCG@k
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        
        return ndcg
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        retrieval_fn: Callable,
        k_values: List[int] = [1, 3, 5, 10],
        run_name: str = "default_run"
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate retrieval function on a set of queries with ground truth.
        
        Args:
            queries: List of query strings
            ground_truth: List of lists of relevant document IDs
            retrieval_fn: Function that takes a query and returns a list of retrieved documents
            k_values: List of k values to evaluate at
            run_name: Name for this evaluation run
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(queries) != len(ground_truth):
            raise ValueError("Number of queries must match number of ground truth lists")
        
        # Initialize results
        results = {
            "precision": {k: 0.0 for k in k_values},
            "recall": {k: 0.0 for k in k_values},
            "f1": {k: 0.0 for k in k_values},
            "map": 0.0,
            "mrr": 0.0,
            "query_level_results": []
        }
        
        # Lists for MAP and MRR calculation
        retrieved_lists = []
        
        # Evaluate each query
        for i, (query, relevant) in enumerate(zip(queries, ground_truth)):
            try:
                # Execute query
                retrieved_docs = retrieval_fn(query)
                
                # Extract document IDs
                retrieved = [doc["id"] for doc in retrieved_docs]
                
                # Store for MAP and MRR calculation
                retrieved_lists.append(retrieved)
                
                # Calculate metrics for each k
                query_results = {
                    "query": query,
                    "retrieved": retrieved,
                    "relevant": relevant,
                    "metrics": {}
                }
                
                for k in k_values:
                    precision = self.precision_at_k(retrieved, relevant, k)
                    recall = self.recall_at_k(retrieved, relevant, k)
                    f1 = self.f1_at_k(retrieved, relevant, k)
                    
                    # Add to overall results
                    results["precision"][k] += precision
                    results["recall"][k] += recall
                    results["f1"][k] += f1
                    
                    # Add to query-level results
                    query_results["metrics"][f"precision@{k}"] = precision
                    query_results["metrics"][f"recall@{k}"] = recall
                    query_results["metrics"][f"f1@{k}"] = f1
                
                # Calculate AP for this query
                ap = self.average_precision(retrieved, relevant)
                query_results["metrics"]["ap"] = ap
                
                # Add to query-level results
                results["query_level_results"].append(query_results)
                
            except Exception as e:
                logger.error(f"Error evaluating query {i}: {str(e)}")
                # Continue with next query
        
        # Calculate average metrics
        num_queries = len(queries)
        for k in k_values:
            results["precision"][k] /= num_queries
            results["recall"][k] /= num_queries
            results["f1"][k] /= num_queries
        
        # Calculate MAP and MRR
        results["map"] = self.mean_average_precision(retrieved_lists, ground_truth)
        results["mrr"] = self.mean_reciprocal_rank(retrieved_lists, ground_truth)
        
        # Store results
        self.evaluation_runs[run_name] = results
        
        return results
    
    def evaluate_with_relevance_scores(
        self,
        queries: List[str],
        ground_truth_scores: List[Dict[str, float]],
        retrieval_fn: Callable,
        k_values: List[int] = [1, 3, 5, 10],
        run_name: str = "default_run_with_scores"
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate retrieval function with graded relevance judgments.
        
        Args:
            queries: List of query strings
            ground_truth_scores: List of dictionaries mapping doc IDs to relevance scores
            retrieval_fn: Function that takes a query and returns a list of retrieved documents
            k_values: List of k values to evaluate at
            run_name: Name for this evaluation run
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(queries) != len(ground_truth_scores):
            raise ValueError("Number of queries must match number of ground truth score dictionaries")
        
        # Initialize results
        results = {
            "ndcg": {k: 0.0 for k in k_values},
            "query_level_results": []
        }
        
        # Also track binary metrics for comparison
        binary_ground_truth = [[doc_id for doc_id, score in scores.items() if score > 0]
                              for scores in ground_truth_scores]
        retrieved_lists = []
        
        # Evaluate each query
        for i, (query, relevant_scores) in enumerate(zip(queries, ground_truth_scores)):
            try:
                # Execute query
                retrieved_docs = retrieval_fn(query)
                
                # Extract document IDs
                retrieved = [doc["id"] for doc in retrieved_docs]
                
                # Store for binary metrics
                retrieved_lists.append(retrieved)
                
                # Calculate nDCG for each k
                query_results = {
                    "query": query,
                    "retrieved": retrieved,
                    "relevant_scores": relevant_scores,
                    "metrics": {}
                }
                
                for k in k_values:
                    ndcg = self.normalized_discounted_cumulative_gain(retrieved, relevant_scores, k)
                    
                    # Add to overall results
                    results["ndcg"][k] += ndcg
                    
                    # Add to query-level results
                    query_results["metrics"][f"ndcg@{k}"] = ndcg
                
                # Add to query-level results
                results["query_level_results"].append(query_results)
                
            except Exception as e:
                logger.error(f"Error evaluating query {i} with relevance scores: {str(e)}")
                # Continue with next query
        
        # Calculate average metrics
        num_queries = len(queries)
        for k in k_values:
            results["ndcg"][k] /= num_queries
        
        # Add binary metrics for comparison
        binary_results = self.evaluate_retrieval(
            queries,
            binary_ground_truth,
            retrieval_fn,
            k_values,
            f"{run_name}_binary"
        )
        
        # Merge results
        results["binary_metrics"] = {
            "precision": binary_results["precision"],
            "recall": binary_results["recall"],
            "f1": binary_results["f1"],
            "map": binary_results["map"],
            "mrr": binary_results["mrr"]
        }
        
        # Store results
        self.evaluation_runs[run_name] = results
        
        return results
    
    def compare_runs(self, run_names: List[str], metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Compare multiple evaluation runs.
        
        Args:
            run_names: List of run names to compare
            metrics: List of metrics to compare (default: all)
            
        Returns:
            Dictionary with comparison results
        """
        if not all(run_name in self.evaluation_runs for run_name in run_names):
            missing = [run for run in run_names if run not in self.evaluation_runs]
            raise ValueError(f"Missing evaluation runs: {missing}")
        
        comparison = {}
        
        for run_name in run_names:
            run_results = self.evaluation_runs[run_name]
            
            # Extract all metrics if not specified
            if metrics is None:
                run_metrics = {}
                
                # Extract flat metrics (MAP, MRR)
                for key, value in run_results.items():
                    if key != "query_level_results" and key != "binary_metrics" and isinstance(value, (int, float)):
                        run_metrics[key] = value
                
                # Extract metrics at different k values
                for metric_type, values in run_results.items():
                    if isinstance(values, dict) and metric_type != "query_level_results" and metric_type != "binary_metrics":
                        for k, value in values.items():
                            run_metrics[f"{metric_type}@{k}"] = value
                
                # Extract binary metrics if available
                if "binary_metrics" in run_results:
                    binary_metrics = run_results["binary_metrics"]
                    for metric_type, values in binary_metrics.items():
                        if isinstance(values, dict):
                            for k, value in values.items():
                                run_metrics[f"binary_{metric_type}@{k}"] = value
                        else:
                            run_metrics[f"binary_{metric_type}"] = values
            else:
                # Extract specified metrics
                run_metrics = {}
                for metric in metrics:
                    parts = metric.split("@")
                    if len(parts) == 1:
                        # Flat metric (MAP, MRR)
                        if metric in run_results:
                            run_metrics[metric] = run_results[metric]
                        elif "binary_metrics" in run_results and metric in run_results["binary_metrics"]:
                            run_metrics[metric] = run_results["binary_metrics"][metric]
                    else:
                        # Metric at k (precision@5, ndcg@10)
                        metric_type = parts[0]
                        k = int(parts[1])
                        
                        if metric_type in run_results and k in run_results[metric_type]:
                            run_metrics[metric] = run_results[metric_type][k]
                        elif "binary_metrics" in run_results and metric_type in run_results["binary_metrics"] and k in run_results["binary_metrics"][metric_type]:
                            run_metrics[metric] = run_results["binary_metrics"][metric_type][k]
            
            comparison[run_name] = run_metrics
        
        return comparison
    
    def visualize_comparison(
        self, 
        comparison: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        output_file: str = None
    ) -> None:
        """Visualize comparison results.
        
        Args:
            comparison: Comparison results from compare_runs
            metrics: List of metrics to visualize (default: all)
            output_file: Optional file to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.error("Matplotlib is required for visualization")
            return
        
        # Get all metrics if not specified
        if metrics is None:
            metrics = set()
            for run_metrics in comparison.values():
                metrics.update(run_metrics.keys())
            metrics = sorted(list(metrics))
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set bar width
        bar_width = 0.8 / len(comparison)
        
        # Set up positions
        positions = np.arange(len(metrics))
        
        # Plot bars for each run
        for i, (run_name, run_metrics) in enumerate(comparison.items()):
            # Extract values for each metric (default to 0 if missing)
            values = [run_metrics.get(metric, 0) for metric in metrics]
            
            # Plot bars
            offset = i * bar_width - (len(comparison) - 1) * bar_width / 2
            ax.bar(positions + offset, values, width=bar_width, label=run_name)
        
        # Set labels and title
        ax.set_ylabel('Score')
        ax.set_title('Comparison of Retrieval Approaches')
        ax.set_xticks(positions)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save if output file provided
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved visualization to {output_file}")
        
        # Show plot
        plt.show()
    
    def save_results(self, run_name: str, output_file: str = None) -> None:
        """Save evaluation results to file.
        
        Args:
            run_name: Name of the evaluation run
            output_file: Optional output file (default: {results_dir}/{run_name}.json)
        """
        if run_name not in self.evaluation_runs:
            raise ValueError(f"Unknown evaluation run: {run_name}")
        
        # Default output file
        if output_file is None:
            output_file = os.path.join(self.results_dir, f"{run_name}.json")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(self.evaluation_runs[run_name], f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")
    
    def load_results(self, input_file: str, run_name: str = None) -> Dict[str, Any]:
        """Load evaluation results from file.
        
        Args:
            input_file: Input file path
            run_name: Optional name for the loaded run (default: basename of input file)
            
        Returns:
            Loaded results
        """
        # Default run name from file
        if run_name is None:
            run_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Load results
        with open(input_file, 'r') as f:
            results = json.load(f)
        
        # Store results
        self.evaluation_runs[run_name] = results
        
        logger.info(f"Loaded evaluation results from {input_file} as '{run_name}'")
        
        return results
    
    def record_user_feedback(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        relevance_judgments: List[int],
        feedback_id: str = None
    ) -> str:
        """Record user feedback on retrieved documents.
        
        Args:
            query: Query string
            retrieved_docs: List of retrieved documents
            relevance_judgments: List of relevance judgments (0-3, where 0=not relevant, 3=highly relevant)
            feedback_id: Optional ID for this feedback (default: generated)
            
        Returns:
            Feedback ID
        """
        # Generate feedback ID if not provided
        if feedback_id is None:
            import uuid
            feedback_id = str(uuid.uuid4())
        
        # Create feedback record
        feedback = {
            "query": query,
            "timestamp": import_time().time(),
            "retrieved_docs": [
                {
                    "id": doc["id"],
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {})
                }
                for doc in retrieved_docs
            ],
            "relevance_judgments": relevance_judgments
        }
        
        # Store feedback
        self.feedback_history[feedback_id].append(feedback)
        
        # Save feedback to file
        feedback_file = os.path.join(self.results_dir, f"feedback_{feedback_id}.json")
        with open(feedback_file, 'w') as f:
            json.dump(self.feedback_history[feedback_id], f, indent=2)
        
        logger.info(f"Recorded user feedback with ID {feedback_id}")
        
        return feedback_id
    
    def get_feedback_derived_ground_truth(self, feedback_ids: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Generate ground truth from user feedback.
        
        Args:
            feedback_ids: Optional list of feedback IDs to use (default: all)
            
        Returns:
            Dictionary mapping queries to dictionaries of doc_id -> relevance_score
        """
        ground_truth = defaultdict(dict)
        
        # Use all feedback if not specified
        if feedback_ids is None:
            feedback_ids = list(self.feedback_history.keys())
        
        # Process feedback
        for feedback_id in feedback_ids:
            if feedback_id not in self.feedback_history:
                logger.warning(f"Unknown feedback ID: {feedback_id}")
                continue
            
            for feedback in self.feedback_history[feedback_id]:
                query = feedback["query"]
                
                for i, doc in enumerate(feedback["retrieved_docs"]):
                    doc_id = doc["id"]
                    relevance = feedback["relevance_judgments"][i]
                    
                    # Normalize relevance to 0-1 range
                    normalized_relevance = relevance / 3.0
                    
                    # Update ground truth (average if multiple judgments)
                    if doc_id in ground_truth[query]:
                        # Average with existing score
                        existing_score = ground_truth[query][doc_id]
                        ground_truth[query][doc_id] = (existing_score + normalized_relevance) / 2
                    else:
                        ground_truth[query][doc_id] = normalized_relevance
        
        return dict(ground_truth)


# Helper function for importing time
def import_time():
    import time
    return time 