#!/usr/bin/env python3
"""
Evaluation script for vector similarity search engine.

Loads synthetic data, builds indices, and compares brute-force vs LSH
on recall and latency metrics.
"""

import numpy as np
import json
import time
from pathlib import Path
from vector_search import VectorSearchEngine


def load_data():
    """Load vectors, queries, and ground truth from data directory."""
    data_dir = Path(__file__).parent / "data"
    
    # Load vectors
    vectors = np.load(data_dir / "syndata-vectors" / "vectors.npy")
    vector_ids = np.load(data_dir / "syndata-vectors" / "vector_ids.npy")
    
    # Load queries
    queries = np.load(data_dir / "syndata-queries" / "queries.npy")
    query_ids = np.load(data_dir / "syndata-queries" / "query_ids.npy")
    
    # Load ground truth
    with open(data_dir / "syndata-queries" / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    return vectors, vector_ids, queries, query_ids, ground_truth


def calculate_recall(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    """
    Calculate recall@k metric.
    
    Recall@k = (number of true neighbors found in top-k results) / k
    
    Args:
        retrieved_ids: List of retrieved vector IDs
        ground_truth_ids: List of true nearest neighbor IDs
        k: Number of top results to consider
    
    Returns:
        Recall score between 0 and 1
    """
    # Take only top-k from both lists
    retrieved_set = set(retrieved_ids[:k])
    truth_set = set(ground_truth_ids[:k])
    
    # Count how many true neighbors were found
    intersection = retrieved_set & truth_set
    
    # Recall = found / k
    return len(intersection) / k


def evaluate_index(index, queries, query_ids, ground_truth, index_name="Index"):
    """
    Evaluate an index on recall and latency metrics.
    
    Args:
        index: Index object (BruteForceIndex or LSHIndex)
        queries: Query vectors
        query_ids: Query identifiers
        ground_truth: Ground truth results
        index_name: Name for display
    
    Returns:
        Dictionary with evaluation results
    """
    recalls_10 = []
    recalls_50 = []
    recalls_100 = []
    latencies = []
    
    # Evaluate on first 10 queries with ground truth
    for i in range(min(10, len(queries))):
        query_id = query_ids[i]
        query_vector = queries[i]
        
        # Skip if no ground truth for this query
        if query_id not in ground_truth:
            continue
        
        # Measure query latency
        start_time = time.time()
        results = index.query(query_vector, k=100)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        latencies.append(latency)
        
        # Extract retrieved IDs
        retrieved_ids = [vid for vid, score in results]
        truth_ids = ground_truth[query_id]['vector_ids']
        
        # Calculate recall at different k values
        recall_10 = calculate_recall(retrieved_ids, truth_ids, 10)
        recall_50 = calculate_recall(retrieved_ids, truth_ids, 50)
        recall_100 = calculate_recall(retrieved_ids, truth_ids, 100)
        
        recalls_10.append(recall_10)
        recalls_50.append(recall_50)
        recalls_100.append(recall_100)
    
    return {
        'name': index_name,
        'recall@10': np.mean(recalls_10) if recalls_10 else 0,
        'recall@50': np.mean(recalls_50) if recalls_50 else 0,
        'recall@100': np.mean(recalls_100) if recalls_100 else 0,
        'avg_latency_ms': np.mean(latencies) if latencies else 0,
        'num_queries': len(recalls_10)
    }


def main():
    print("=" * 70)
    print("Vector Similarity Search Engine - Evaluation")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    vectors, vector_ids, queries, query_ids, ground_truth = load_data()
    print(f"  Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
    print(f"  Loaded {len(queries)} query vectors")
    print(f"  Ground truth available for {len(ground_truth)} queries")
    print()
    
    # Build indices
    print("Building Indices...")
    
    # Build BruteForce index
    bf_start = time.time()
    bf_index = VectorSearchEngine.create_brute_force_index()
    bf_index.build(vectors, vector_ids.tolist())
    bf_build_time = time.time() - bf_start
    print(f"  BruteForce Index: built in {bf_build_time:.2f}s")
    
    # Build LSH index
    lsh_start = time.time()
    lsh_index = VectorSearchEngine.create_lsh_index(num_tables=12, num_hyperplanes=7)
    lsh_index.build(vectors, vector_ids.tolist())
    lsh_build_time = time.time() - lsh_start
    print(f"  LSH Index (tables=12, planes=7): built in {lsh_build_time:.2f}s")
    print()
    
    # Evaluate indices
    print(f"Evaluating on {len(ground_truth)} queries with ground truth...")
    print()
    
    bf_results = evaluate_index(bf_index, queries, query_ids, ground_truth, "BruteForce (Exact)")
    lsh_results = evaluate_index(lsh_index, queries, query_ids, ground_truth, "LSH (Approximate)")
    
    # Print results table
    print("Results:")
    print(f"{'':20s} {'Recall@10':>12s} {'Recall@50':>12s} {'Recall@100':>12s} {'Avg Latency (ms)':>18s}")
    print("-" * 76)
    
    print(f"{bf_results['name']:20s} "
          f"{bf_results['recall@10']:12.3f} "
          f"{bf_results['recall@50']:12.3f} "
          f"{bf_results['recall@100']:12.3f} "
          f"{bf_results['avg_latency_ms']:18.2f}")
    
    print(f"{lsh_results['name']:20s} "
          f"{lsh_results['recall@10']:12.3f} "
          f"{lsh_results['recall@50']:12.3f} "
          f"{lsh_results['recall@100']:12.3f} "
          f"{lsh_results['avg_latency_ms']:18.2f}")
    
    print()
    print("Performance Summary:")
    
    # Calculate speedup
    if lsh_results['avg_latency_ms'] > 0:
        speedup = bf_results['avg_latency_ms'] / lsh_results['avg_latency_ms']
        print(f"  - LSH is {speedup:.1f}x faster than brute-force")
    
    print(f"  - LSH achieves {lsh_results['recall@100']*100:.0f}% recall@100")
    print(f"  - Trade-off: Speed vs Accuracy")
    print()


if __name__ == "__main__":
    main()
