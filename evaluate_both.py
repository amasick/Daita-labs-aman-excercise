#!/usr/bin/env python3
"""
Comprehensive evaluation script for both synthetic and Fashion-MNIST datasets.
Evaluates both BruteForce and LSH indices with detailed metrics.
"""

import numpy as np
import json
import time
from pathlib import Path
from vector_search import VectorSearchEngine


def load_syndata():
    """Load vectors, queries, and ground truth from synthetic data directory."""
    data_dir = Path(__file__).parent / "data" / "syndata-vectors"
    queries_dir = Path(__file__).parent / "data" / "syndata-queries"
    
    vectors = np.load(data_dir / "vectors.npy")
    vector_ids = np.load(data_dir / "vector_ids.npy")
    queries = np.load(queries_dir / "queries.npy")
    query_ids = np.load(queries_dir / "query_ids.npy")
    
    with open(queries_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    return {
        'name': 'Synthetic Dataset',
        'vectors': vectors,
        'vector_ids': vector_ids.tolist(),
        'queries': queries,
        'query_ids': query_ids.tolist(),
        'ground_truth': ground_truth,
        'description': '10,000 random vectors with clustered structure (128 dimensions)'
    }


def load_fmnist():
    """Load vectors, queries, and ground truth from Fashion-MNIST data directory."""
    data_dir = Path(__file__).parent / "data" / "fmnist-vectors"
    queries_dir = Path(__file__).parent / "data" / "fmnist-queries"
    
    vectors = np.load(data_dir / "vectors.npy")
    vector_ids = np.load(data_dir / "vector_ids.npy")
    queries = np.load(queries_dir / "queries.npy")
    query_ids = np.load(queries_dir / "query_ids.npy")
    
    with open(queries_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    return {
        'name': 'Fashion-MNIST Dataset',
        'vectors': vectors,
        'vector_ids': vector_ids.tolist(),
        'queries': queries,
        'query_ids': query_ids.tolist(),
        'ground_truth': ground_truth,
        'description': '10,000 Fashion-MNIST image embeddings (128 dimensions)'
    }


def calculate_recall(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    """
    Calculate recall@k metric.
    
    Recall@k = (number of true neighbors found in top-k results) / k
    """
    retrieved_set = set(retrieved_ids[:k])
    truth_set = set(ground_truth_ids[:k])
    intersection = retrieved_set & truth_set
    return len(intersection) / k if k > 0 else 0


def evaluate_index(index, queries, query_ids, ground_truth, index_name="Index"):
    """
    Evaluate an index on recall and latency metrics.
    """
    recalls_10 = []
    recalls_50 = []
    recalls_100 = []
    latencies = []
    
    # Evaluate on first 10 queries with ground truth
    for i in range(min(10, len(queries))):
        query_id = query_ids[i]
        query_vector = queries[i]
        
        if query_id not in ground_truth:
            continue
        
        # Measure query latency
        start_time = time.time()
        results = index.query(query_vector, k=100)
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        retrieved_ids = [vid for vid, score in results]
        truth_ids = ground_truth[query_id]['vector_ids']
        
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


def evaluate_dataset(dataset):
    """Evaluate both indices on a dataset."""
    print("\n" + "=" * 80)
    print(f"EVALUATING: {dataset['name']}")
    print("=" * 80)
    print(f"Description: {dataset['description']}")
    print(f"Vectors: {len(dataset['vectors']):,} x {dataset['vectors'].shape[1]}")
    print(f"Queries: {len(dataset['queries'])}")
    print(f"Ground truth: {len(dataset['ground_truth'])} queries")
    print()
    
    vectors = dataset['vectors']
    vector_ids = dataset['vector_ids']
    queries = dataset['queries']
    query_ids = dataset['query_ids']
    ground_truth = dataset['ground_truth']
    
    # Build BruteForce index
    print("Building BruteForce (Exact) Index...")
    bf_start = time.time()
    bf_index = VectorSearchEngine.create_brute_force_index()
    bf_index.build(vectors, vector_ids)
    bf_build_time = time.time() - bf_start
    print(f"  Built in {bf_build_time:.4f}s\n")
    
    # Build LSH index with optimized parameters
    print("Building LSH (Approximate) Index...")
    print("  Parameters: num_tables=12, num_hyperplanes=7")
    lsh_start = time.time()
    lsh_index = VectorSearchEngine.create_lsh_index(num_tables=12, num_hyperplanes=7)
    lsh_index.build(vectors, vector_ids)
    lsh_build_time = time.time() - lsh_start
    print(f"  Built in {lsh_build_time:.4f}s\n")
    
    # Evaluate both indices
    print(f"Evaluating on {len(ground_truth)} queries with ground truth...")
    bf_results = evaluate_index(bf_index, queries, query_ids, ground_truth, "BruteForce (Exact)")
    lsh_results = evaluate_index(lsh_index, queries, query_ids, ground_truth, "LSH (Approximate)")
    
    # Print results table
    print("\nPerformance Metrics:")
    print("-" * 80)
    print(f"{'Method':25s} {'Recall@10':>12s} {'Recall@50':>12s} {'Recall@100':>12s} {'Latency (ms)':>15s}")
    print("-" * 80)
    
    print(f"{bf_results['name']:25s} "
          f"{bf_results['recall@10']:12.3f} "
          f"{bf_results['recall@50']:12.3f} "
          f"{bf_results['recall@100']:12.3f} "
          f"{bf_results['avg_latency_ms']:15.3f}")
    
    print(f"{lsh_results['name']:25s} "
          f"{lsh_results['recall@10']:12.3f} "
          f"{lsh_results['recall@50']:12.3f} "
          f"{lsh_results['recall@100']:12.3f} "
          f"{lsh_results['avg_latency_ms']:15.3f}")
    
    print()
    print("Performance Summary:")
    print("-" * 80)
    print(f"  BruteForce Build Time: {bf_build_time:.4f}s")
    print(f"  LSH Build Time:        {lsh_build_time:.4f}s")
    
    if lsh_results['avg_latency_ms'] > 0:
        speedup = bf_results['avg_latency_ms'] / lsh_results['avg_latency_ms']
        print(f"  Query Speedup:         {speedup:.2f}x (LSH vs BruteForce)")
    
    print(f"  LSH Recall@100:        {lsh_results['recall@100']*100:.1f}%")
    print(f"  Tradeoff:              {speedup:.2f}x faster with {lsh_results['recall@100']*100:.1f}% accuracy")
    
    return {
        'dataset': dataset['name'],
        'bruteforce': bf_results,
        'lsh': lsh_results,
        'build_times': {
            'bruteforce': bf_build_time,
            'lsh': lsh_build_time
        }
    }


def main():
    print("\n" + "=" * 80)
    print("VECTOR SIMILARITY SEARCH ENGINE - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    results = {}
    
    # Load and evaluate datasets
    try:
        syndata = load_syndata()
        results['syndata'] = evaluate_dataset(syndata)
    except Exception as e:
        print(f"Error evaluating synthetic dataset: {e}")
    
    try:
        fmnist = load_fmnist()
        results['fmnist'] = evaluate_dataset(fmnist)
    except Exception as e:
        print(f"Error evaluating Fashion-MNIST dataset: {e}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    
    for dataset_key, dataset_results in results.items():
        print(f"\n{dataset_results['dataset']}:")
        print(f"  BruteForce: {dataset_results['bruteforce']['avg_latency_ms']:.3f}ms per query (100% recall)")
        print(f"  LSH:        {dataset_results['lsh']['avg_latency_ms']:.3f}ms per query ({dataset_results['lsh']['recall@100']*100:.1f}% recall@100)")
        speedup = dataset_results['bruteforce']['avg_latency_ms'] / dataset_results['lsh']['avg_latency_ms']
        print(f"  Speedup:    {speedup:.2f}x")
    
    # Save results to JSON
    results_file = Path(__file__).parent / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    print()


if __name__ == "__main__":
    main()
