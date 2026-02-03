#!/usr/bin/env python3
"""
Demo script for vector similarity search engine.

Demonstrates the basic usage of both BruteForce and LSH indices
with a simple example.
"""

import numpy as np
from vector_search import VectorSearchEngine


def generate_demo_data(n_vectors=100, dim=32, seed=42):
    """
    Generate a small demo dataset for demonstration.
    
    Args:
        n_vectors: Number of vectors to generate
        dim: Vector dimension
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (vectors, vector_ids)
    """
    np.random.seed(seed)
    
    # Generate random vectors and normalize
    vectors = np.random.randn(n_vectors, dim)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Create simple IDs
    vector_ids = [f"vec_{i:03d}" for i in range(n_vectors)]
    
    return vectors, vector_ids


def main():
    print("=" * 70)
    print("Vector Similarity Search Engine - Demo")
    print("=" * 70)
    print()
    
    # Generate demo data
    print("Generating demo dataset...")
    n_vectors = 100
    dim = 32
    vectors, vector_ids = generate_demo_data(n_vectors=n_vectors, dim=dim)
    print(f"  Created {n_vectors} vectors of dimension {dim}")
    print()
    
    # Build indices
    print("Building search indices...")
    
    # BruteForce index
    bf_index = VectorSearchEngine.create_brute_force_index()
    bf_index.build(vectors, vector_ids)
    print("  ✓ BruteForce index built")
    
    # LSH index
    lsh_index = VectorSearchEngine.create_lsh_index(num_tables=5, num_hyperplanes=8)
    lsh_index.build(vectors, vector_ids)
    print("  ✓ LSH index built")
    print()
    
    # Run example queries
    print("Running example queries...")
    print()
    
    # Query 1: Use first vector as query
    query_vector = vectors[0]
    k = 5
    
    print(f"Query 1: Finding top-{k} neighbors for {vector_ids[0]}")
    print("-" * 70)
    
    # BruteForce results
    bf_results = bf_index.query(query_vector, k=k)
    print("BruteForce (Exact) Results:")
    for i, (vid, score) in enumerate(bf_results, 1):
        print(f"  {i}. {vid}: {score:.4f}")
    print()
    
    # LSH results
    lsh_results = lsh_index.query(query_vector, k=k)
    print("LSH (Approximate) Results:")
    if lsh_results:
        for i, (vid, score) in enumerate(lsh_results, 1):
            print(f"  {i}. {vid}: {score:.4f}")
    else:
        print("  No results found (no vectors in same hash bucket)")
    print()
    
    # Query 2: Use a random query vector
    np.random.seed(123)
    query_vector = np.random.randn(dim)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    print(f"Query 2: Finding top-{k} neighbors for random query vector")
    print("-" * 70)
    
    # BruteForce results
    bf_results = bf_index.query(query_vector, k=k)
    print("BruteForce (Exact) Results:")
    for i, (vid, score) in enumerate(bf_results, 1):
        print(f"  {i}. {vid}: {score:.4f}")
    print()
    
    # LSH results
    lsh_results = lsh_index.query(query_vector, k=k)
    print("LSH (Approximate) Results:")
    if lsh_results:
        for i, (vid, score) in enumerate(lsh_results, 1):
            print(f"  {i}. {vid}: {score:.4f}")
    else:
        print("  No results found (no vectors in same hash bucket)")
    print()
    
    # Compare results
    print("Comparison:")
    print("-" * 70)
    if bf_results and lsh_results:
        bf_ids = set(vid for vid, _ in bf_results)
        lsh_ids = set(vid for vid, _ in lsh_results)
        overlap = bf_ids & lsh_ids
        recall = len(overlap) / len(bf_ids)
        print(f"  LSH found {len(overlap)}/{len(bf_ids)} of the exact top-{k} results")
        print(f"  Recall@{k}: {recall:.2%}")
    else:
        print("  Cannot compare (one or both indices returned no results)")
    print()
    
    print("Demo complete!")
    print()
    print("Note: LSH is an approximate method, so it may not always find")
    print("the exact same neighbors as brute-force. The trade-off is that")
    print("it's much faster for large datasets.")


if __name__ == "__main__":
    main()
