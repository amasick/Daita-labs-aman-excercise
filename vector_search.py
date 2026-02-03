#!/usr/bin/env python3
"""
Vector Similarity Search Engine Implementation

Implements exact (brute-force) and approximate (LSH) nearest neighbor search
for high-dimensional vectors using cosine similarity.
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict
import time


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    For normalized vectors, this is simply the dot product.
    
    Args:
        vec1: First vector (assumed normalized)
        vec2: Second vector (assumed normalized)
    
    Returns:
        Cosine similarity score
    """
    return np.dot(vec1, vec2)


def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple vectors.
    
    Args:
        query: Query vector of shape (dim,)
        vectors: Array of vectors of shape (n_vectors, dim)
    
    Returns:
        Array of similarity scores of shape (n_vectors,)
    """
    return np.dot(vectors, query)


class BruteForceIndex:
    """
    Exact nearest neighbor search using brute-force comparison.
    
    This computes the similarity between the query and all vectors in the index,
    then returns the top-k most similar vectors. This is the baseline exact method.
    """
    
    def __init__(self):
        self.vectors = None
        self.vector_ids = None
        
    def build(self, vectors: np.ndarray, vector_ids: List[str]):
        """
        Build the index from vectors.
        
        Args:
            vectors: Array of shape (n_vectors, dim) - normalized vectors
            vector_ids: List of vector identifiers
        """
        self.vectors = vectors
        self.vector_ids = vector_ids
        
    def query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Find top-k most similar vectors to the query.
        
        Args:
            query_vector: Query vector (normalized)
            k: Number of neighbors to return
        
        Returns:
            List of tuples (vector_id, similarity_score) sorted by score descending
        """
        if self.vectors is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Compute similarity with all vectors
        similarities = cosine_similarity_batch(query_vector, self.vectors)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Return results as list of tuples
        results = [
            (self.vector_ids[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]
        
        return results


class LSHIndex:
    """
    Approximate nearest neighbor search using Locality Sensitive Hashing (LSH).
    
    Uses random hyperplanes to hash vectors into buckets. Vectors that hash to
    the same bucket are likely to be similar. Uses multiple hash tables to
    improve recall.
    """
    
    def __init__(self, num_tables: int = 10, num_hyperplanes: int = 16):
        """
        Initialize LSH index.
        
        Args:
            num_tables: Number of hash tables to use (more tables = better recall)
            num_hyperplanes: Number of hyperplanes per table (more = finer buckets)
        """
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        self.hash_tables = []
        self.hyperplanes = []
        self.vectors = None
        self.vector_ids = None
        
    def _generate_hyperplanes(self, dim: int):
        """
        Generate random hyperplanes for hashing.
        
        Args:
            dim: Dimension of vectors
        """
        self.hyperplanes = []
        for _ in range(self.num_tables):
            # Generate random unit vectors as hyperplanes
            planes = np.random.randn(self.num_hyperplanes, dim)
            # Normalize to unit vectors
            planes = planes / np.linalg.norm(planes, axis=1, keepdims=True)
            self.hyperplanes.append(planes)
    
    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        """
        Hash a vector using the hyperplanes of a specific table.
        
        Args:
            vector: Vector to hash
            table_idx: Index of hash table to use
        
        Returns:
            Binary hash string
        """
        # Compute dot products with all hyperplanes
        projections = np.dot(self.hyperplanes[table_idx], vector)
        # Create binary hash based on sign of projection
        hash_bits = (projections >= 0).astype(int)
        # Convert to string for use as dictionary key
        return ''.join(map(str, hash_bits))
    
    def build(self, vectors: np.ndarray, vector_ids: List[str]):
        """
        Build the LSH index from vectors.
        
        Args:
            vectors: Array of shape (n_vectors, dim) - normalized vectors
            vector_ids: List of vector identifiers
        """
        self.vectors = vectors
        self.vector_ids = vector_ids
        
        # Generate random hyperplanes
        dim = vectors.shape[1]
        self._generate_hyperplanes(dim)
        
        # Build hash tables
        self.hash_tables = []
        for table_idx in range(self.num_tables):
            hash_table = defaultdict(list)
            
            # Hash each vector and add to buckets
            for i, vector in enumerate(vectors):
                hash_key = self._hash_vector(vector, table_idx)
                hash_table[hash_key].append(i)
            
            self.hash_tables.append(hash_table)
    
    def query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Find approximate top-k most similar vectors to the query.
        
        Args:
            query_vector: Query vector (normalized)
            k: Number of neighbors to return
        
        Returns:
            List of tuples (vector_id, similarity_score) sorted by score descending
        """
        if self.vectors is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Collect candidate indices from all hash tables
        candidate_indices = set()
        
        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(query_vector, table_idx)
            # Get all vectors in the same bucket
            if hash_key in self.hash_tables[table_idx]:
                candidate_indices.update(self.hash_tables[table_idx][hash_key])
        
        # If we have candidates, compute exact similarity only for candidates
        if candidate_indices:
            candidate_indices = list(candidate_indices)
            candidate_vectors = self.vectors[candidate_indices]
            similarities = cosine_similarity_batch(query_vector, candidate_vectors)
            
            # Sort by similarity and take top-k
            sorted_indices = np.argsort(similarities)[::-1][:k]
            
            results = [
                (self.vector_ids[candidate_indices[idx]], float(similarities[idx]))
                for idx in sorted_indices
            ]
        else:
            # No candidates found, return empty list
            results = []
        
        return results


class VectorSearchEngine:
    """
    Main interface for vector similarity search.
    
    Provides factory methods to create and use both exact and approximate search indices.
    """
    
    @staticmethod
    def create_brute_force_index():
        """Create a brute-force (exact) search index."""
        return BruteForceIndex()
    
    @staticmethod
    def create_lsh_index(num_tables: int = 10, num_hyperplanes: int = 16):
        """
        Create an LSH (approximate) search index.
        
        Args:
            num_tables: Number of hash tables
            num_hyperplanes: Number of hyperplanes per table
        """
        return LSHIndex(num_tables=num_tables, num_hyperplanes=num_hyperplanes)
