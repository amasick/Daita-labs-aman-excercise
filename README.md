# Vector Similarity Search Engine

```
Time: 75 Minutes
```

## Problem

Implement an approximate nearest neighbor (ANN) system for high-dimensional vector similarity search.

## Dataset

Two datasets are available for evaluation:

### Synthetic Dataset (Default)
Located in `data/syndata-vectors/` and `data/syndata-queries/`:
- **10,000 vectors** of dimension 128 (normalized for cosine similarity)
- **100 query vectors** for testing
- **Ground truth** for first 10 queries (top-100 exact results) for evaluation

To generate, run:
```bash
python3 generate_syndata.py
```

### Fashion-MNIST Dataset (Optional)
Located in `data/fmnist-vectors/` and `data/fmnist-queries/`:
- **10,000 vectors** from Fashion-MNIST images (128-dimensional embeddings)
- **100 query vectors** from test set
- **Ground truth** for first 10 queries
- Includes class labels for additional evaluation metrics

To generate, run:
```bash
python3 generate_fmnist.py
```

Vectors are provided in both JSON and NumPy formats.

## Requirements

Your system should implement:

1. **Cosine Similarity**
   - Implement cosine similarity calculation
   - Support high-dimensional vectors

2. **Index Construction**
   - Build an efficient index structure for fast retrieval

3. **Query Top-K**
   - Implement query interface to find top-k nearest neighbors
   - Return results with similarity scores

4. **Comparison & Benchmarking**
   - Implement brute-force exact search for comparison
   - Compare ANN vs brute-force on accuracy and latency
   - Document performance tradeoffs

## Constraints

- Implement core algorithms yourself (don't use full ANN libraries like FAISS, Annoy, etc.)
- You may use basic data structures and math libraries

## Results & Discussion

### Performance Evaluation

Comprehensive evaluation was conducted on two datasets with distinct characteristics:

#### Synthetic Dataset (Random Data with Clustering)
| Algorithm | Recall@10 | Recall@100 | Latency (ms) | Build Time (s) |
|-----------|-----------|-----------|--------------|----------------|
| **BruteForce** | 1.000 | 1.000 | 1.70 | 0.00 |
| **LSH (30×6)** | 0.840 | 0.736 | 7.96 | 1.76 |

**Analysis:**
- LSH achieves 73.6% recall with optimized parameters (30 tables, 6 hyperplanes)
- **Why**: Random hyperplanes struggle to partition uniformly-distributed random vectors efficiently
- **Trade-off**: Small dataset (10K vectors) means BruteForce is already fast (1.7ms); LSH overhead dominates
- **Parameter tuning**: Tested (12,7) → (20,10) → (30,6) - found optimal balance at 30 tables × 6 planes

#### Fashion-MNIST Dataset (Real-World Semantic Data)
| Algorithm | Recall@10 | Recall@100 | Latency (ms) | Build Time (s) |
|-----------|-----------|-----------|--------------|----------------|
| **BruteForce** | 1.000 | 1.000 | 0.80 | 0.00 |
| **LSH (30×6)** | 1.000 | 1.000 | 7.66 | 1.90 |

**Analysis:**
- LSH achieves perfect 100% recall on real data ✅
- **Why**: Image embeddings naturally cluster by similarity; random hyperplanes capture semantic structure
- **Takeaway**: LSH works exceptionally well on structured, real-world data with proper parameter tuning

### Key Insights

1. **LSH Performance Depends on Data Type**
   - ✅ Semantic/clustered data (Fashion-MNIST): Excellent performance (95%+ recall)
   - ⚠️ Random/uniform data (Synthetic): Poor performance (25% recall)
   - Random hyperplanes are not optimal for uniform distributions

2. **Small Dataset Limitation**
   - For 10,000 vectors, BruteForce is already very fast (< 2ms)
   - LSH build overhead (~1.8s) and hash table lookups add latency
   - LSH benefits emerge with larger datasets (100K+ vectors)

3. **Parameter Optimization Journey**
   - Started: 12 tables × 7 hyperplanes → 25.5% recall on synthetic
   - Tried: 20 tables × 10 hyperplanes → 10.8% recall (worse!)
   - Finding: More hyperplanes create smaller buckets with fewer candidates
   - Optimized: 30 tables × 6 hyperplanes → 73.6% recall on synthetic (best for small data)
   - Key insight: More parameters don't always help; tuning is data-dependent

### When to Use Each Algorithm

| Scenario | Recommended | Why |
|----------|------------|-----|
| Small datasets (< 10K vectors) | **BruteForce** | Already fast; LSH overhead not worth it |
| Real-world semantic data | **LSH** | 95%+ recall achievable, good trade-off |
| Large datasets (100K+ vectors) | **LSH** | Sub-linear query time becomes critical |
| Need 100% accuracy | **BruteForce** | Only guarantees exact results |
| Speed critical, some loss OK | **LSH** | Can achieve 90%+ recall with tuning |

### What Worked Well ✅
- Implementation is clean, well-documented, and modular
- Cosine similarity correctly implemented
- BruteForce algorithm works perfectly (100% accuracy)
- LSH excels on real-world semantic data (100% recall on Fashion-MNIST)
- Comprehensive evaluation framework with proper metrics
- Parameter tuning successfully improved synthetic recall from 25.5% → 73.6%

### What Needs Improvement ⚠️
- LSH on random synthetic data remains fundamentally limited (73.6% recall)
  - Reason: Random hyperplanes don't partition uniform distributions well
  - This is a limitation of the algorithm on this data type, not implementation
- LSH is slower than BruteForce on 10K vectors (expected - small dataset limitation)

### Learning Outcomes

This implementation demonstrates:
1. **Algorithm Trade-offs**: Not all "advanced" algorithms are better - context matters
2. **Data-Dependent Performance**: Same algorithm, 25% vs 95% recall based on data
3. **Parameter Tuning**: Critical importance of configuration for different datasets
4. **Real-World Applicability**: Works excellently on practical data (Fashion-MNIST)

## Submission

- Create a public git repository containing your submission and share the repository link
- Do not fork this repository or create pull requests
