# Quick Start Guide - Vector Similarity Search Engine

## 📋 Table of Contents
1. [Installation](#installation)
2. [Data Generation](#data-generation)
3. [Running Examples](#running-examples)
4. [Comprehensive Evaluation](#comprehensive-evaluation)
5. [API Usage](#api-usage)
6. [Parameter Tuning](#parameter-tuning)
7. [Results & Performance](#results--performance)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- NumPy (for efficient vector operations)

### Step 2: Verify Installation
```bash
python -c "import numpy; import vector_search; print('Ready!')"
```

---

## Data Generation

### Synthetic Dataset (Recommended for Quick Start)
Generate 10,000 random vectors with clustering structure:
```bash
python3 generate_syndata.py
```

**Output:**
- 10,000 vectors (128 dimensions) - `data/syndata-vectors/`
- 100 query vectors - `data/syndata-queries/`
- Ground truth for first 10 queries
- Size: ~9.77 MB

**Use case:** Test on random data with cluster structure

### Fashion-MNIST Dataset (Real-World Data)
Generate image embeddings from Fashion-MNIST dataset:
```bash
python3 generate_fmnist.py
```

**Output:**
- 10,000 image embeddings (128 dimensions) - `data/fmnist-vectors/`
- 100 query vectors from test set - `data/fmnist-queries/`
- Ground truth for first 10 queries
- Class labels (10 clothing types)
- Size: ~4.88 MB

**Use case:** Test on semantically meaningful data

---

## Running Examples

### 1. Demo - Quick Interactive Example
Run a simple demo with 100 vectors:
```bash
python3 demo.py
```

**What it does:**
- Generates 100 random vectors
- Builds both BruteForce and LSH indices
- Runs 2 example queries
- Compares results between exact and approximate methods
- Computes recall metric

**Expected output:**
```
Vector Similarity Search Engine - Demo
Generated 100 vectors of dimension 32
Building indices...
BruteForce index: OK
LSH index: OK
Query 1 results: 5 neighbors found
Query 2 results: 4 neighbors found
```

**Execution time:** < 1 second

---

### 2. Evaluation on Synthetic Data
Evaluate both algorithms on synthetic dataset:
```bash
python3 evaluate.py
```

**What it does:**
- Loads 10,000 synthetic vectors
- Builds both indices
- Evaluates on 10 ground truth queries
- Computes: Recall@10, Recall@50, Recall@100, Query Latency
- Displays comparison table

**Expected output:**
```
Loading data...
Loaded 10,000 vectors (128 dimensions)
Loaded 100 queries

Building Indices...
BruteForce Index: built in 0.00s
LSH Index: built in 0.71s

Results:
                      Recall@10  Recall@50  Recall@100  Avg Latency
BruteForce (Exact)    1.000      1.000      1.000       1.08 ms
LSH (Approximate)     0.310      0.284      0.255       1.33 ms
```

**Execution time:** ~10 seconds

---

## Comprehensive Evaluation

### Evaluate BOTH Datasets
For complete evaluation on both synthetic and Fashion-MNIST data:
```bash
python3 evaluate_both.py
```

**What it does:**
1. Loads both datasets
2. Builds indices for each
3. Evaluates all 4 combinations:
   - Synthetic + BruteForce
   - Synthetic + LSH
   - Fashion-MNIST + BruteForce
   - Fashion-MNIST + LSH
4. Computes all metrics
5. Saves detailed results to `evaluation_results.json`

**Expected output:**
```
EVALUATING: Synthetic Dataset
Vectors: 10,000 x 128
Building BruteForce (Exact) Index...
  Built in 0.0000s
Building LSH (Approximate) Index...
  Parameters: num_tables=12, num_hyperplanes=7
  Built in 0.7140s

Performance Metrics:
                          Recall@10  Recall@50  Recall@100  Latency (ms)
BruteForce (Exact)        1.000      1.000      1.000       1.082
LSH (Approximate)         0.310      0.284      0.255       1.328

EVALUATING: Fashion-MNIST Dataset
Vectors: 10,000 x 128
Building indices...
  Built in 0.0000s / 0.8005s

Performance Metrics:
                          Recall@10  Recall@50  Recall@100  Latency (ms)
BruteForce (Exact)        1.000      1.000      1.000       0.419
LSH (Approximate)         0.980      0.956      0.951       4.605

FINAL COMPARISON SUMMARY
Synthetic Dataset:
  BruteForce: 1.082ms per query (100% recall)
  LSH:        1.328ms per query (25.5% recall@100)
Fashion-MNIST Dataset:
  BruteForce: 0.419ms per query (100% recall)
  LSH:        4.605ms per query (95.1% recall@100)
```

**Execution time:** ~20-30 seconds (includes downloads if first run)

**Results file:** `evaluation_results.json` (structured JSON with all metrics)

---

## API Usage

### Basic Setup

```python
import numpy as np
from vector_search import VectorSearchEngine

# Create or load your vectors (must be normalized!)
vectors = np.random.randn(1000, 128)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Create vector IDs
ids = [f"vec_{i:04d}" for i in range(len(vectors))]
```

### Creating Indices

#### Option 1: BruteForce (Exact Search)
```python
# Create and build index
bf_index = VectorSearchEngine.create_brute_force_index()
bf_index.build(vectors, ids)
```

#### Option 2: LSH (Approximate Search)
```python
# Create with custom parameters
lsh_index = VectorSearchEngine.create_lsh_index(
    num_tables=12,           # Number of hash tables (more = better recall)
    num_hyperplanes=7        # Hyperplanes per table (controls bucket granularity)
)
lsh_index.build(vectors, ids)
```

### Querying

```python
# Prepare a normalized query vector
query = np.random.randn(128)
query = query / np.linalg.norm(query)

# Get top-10 nearest neighbors
k = 10

# Exact search
bf_results = bf_index.query(query, k=k)
print(f"BruteForce found {len(bf_results)} neighbors:")
for vector_id, similarity_score in bf_results:
    print(f"  {vector_id}: {similarity_score:.4f}")

# Approximate search
lsh_results = lsh_index.query(query, k=k)
print(f"LSH found {len(lsh_results)} neighbors:")
for vector_id, similarity_score in lsh_results:
    print(f"  {vector_id}: {similarity_score:.4f}")
```

### Computing Cosine Similarity Directly

```python
from vector_search import cosine_similarity, cosine_similarity_batch

# Single vector pair
sim = cosine_similarity(vec1, vec2)

# Query against multiple vectors
similarities = cosine_similarity_batch(query_vec, all_vectors)
```

---

## Parameter Tuning

### Understanding Parameters

**num_tables**
- Number of independent hash tables
- **Effect on recall:** More tables = better recall (more diverse candidates)
- **Effect on speed:** More tables = slower (more hash table lookups)
- **Effect on memory:** O(n × num_tables × num_hyperplanes)
- **Range:** 5-20 typical

**num_hyperplanes**
- Hyperplanes per table = 2^num_hyperplanes hash buckets
- **Effect on recall:** More planes = smaller buckets = fewer candidates
- **Effect on speed:** Larger buckets = more candidates = slower
- **Effect on memory:** O(n × num_tables × num_hyperplanes)
- **Range:** 5-15 typical

### Tuning by Dataset Size

| Dataset Size | Method | Recommended Parameters | Rationale |
|--------------|--------|------------------------|-----------|
| < 1K | BruteForce | N/A | Brute-force faster than LSH overhead |
| 1K-10K | LSH | tables=12, planes=7 | Good balance, default recommendation |
| 10K-100K | LSH | tables=16, planes=10 | More candidates for accuracy |
| 100K-1M | LSH | tables=20, planes=12 | Optimize for large-scale |
| > 1M | Other techniques | See notes | Consider HNSW, Spectral hashing, etc. |

### Tuning by Data Type

#### Random/Uniform Data
- **Problem:** Poor recall (25-30%)
- **Solution:** Increase num_tables (16-20)
- **Example:** `num_tables=20, num_hyperplanes=7`

#### Semantic/Clustered Data
- **Problem:** Large candidate sets, slower queries
- **Solution:** Reduce num_tables if latency critical
- **Example:** `num_tables=8, num_hyperplanes=7` for Fashion-MNIST

#### Balanced Configuration
- **Goal:** 70-85% recall with reasonable latency
- **Recommended:** `num_tables=14, num_hyperplanes=8`

---

## Results & Performance

### Summary of Evaluation Results

See [RESULTS.md](RESULTS.md) for comprehensive analysis of both datasets.

### Quick Performance Comparison

**Synthetic Dataset (Random Data)**
| Method | Latency | Recall@100 | Best For |
|--------|---------|-----------|----------|
| BruteForce | 1.08 ms | 100% | Accuracy critical |
| LSH | 1.33 ms | 25.5% | (not recommended for this data) |

**Fashion-MNIST Dataset (Semantic Data)**
| Method | Latency | Recall@100 | Best For |
|--------|---------|-----------|----------|
| BruteForce | 0.42 ms | 100% | Small datasets |
| LSH | 4.61 ms | 95.1% | Large-scale, good accuracy trade-off |

### Key Insights

1. **LSH shines with semantic data** - Achieved 95.1% recall on Fashion-MNIST
2. **LSH struggles with random data** - Only 25.5% recall on random vectors
3. **Candidate set size determines latency** - More similar vectors = larger candidate sets = slower
4. **Data distribution matters most** - Not dataset size, but how vectors cluster

---

## File Reference

| File | Purpose | Size |
|------|---------|------|
| `vector_search.py` | Core ANN implementation | 8.3 KB |
| `demo.py` | Simple interactive demo | 4.3 KB |
| `evaluate.py` | Single-dataset benchmark | 6.7 KB |
| `evaluate_both.py` | Two-dataset evaluation | 10.2 KB |
| `generate_syndata.py` | Synthetic data generator | 4.2 KB |
| `generate_fmnist.py` | Fashion-MNIST processor | 8.8 KB |
| `requirements.txt` | Dependencies | 0.3 KB |
| `RESULTS.md` | Comprehensive analysis | 15+ KB |
| `IMPLEMENTATION.md` | Technical documentation | 10.8 KB |

---

## Troubleshooting

### Problem: "No module named 'numpy'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Problem: "ImportError: No module named 'vector_search'"
**Solution:** Ensure you're in the project root directory
```bash
cd c:\Users\akash\Desktop\Daita-excercise\ai-dos
python demo.py
```

### Problem: "Data not found" when running evaluate.py
**Solution:** Generate data first
```bash
python3 generate_syndata.py
python3 generate_fmnist.py
```

### Problem: LSH returns few or no results
**Causes:**
- Query vector not normalized
- num_hyperplanes too large (buckets too small)
- Data very random (hash buckets don't overlap)

**Solutions:**
1. Check vector normalization: `norm = np.linalg.norm(vec); vec = vec / norm`
2. Reduce num_hyperplanes: `create_lsh_index(num_tables=12, num_hyperplanes=5)`
3. Increase num_tables for more candidates: `create_lsh_index(num_tables=20, num_hyperplanes=7)`

### Problem: Evaluation runs slowly
- LSH index building: ~0.8 seconds (expected)
- Full two-dataset evaluation: ~20-30 seconds (expected)
- First Fashion-MNIST generation: ~1 minute (downloading datasets)

---

## Next Steps

1. **Understand your data**
   - Run `generate_syndata.py` and `generate_fmnist.py`
   - Compare results with `evaluate_both.py`

2. **Experiment with parameters**
   - Modify num_tables and num_hyperplanes in your code
   - Re-run evaluation to see impact

3. **Integrate into your project**
   - Import VectorSearchEngine
   - Build index from your vectors
   - Run queries

4. **Scale up**
   - For larger datasets (> 100K), consider other techniques
   - See IMPLEMENTATION.md for advanced topics

---

## Documentation Links

- **[RESULTS.md](RESULTS.md)** - Comprehensive evaluation results and analysis
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Technical architecture and algorithms
- **[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)** - Requirements verification
- **[PROJECT_COMPLETE.md](PROJECT_COMPLETE.md)** - Project completion summary

---

## Repository

**GitHub:** https://github.com/amasick/ai-dos

**Branch:** `copilot/implement-vector-similarity-search`

---

## Summary

This Vector Similarity Search Engine provides:
- ✅ Exact nearest neighbor search (BruteForce)
- ✅ Approximate nearest neighbor search (LSH)
- ✅ Comprehensive benchmarking framework
- ✅ Two datasets for evaluation
- ✅ Detailed documentation and examples

Start with `python3 demo.py` to see it in action!
