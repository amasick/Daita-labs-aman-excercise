# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate test data:**
   ```bash
   python3 generate_syndata.py
   ```

## Run Examples

### Demo (Quick start)
```bash
python3 demo.py
```
Shows basic usage with 100 vectors and compares BruteForce vs LSH results.

### Full Evaluation
```bash
python3 evaluate.py
```
Benchmarks both methods on 10,000 vectors with metrics:
- Recall@10, Recall@50, Recall@100
- Query latency comparison

### Generate Fashion-MNIST Data (Optional)
```bash
python3 generate_fmnist.py
```
Downloads and processes Fashion-MNIST images as alternative dataset.

## Key Results

**Default Configuration (Synthetic Dataset):**
- BruteForce: 1.80ms per query (100% recall)
- LSH (12 tables, 7 planes): 1.70ms per query (25% recall@100)
- Trade-off: ~1.1x faster with 25% of true neighbors in top-100

## API Usage

### Create Indices
```python
from vector_search import VectorSearchEngine
import numpy as np

# Prepare normalized vectors
vectors = np.random.randn(1000, 128)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
ids = [f"vec_{i}" for i in range(1000)]

# Create BruteForce index
bf_index = VectorSearchEngine.create_brute_force_index()
bf_index.build(vectors, ids)

# Create LSH index
lsh_index = VectorSearchEngine.create_lsh_index(num_tables=12, num_hyperplanes=7)
lsh_index.build(vectors, ids)
```

### Query
```python
query = vectors[0]  # Query vector (normalized)
k = 10

# Exact search
results = bf_index.query(query, k=k)
for vector_id, similarity in results:
    print(f"{vector_id}: {similarity:.4f}")

# Approximate search
results = lsh_index.query(query, k=k)
```

## Files Reference

| File | Purpose |
|------|---------|
| `vector_search.py` | Core implementation (cosine similarity, indices) |
| `demo.py` | Simple example with 100 vectors |
| `evaluate.py` | Benchmark script with ground truth evaluation |
| `generate_syndata.py` | Create synthetic test data |
| `generate_fmnist.py` | Create Fashion-MNIST test data |
| `requirements.txt` | Dependencies |
| `IMPLEMENTATION.md` | Detailed implementation documentation |

## Tuning Parameters

For different dataset sizes:

| Size | Method | Parameters |
|------|--------|-----------|
| < 1K | BruteForce | N/A |
| 1K-10K | LSH | tables=12, planes=7 |
| 10K-100K | LSH | tables=16, planes=10 |
| > 100K | LSH+ | tables=20, planes=12 or other techniques |

## Performance Notes

- **Initialization:** Vectorized NumPy operations for efficiency
- **Build time:** O(n) for BruteForce, O(nm) for LSH where m = tables × planes
- **Query time:** O(n) for BruteForce, O(m + c) for LSH where c = candidates
- **Memory:** O(n) for BruteForce, O(nm) for LSH

## Troubleshooting

**ImportError for vector_search:**
- Ensure you're in the project directory
- Check PYTHONPATH includes current directory

**No results from LSH:**
- Query might not hash to any populated bucket
- Increase num_tables or decrease num_hyperplanes
- Check vector normalization

**Slow evaluation:**
- With 10K vectors and ground truth for 10 queries, should complete in ~5 seconds
- LSH index building takes ~0.8 seconds due to randomness

## Repository

https://github.com/amasick/ai-dos

Branch: `copilot/implement-vector-similarity-search`
