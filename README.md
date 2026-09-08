# Vector Database From Scratch

An educational, portfolio-grade vector database built from fundamental mathematical principles using **NumPy** for vectorized linear algebra.

> **STRICT ZERO-ANN / ZERO-VECTOR-DB RESTRICTIONS**:
> This project implements all core vector database concepts without using external vector databases or approximate nearest neighbor (ANN) / clustering libraries:
> - **NO** FAISS
> - **NO** Pinecone, Chroma, Milvus, Weaviate
> - **NO** Annoy, hnswlib
> - **NO** `sklearn.neighbors` or `sklearn.cluster.KMeans`

---

## Key Features

1. **Exact Vector Index**: NumPy-vectorized brute-force linear scan baseline.
2. **Cosine Similarity & Top-K**: Pure mathematical implementation optimized via unit-norm dot product and $O(N + k \log k)$ partition selection.
3. **K-Means Clustering from Scratch**: Custom Lloyd's algorithm with K-Means++ initialization and empty-cluster outlier recovery.
4. **IVF-Flat Index (Inverted File Flat)**: Inverted posting lists, Voronoi cell routing, and multi-cluster probing (`nprobe`).
5. **Dynamic Document Operations**: Insert, query, lazy deletion (tombstoning), and full index rebuilding/compaction.
6. **Real Text Corpus & Embeddings**: Semantic text search over a real corpus using dense embeddings (`all-MiniLM-L6-v2`).
7. **Comprehensive Benchmark Suite**: 500 query vectors evaluated against exact top-10 ground truth; measures Recall@10, Mean/P50/P95/P99 latency, vectors examined, and speedup factors.
8. **FastAPI REST API**: Asynchronous endpoints for ingestion, search, deletion, rebuild, stats, and benchmarks.
9. **Interactive Web Dashboard**: Single-page browser interface featuring real-time search, side-by-side Exact vs IVF comparison, cluster distribution visualizer, and benchmark metrics.
10. **Persistence**: Zero-dependency state serialization using `.npz` vector arrays and `.json` metadata.

---

## Directory Structure

```
vector-db-from-scratch/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── similarity.py
│   │   ├── exact_index.py
│   │   ├── kmeans.py
│   │   ├── ivf_index.py
│   │   └── vector_store.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py
│
├── data/
│   └── README.md
│
├── experiments/
│   ├── benchmark.py
│   ├── ground_truth.py
│   └── metrics.py
│
├── tests/
│   ├── __init__.py
│   ├── test_similarity.py
│   ├── test_exact_index.py
│   ├── test_kmeans.py
│   ├── test_ivf_index.py
│   ├── test_vector_store.py
│   └── test_api.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Getting Started

### 1. Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Application
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### 3. Running Experiments & Tests
```bash
pytest tests/
python -m experiments.benchmark
```
