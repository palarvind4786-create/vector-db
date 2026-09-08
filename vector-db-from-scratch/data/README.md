# Dataset & Embeddings Documentation

This directory contains the real text corpus, pre-computed dense embeddings, and metadata used for indexing and benchmarking **Vector Database From Scratch**.

---

## 1. Overview Specifications

| Attribute | Specification |
| :--- | :--- |
| **Corpus Source** | AG News Topic Classification Dataset |
| **Number of Index Documents** | **10,000** short news texts |
| **Number of Held-Out Queries** | **500** distinct query texts (for ground truth evaluation) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Embedding Dimension ($D$)** | **384** dimensions |
| **Data Type** | `float32` (IEEE 754 single-precision float) |
| **Vector Normalization** | Unit L2-norm ($\|x\|_2 = 1.0$) for dot-product cosine equivalence |
| **Storage Format** | NumPy binary (`.npy`) for vectors; JSON (`.json`) for metadata |
| **Zero ANN Library Rule** | The embedding model is used **strictly for vector inference**. It performs **no indexing or search**. |

---

## 2. Storage Format & Layout

```
data/
├── embeddings.npy     # 10,000 x 384 NumPy float32 matrix (14.65 MB)
├── documents.json     # 10,000 document records with text & metadata (3.97 MB)
├── queries.npy        # 500 x 384 NumPy float32 matrix (750 KB)
├── queries.json       # 500 query records with text & metadata
└── README.md          # Pipeline documentation
```

### File Structure:
- **`embeddings.npy`**: 2D binary matrix of shape `(10000, 384)` with contiguous row-major layout.
- **`documents.json`**: List of dictionaries:
  ```json
  [
    {
      "id": "doc_00000",
      "text": "FDA Weighs Antidepressant Risks for Kids BETHESDA, Md. (Reuters) - A U.S. advisory panel began meeting Monday...",
      "metadata": {
        "label": 2,
        "category": "Business",
        "char_length": 267,
        "source": "AG News"
      }
    }
  ]
  ```
- **`queries.npy`**: 2D binary matrix of shape `(500, 384)` containing held-out test queries disjoint from the index documents.
- **`queries.json`**: List of 500 evaluation query texts.

---

## 3. Preprocessing Steps

1. **HTML Entity Unescaping**: Decoded standard HTML entities (such as `&amp;` $\to$ `&`, `&#39;` $\to$ `'`, `&quot;` $\to$ `"`, `&gt;` $\to$ `>`).
2. **Backslash Artifact Cleanup**: Stripped literal backslashes from legacy news wire markers (e.g. `Reuters\ -` $\to$ `Reuters -`).
3. **Whitespace Normalization**: Collapsed consecutive whitespace, tabs, and newline characters into single spaces.
4. **Length Filtering**: Filtered out invalid, truncated, or short snippets ($< 30$ characters).
5. **Stratified Balanced Sampling**:
   - Sampled evenly across all 4 AG News categories:
     - Class 0: **World** (2,500 docs, 125 queries)
     - Class 1: **Sports** (2,500 docs, 125 queries)
     - Class 2: **Business** (2,500 docs, 125 queries)
     - Class 3: **Sci/Tech** (2,500 docs, 125 queries)
6. **Reproducible Split**: Used a fixed pseudo-random seed (`seed=42`) ensuring identical document selection across runs.
7. **L2 Normalization**: Each generated vector is normalized to $\|v\|_2 = 1.0$. This ensures that cosine similarity $\frac{u \cdot v}{\|u\| \|v\|}$ simplifies to a pure matrix dot product $u \cdot v$, enabling fast BLAS GEMM computation.

---

## 4. Regenerating the Dataset

To regenerate or resize the dataset, run the provided generation script:

```bash
# Generate default 10,000 documents and 500 queries
python scripts/generate_dataset.py --num_docs 10000 --num_queries 500 --seed 42

# Optional arguments:
# --num_docs       Number of documents to index (default: 10000)
# --num_queries    Number of held-out query texts (default: 500)
# --seed           Random seed for reproducibility (default: 42)
# --batch_size     Inference batch size (default: 128)
# --output_dir     Destination folder (default: data)
```
