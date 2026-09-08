"""
Dataset and Embedding Generation Pipeline for Vector Database From Scratch.
Downloads real text corpus (AG News), cleans and samples 10,000 documents,
generates 384-dimensional dense embeddings using all-MiniLM-L6-v2,
and persists vectors in NumPy format (.npy) and metadata in JSON (.json).
Zero external vector databases or ANN libraries used.
"""

import os
import sys
import json
import html
import argparse
import time
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# Mapping of AG News numerical labels to semantic categories
LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def clean_text(raw_text: str) -> str:
    """
    Cleans raw news article text:
    - Decodes HTML entities (e.g., &amp;, &#39;, &quot;)
    - Removes formatting backslashes and redundant escape characters
    - Normalizes multiple spaces and linebreaks to single spaces
    """
    if not isinstance(raw_text, str):
        return ""
    text = html.unescape(raw_text)
    # Remove literal backslashes used in AG News formatting (e.g. "reuters\ -")
    text = text.replace("\\", " ")
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def load_raw_corpus() -> pd.DataFrame:
    """
    Downloads or retrieves the AG News dataset from Hugging Face Hub cache.
    Returns DataFrame with columns ['text', 'label'].
    """
    print("[1/5] Fetching AG News corpus from Hugging Face Hub...")
    file_path = hf_hub_download(
        repo_id="ag_news",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(file_path)
    print(f"      Loaded {len(df):,} total raw articles.")
    return df


def preprocess_and_sample(
    df: pd.DataFrame,
    num_docs: int = 10000,
    num_queries: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans text, filters low-quality entries, and performs stratified sampling
    to create balanced document and query splits.
    """
    print(f"[2/5] Preprocessing and sampling {num_docs:,} documents & {num_queries:,} queries (seed={seed})...")
    np.random.seed(seed)

    # Clean text column
    df["clean_text"] = df["text"].apply(clean_text)

    # Filter out empty or excessively short documents
    valid_df = df[df["clean_text"].str.len() >= 30].copy()

    # Stratified split: balanced across 4 classes
    classes = sorted(valid_df["label"].unique())
    docs_per_class = num_docs // len(classes)
    queries_per_class = num_queries // len(classes)

    doc_subsets = []
    query_subsets = []

    for c in classes:
        class_df = valid_df[valid_df["label"] == c].sample(
            n=docs_per_class + queries_per_class,
            random_state=seed + c,
            replace=False,
        )
        doc_subsets.append(class_df.iloc[:docs_per_class])
        query_subsets.append(class_df.iloc[docs_per_class:])

    final_docs = pd.concat(doc_subsets).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    final_queries = pd.concat(query_subsets).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"      Sampled {len(final_docs):,} index documents ({docs_per_class} per class).")
    print(f"      Sampled {len(final_queries):,} held-out queries ({queries_per_class} per class).")
    return final_docs, final_queries


def generate_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Generates dense L2-normalized float32 embeddings using SentenceTransformer.
    Important: The model is used exclusively for vector generation, NOT search.
    """
    print(f"[3/5] Loading embedding model '{model_name}'...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    print(f"      Model loaded in {time.perf_counter() - t0:.2f}s.")

    print(f"[4/5] Encoding {len(texts):,} texts in batches of {batch_size}...")
    t_start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Unit-norm L2=1.0
        precision="float32",
    )
    duration = time.perf_counter() - t_start
    throughput = len(texts) / duration
    print(f"      Encoding finished in {duration:.2f}s ({throughput:,.1f} sentences/sec).")
    return np.asarray(embeddings, dtype=np.float32)


def save_dataset(
    docs_df: pd.DataFrame,
    doc_embeddings: np.ndarray,
    queries_df: pd.DataFrame,
    query_embeddings: np.ndarray,
    output_dir: str = "data",
) -> None:
    """
    Saves vectors as NumPy arrays (.npy) and metadata as JSON (.json).
    """
    print(f"[5/5] Persisting vectors and metadata into '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save document vectors as .npy
    docs_npy_path = os.path.join(output_dir, "embeddings.npy")
    np.save(docs_npy_path, doc_embeddings)

    # 2. Save document metadata as .json
    doc_records = []
    for i, row in docs_df.iterrows():
        doc_records.append({
            "id": f"doc_{i:05d}",
            "text": row["clean_text"],
            "metadata": {
                "label": int(row["label"]),
                "category": LABEL_NAMES.get(int(row["label"]), "Unknown"),
                "char_length": len(row["clean_text"]),
                "source": "AG News",
            },
        })

    docs_json_path = os.path.join(output_dir, "documents.json")
    with open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(doc_records, f, indent=2, ensure_ascii=False)

    # 3. Save query vectors as .npy
    queries_npy_path = os.path.join(output_dir, "queries.npy")
    np.save(queries_npy_path, query_embeddings)

    # 4. Save query metadata as .json
    query_records = []
    for i, row in queries_df.iterrows():
        query_records.append({
            "query_id": f"query_{i:04d}",
            "text": row["clean_text"],
            "metadata": {
                "label": int(row["label"]),
                "category": LABEL_NAMES.get(int(row["label"]), "Unknown"),
                "char_length": len(row["clean_text"]),
            },
        })

    queries_json_path = os.path.join(output_dir, "queries.json")
    with open(queries_json_path, "w", encoding="utf-8") as f:
        json.dump(query_records, f, indent=2, ensure_ascii=False)

    # Summary
    print("\nDataset successfully generated and saved:")
    print(f"  • Documents Vectors : {docs_npy_path} (Shape: {doc_embeddings.shape}, Size: {os.path.getsize(docs_npy_path) / (1024*1024):.2f} MB)")
    print(f"  • Documents Metadata: {docs_json_path} ({len(doc_records):,} items, Size: {os.path.getsize(docs_json_path) / (1024*1024):.2f} MB)")
    print(f"  • Query Vectors     : {queries_npy_path} (Shape: {query_embeddings.shape}, Size: {os.path.getsize(queries_npy_path) / 1024:.2f} KB)")
    print(f"  • Query Metadata    : {queries_json_path} ({len(query_records):,} items)")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and dataset for Vector DB From Scratch.")
    parser.add_argument("--num_docs", type=int, default=10000, help="Number of documents to index (default: 10000)")
    parser.add_argument("--num_queries", type=int, default=500, help="Number of held-out query documents (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch_size", type=int, default=128, help="Encoding batch size")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for data files")
    args = parser.parse_args()

    raw_df = load_raw_corpus()
    docs_df, queries_df = preprocess_and_sample(
        raw_df,
        num_docs=args.num_docs,
        num_queries=args.num_queries,
        seed=args.seed,
    )

    all_texts = list(docs_df["clean_text"]) + list(queries_df["clean_text"])
    all_embeddings = generate_embeddings(
        all_texts,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    doc_embeddings = all_embeddings[: len(docs_df)]
    query_embeddings = all_embeddings[len(docs_df) :]

    save_dataset(
        docs_df=docs_df,
        doc_embeddings=doc_embeddings,
        queries_df=queries_df,
        query_embeddings=query_embeddings,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
