"""
Generates 500 query vectors and computes exact top-10 ground truth using ExactIndex.
"""

import os
import json
import time
from typing import Any
import numpy as np
from app.core.exact_index import ExactIndex


def sample_or_generate_queries(
    vectors: np.ndarray,
    num_queries: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample or generate 500 query vectors.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(vectors)
    if n_samples >= num_queries:
        indices = rng.choice(n_samples, size=num_queries, replace=False)
        return vectors[indices].copy()
    else:
        # Generate on same unit sphere
        dim = vectors.shape[1]
        q = rng.normal(size=(num_queries, dim)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q


def compute_ground_truth(
    queries: np.ndarray,
    exact_index: ExactIndex,
    k: int = 10,
) -> tuple[list[list[Any]], list[list[float]], list[float]]:
    """
    Computes exact top-k nearest neighbor IDs and scores for each query vector
    using ExactIndex. Also records query latencies.

    Returns:
        (ground_truth_ids, ground_truth_scores, query_latencies_ms)
    """
    gt_ids = []
    gt_scores = []
    latencies = []

    for q in queries:
        t0 = time.perf_counter_ns()
        res = exact_index.search(q, k=k)
        t1 = time.perf_counter_ns()

        gt_ids.append(list(res.ids))
        gt_scores.append([float(s) for s in res.scores])
        latencies.append((t1 - t0) / 1e6)

    return gt_ids, gt_scores, latencies


def save_ground_truth(
    filepath: str,
    queries: np.ndarray,
    ground_truth_ids: list[list[Any]],
    ground_truth_scores: list[list[float]],
) -> None:
    """Save queries and ground truth matrices to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "num_queries": len(queries),
        "k": len(ground_truth_ids[0]) if ground_truth_ids else 0,
        "ground_truth_ids": ground_truth_ids,
        "ground_truth_scores": ground_truth_scores,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    npy_path = filepath.replace(".json", "_queries.npy")
    np.save(npy_path, queries)


def load_ground_truth(filepath: str) -> tuple[np.ndarray, list[list[Any]]]:
    """Load queries and ground truth matrices from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    npy_path = filepath.replace(".json", "_queries.npy")
    queries = np.load(npy_path)
    return queries, data["ground_truth_ids"]

