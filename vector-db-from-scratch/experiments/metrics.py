"""
Evaluation metrics: Recall@K, Latency Percentiles (Mean, P50, P95, P99), and Speedup.
"""

from typing import Any, Sequence
import numpy as np


def compute_recall_at_k(
    ground_truth: Sequence[Sequence[Any]],
    predictions: Sequence[Sequence[Any]],
    k: int = 10,
) -> float:
    """
    Computes average Recall@K across all queries:
        Recall@K = (1 / Q) * sum( |retrieved_k ∩ ground_truth_k| / k )

    Parameters:
        ground_truth: List of true nearest neighbor IDs for each query.
        predictions: List of retrieved neighbor IDs for each query.
        k: Top-K cutoff.

    Returns:
        Float value in range [0.0, 1.0].
    """
    if len(ground_truth) == 0 or len(predictions) == 0:
        return 0.0

    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Queries count mismatch: ground_truth ({len(ground_truth)}) != predictions ({len(predictions)})."
        )

    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(list(gt)[:k])
        pred_set = set(list(pred)[:k])
        if len(gt_set) == 0:
            recalls.append(0.0)
        else:
            intersection = len(gt_set.intersection(pred_set))
            recalls.append(intersection / float(k))

    return float(np.mean(recalls))


def compute_latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """
    Computes Mean, P50 (median), P95, and P99 latencies in milliseconds.
    """
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }

    return {
        "avg_latency_ms": float(np.mean(arr)),
        "p50_latency_ms": float(np.median(arr)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "min_latency_ms": float(np.min(arr)),
        "max_latency_ms": float(np.max(arr)),
    }


def compute_speedup(exact_latency_ms: float, ivf_latency_ms: float) -> float:
    """
    Computes speedup factor: exact_latency / ivf_latency.
    """
    if ivf_latency_ms <= 0:
        return 1.0
    return float(exact_latency_ms / ivf_latency_ms)

