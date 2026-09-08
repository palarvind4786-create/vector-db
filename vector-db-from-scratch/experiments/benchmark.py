"""
Rigorous benchmark comparing ExactIndex and IVF-Flat.
Evaluates Recall@10, latency percentiles (Mean, P50, P95, P99),
candidate count, percentage of dataset examined, and speedup factor.
Saves raw results to CSV and JSON, and generates 4 publication-grade plots.
Strictly ZERO external ANN or clustering libraries.
"""

import os
import sys
import csv
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.exact_index import ExactIndex
from app.core.ivf_index import IVFFlatIndex
from experiments.metrics import (
    compute_recall_at_k,
    compute_latency_percentiles,
    compute_speedup,
)


@dataclass
class BenchmarkRecord:
    name: str
    index_type: str
    nprobe: int | None
    recall_at_10: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_candidates: float
    pct_examined: float
    speedup: float


def prepare_50k_dataset(
    data_dir: str = "data",
    total_vectors: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads real embeddings (10,000 vectors) and held-out queries (500 vectors),
    and augments the real embeddings to 50,000 vectors preserving the natural
    semantic cluster geometry on the unit sphere S^(D-1).
    """
    real_emb_path = os.path.join(data_dir, "embeddings.npy")
    queries_path = os.path.join(data_dir, "queries.npy")
    cache_path = os.path.join(data_dir, f"embeddings_{total_vectors // 1000}k.npy")

    if not os.path.exists(real_emb_path):
        raise FileNotFoundError(f"Base embeddings not found at {real_emb_path}")
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries not found at {queries_path}")

    queries = np.load(queries_path).astype(np.float32)

    if os.path.exists(cache_path):
        vectors = np.load(cache_path).astype(np.float32)
        if len(vectors) == total_vectors:
            return vectors, queries

    real_vectors = np.load(real_emb_path).astype(np.float32)
    n_real, dim = real_vectors.shape

    if total_vectors <= n_real:
        vectors = real_vectors[:total_vectors].copy()
    else:
        # Augment real embeddings with realistic semantic manifold neighbors
        needed = total_vectors - n_real
        multiplier = int(np.ceil(needed / n_real))
        rng = np.random.default_rng(seed)

        augmented_list = []
        for _ in range(multiplier):
            noise = rng.normal(loc=0.0, scale=0.05, size=(n_real, dim)).astype(np.float32)
            noisy_vecs = real_vectors + noise
            noisy_vecs /= np.linalg.norm(noisy_vecs, axis=1, keepdims=True)
            augmented_list.append(noisy_vecs)

        stacked_aug = np.vstack(augmented_list)[:needed]
        vectors = np.vstack([real_vectors, stacked_aug]).astype(np.float32)

    # Unit-normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(cache_path, vectors)
    return vectors, queries


class BenchmarkRunner:
    """
    Executes the comprehensive benchmark suite between ExactIndex and IVFFlatIndex.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        queries: np.ndarray,
        nlist: int = 100,
        seed: int = 42,
        output_dir: str = "experiments/results",
    ):
        self.vectors = vectors
        self.queries = queries
        self.num_vectors = len(vectors)
        self.num_queries = len(queries)
        self.dim = vectors.shape[1]
        self.nlist = nlist
        self.seed = seed
        self.output_dir = output_dir

        self.ground_truth_ids: list[list[Any]] = []
        self.exact_record: BenchmarkRecord | None = None
        self.ivf_records: list[BenchmarkRecord] = []

    def run_exact_benchmark(self, k: int = 10, warmup: int = 10) -> BenchmarkRecord:
        """
        Builds ExactIndex, runs all queries to compute exact ground truth top-k IDs,
        and measures exact search latency profile.
        """
        print(f"\n[1/3] Building ExactIndex on {self.num_vectors:,} vectors (D={self.dim})...")
        exact_index = ExactIndex(dim=self.dim, metric="cosine")
        doc_ids = [f"doc_{i}" for i in range(self.num_vectors)]
        exact_index.add(doc_ids, self.vectors)

        # Warm-up pass
        for i in range(min(warmup, self.num_queries)):
            exact_index.search(self.queries[i], k=k)

        print(f"Running ExactIndex on {self.num_queries} queries (k={k})...")
        gt_ids = []
        latencies_ms = []

        for q in self.queries:
            t0 = time.perf_counter_ns()
            res = exact_index.search(q, k=k)
            t1 = time.perf_counter_ns()
            gt_ids.append(list(res.ids))
            latencies_ms.append((t1 - t0) / 1e6)

        self.ground_truth_ids = gt_ids
        lat_stats = compute_latency_percentiles(latencies_ms)

        record = BenchmarkRecord(
            name="ExactIndex (Brute Force)",
            index_type="Exact",
            nprobe=None,
            recall_at_10=1.0,
            avg_latency_ms=lat_stats["avg_latency_ms"],
            p50_latency_ms=lat_stats["p50_latency_ms"],
            p95_latency_ms=lat_stats["p95_latency_ms"],
            p99_latency_ms=lat_stats["p99_latency_ms"],
            avg_candidates=float(self.num_vectors),
            pct_examined=100.0,
            speedup=1.0,
        )
        self.exact_record = record
        print(
            f"  -> ExactIndex complete: Mean={record.avg_latency_ms:.3f}ms, "
            f"P50={record.p50_latency_ms:.3f}ms, P95={record.p95_latency_ms:.3f}ms"
        )
        return record

    def run_ivf_benchmark(
        self,
        nprobes: list[int] = [1, 2, 5, 10, 20, 50],
        k: int = 10,
        warmup: int = 10,
    ) -> list[BenchmarkRecord]:
        """
        Trains and populates IVFFlatIndex, then evaluates across varying nprobes.
        """
        if self.exact_record is None or len(self.ground_truth_ids) == 0:
            raise RuntimeError("Must run exact benchmark first to establish ground truth.")

        print(f"\n[2/3] Training & Building IVFFlatIndex (nlist={self.nlist}, D={self.dim})...")
        t_build_start = time.perf_counter()
        ivf_index = IVFFlatIndex(
            dim=self.dim,
            nlist=self.nlist,
            nprobe=1,
            metric="cosine",
            seed=self.seed,
        )
        doc_ids = [f"doc_{i}" for i in range(self.num_vectors)]
        ivf_index.add(doc_ids, self.vectors)
        t_build_end = time.perf_counter()
        print(f"  -> IVF index built in {t_build_end - t_build_start:.2f}s.")

        # Warm-up pass
        for i in range(min(warmup, self.num_queries)):
            ivf_index.search(self.queries[i], k=k, nprobe=1)

        records: list[BenchmarkRecord] = []
        exact_mean_lat = self.exact_record.avg_latency_ms

        print(f"\n[3/3] Evaluating IVF-Flat across nprobes: {nprobes}...")
        for probe in nprobes:
            pred_ids = []
            latencies_ms = []
            candidates_list = []

            for q in self.queries:
                t0 = time.perf_counter_ns()
                res = ivf_index.search(q, k=k, nprobe=probe)
                t1 = time.perf_counter_ns()
                pred_ids.append(list(res.ids))
                latencies_ms.append((t1 - t0) / 1e6)
                candidates_list.append(res.num_compared)

            recall = compute_recall_at_k(self.ground_truth_ids, pred_ids, k=k)
            lat_stats = compute_latency_percentiles(latencies_ms)
            avg_cands = float(np.mean(candidates_list))
            pct_examined = float((avg_cands / self.num_vectors) * 100.0)
            speedup = compute_speedup(exact_mean_lat, lat_stats["avg_latency_ms"])

            rec = BenchmarkRecord(
                name=f"IVF-Flat (nprobe={probe})",
                index_type="IVF-Flat",
                nprobe=probe,
                recall_at_10=recall,
                avg_latency_ms=lat_stats["avg_latency_ms"],
                p50_latency_ms=lat_stats["p50_latency_ms"],
                p95_latency_ms=lat_stats["p95_latency_ms"],
                p99_latency_ms=lat_stats["p99_latency_ms"],
                avg_candidates=avg_cands,
                pct_examined=pct_examined,
                speedup=speedup,
            )
            records.append(rec)
            print(
                f"  -> nprobe={probe:2d}: Recall@10={rec.recall_at_10:.4f} "
                f"({rec.recall_at_10 * 100:5.1f}%) | "
                f"Mean={rec.avg_latency_ms:6.3f}ms | "
                f"P50={rec.p50_latency_ms:6.3f}ms | "
                f"P95={rec.p95_latency_ms:6.3f}ms | "
                f"Candidates={rec.avg_candidates:6.0f} ({rec.pct_examined:4.1f}%) | "
                f"Speedup={rec.speedup:5.2f}x"
            )

        self.ivf_records = records
        return records

    def run_full_suite(
        self,
        nprobes: list[int] = [1, 2, 5, 10, 20, 50],
        k: int = 10,
    ) -> dict[str, Any]:
        """Runs ExactIndex and IVF-Flat benchmarks, outputs tables, plots, and saves results."""
        self.run_exact_benchmark(k=k)
        self.run_ivf_benchmark(nprobes=nprobes, k=k)

        all_records = [self.exact_record] + self.ivf_records
        results_payload = {
            "metadata": {
                "num_vectors": self.num_vectors,
                "num_queries": self.num_queries,
                "k": k,
                "dim": self.dim,
                "nlist": self.nlist,
                "nprobes_evaluated": nprobes,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "records": [asdict(r) for r in all_records],
        }

        self.save_results(results_payload)
        self.generate_plots()
        self.print_summary_table()
        return results_payload

    def save_results(self, payload: dict[str, Any]) -> None:
        """Saves raw benchmark results to JSON and CSV."""
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. JSON
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Saved JSON] {json_path}")

        # 2. CSV
        csv_path = os.path.join(self.output_dir, "benchmark_results.csv")
        records = payload["records"]
        fieldnames = [
            "name",
            "index_type",
            "nprobe",
            "recall_at_10",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "avg_candidates",
            "pct_examined",
            "speedup",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"[Saved CSV]  {csv_path}")

        # 3. Ground truth JSON
        gt_path = os.path.join(self.output_dir, "ground_truth_50k.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_queries": self.num_queries,
                    "k": len(self.ground_truth_ids[0]) if self.ground_truth_ids else 0,
                    "ground_truth_ids": self.ground_truth_ids,
                },
                f,
                indent=2,
            )
        print(f"[Saved GT]   {gt_path}")

        # Also mirror to top-level results/ directory
        if os.path.normpath(self.output_dir) != os.path.normpath("results"):
            top_results = "results"
            os.makedirs(top_results, exist_ok=True)
            with open(os.path.join(top_results, "benchmark_results.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            with open(os.path.join(top_results, "benchmark_results.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    writer.writerow(r)
            print(f"[Saved Mirror] {top_results}/benchmark_results.json & CSV")

    def generate_plots(self) -> None:
        """
        Generates 4 publication-grade figures illustrating IVF vs Exact performance:
        1. Recall vs nprobe
        2. Latency vs nprobe (P50, Mean, P95, P99)
        3. Recall vs Latency (Pareto frontier)
        4. Percentage of vectors examined vs Recall
        """
        os.makedirs(self.output_dir, exist_ok=True)

        probes = [r.nprobe for r in self.ivf_records]
        recalls = [r.recall_at_10 for r in self.ivf_records]
        means = [r.avg_latency_ms for r in self.ivf_records]
        p50s = [r.p50_latency_ms for r in self.ivf_records]
        p95s = [r.p95_latency_ms for r in self.ivf_records]
        p99s = [r.p99_latency_ms for r in self.ivf_records]
        pcts = [r.pct_examined for r in self.ivf_records]

        exact_mean = self.exact_record.avg_latency_ms if self.exact_record else 0.0

        # Plot 1: Recall vs nprobe
        plt.figure(figsize=(8, 5.2), dpi=150)
        plt.plot(
            probes,
            recalls,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color="#2563eb",
            label="IVF-Flat Recall@10",
        )
        plt.axhline(
            1.0,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label="ExactIndex Reference (1.000)",
        )
        for p, rec in zip(probes, recalls):
            plt.annotate(
                f"{rec * 100:.1f}%",
                (p, rec),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#1e3a8a",
            )
        plt.title(
            f"Recall@10 vs nprobe (50,000 Vectors, nlist={self.nlist})",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("nprobe (Voronoi Centroids Probed)", fontsize=11)
        plt.ylabel("Recall@10", fontsize=11)
        plt.ylim(min(recalls) * 0.85, 1.06)
        plt.xticks(probes)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()
        p1_path = os.path.join(self.output_dir, "recall_vs_nprobe.png")
        plt.savefig(p1_path)
        plt.close()
        print(f"[Plot 1] {p1_path}")

        # Plot 2: Latency vs nprobe
        plt.figure(figsize=(8.5, 5.5), dpi=150)
        plt.plot(
            probes,
            means,
            marker="o",
            linewidth=2.2,
            color="#2563eb",
            label="Mean Latency",
        )
        plt.plot(
            probes,
            p50s,
            marker="s",
            linewidth=1.8,
            linestyle="-.",
            color="#059669",
            label="P50 (Median)",
        )
        plt.plot(
            probes,
            p95s,
            marker="^",
            linewidth=1.8,
            linestyle="--",
            color="#d97706",
            label="P95 Latency",
        )
        plt.plot(
            probes,
            p99s,
            marker="d",
            linewidth=1.8,
            linestyle=":",
            color="#7c3aed",
            label="P99 Latency",
        )
        plt.axhline(
            exact_mean,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            label=f"ExactIndex Mean ({exact_mean:.2f}ms)",
        )
        plt.title(
            f"Search Latency Percentiles vs nprobe (50,000 Vectors)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("nprobe (Voronoi Centroids Probed)", fontsize=11)
        plt.ylabel("Query Latency (ms)", fontsize=11)
        plt.xticks(probes)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()
        p2_path = os.path.join(self.output_dir, "latency_vs_nprobe.png")
        plt.savefig(p2_path)
        plt.close()
        print(f"[Plot 2] {p2_path}")

        # Plot 3: Recall vs Latency (Pareto Frontier)
        plt.figure(figsize=(8.5, 5.5), dpi=150)
        plt.plot(
            means,
            recalls,
            marker="o",
            markersize=8,
            linewidth=2.2,
            color="#2563eb",
            label="IVF-Flat Frontier",
        )
        plt.scatter(
            [exact_mean],
            [1.0],
            color="#dc2626",
            marker="*",
            s=220,
            zorder=5,
            label=f"ExactIndex (Mean={exact_mean:.2f}ms, Recall=1.0)",
        )
        for p, m, rec in zip(probes, means, recalls):
            plt.annotate(
                f"nprobe={p}\n({m:.2f}ms, {rec * 100:.1f}%)",
                (m, rec),
                textcoords="offset points",
                xytext=(8, -12),
                fontsize=8.5,
                color="#1e3a8a",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f1f5f9", ec="#94a3b8", alpha=0.8),
            )
        plt.title(
            "Recall@10 vs Latency (Trade-Off Curve)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("Average Query Latency (ms)", fontsize=11)
        plt.ylabel("Recall@10", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()
        p3_path = os.path.join(self.output_dir, "recall_vs_latency.png")
        plt.savefig(p3_path)
        plt.close()
        print(f"[Plot 3] {p3_path}")

        # Plot 4: Percentage of vectors examined vs recall
        plt.figure(figsize=(8.5, 5.2), dpi=150)
        plt.plot(
            pcts,
            recalls,
            marker="s",
            markersize=8,
            linewidth=2.5,
            color="#059669",
            label="IVF-Flat Examination Curve",
        )
        plt.scatter(
            [100.0],
            [1.0],
            color="#dc2626",
            marker="*",
            s=200,
            zorder=5,
            label="Exact (100% Examined, Recall=1.0)",
        )
        for p, pct, rec in zip(probes, pcts, recalls):
            plt.annotate(
                f"nprobe={p}\n{pct:.1f}% data",
                (pct, rec),
                textcoords="offset points",
                xytext=(-10, 10),
                ha="center",
                fontsize=8.5,
                color="#065f46",
                fontweight="bold",
            )
        plt.title(
            "Recall@10 vs Percentage of Vectors Examined",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        plt.xlabel("Percentage of Dataset Examined (%)", fontsize=11)
        plt.ylabel("Recall@10", fontsize=11)
        plt.xlim(0, max(pcts) * 1.15)
        plt.ylim(min(recalls) * 0.85, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()
        p4_path = os.path.join(self.output_dir, "vectors_examined_vs_recall.png")
        plt.savefig(p4_path)
        plt.close()
        print(f"[Plot 4] {p4_path}")

    def print_summary_table(self) -> None:
        """Prints formatted summary table to console."""
        all_records = [self.exact_record] + self.ivf_records

        header = (
            f"| {'Index Configuration':<24} | {'nprobe':<6} | {'Recall@10':<9} | "
            f"{'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P95 (ms)':<8} | {'P99 (ms)':<8} | "
            f"{'Candidates':<10} | {'Examined':<8} | {'Speedup':<7} |"
        )
        sep = "|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|"

        print("\n" + "=" * len(header))
        print(f"BENCHMARK RESULTS: ExactIndex vs IVF-Flat ({self.num_vectors:,} Vectors, {self.num_queries} Queries, k=10)")
        print("=" * len(header))
        print(header)
        print(sep)

        for r in all_records:
            nprobe_str = str(r.nprobe) if r.nprobe is not None else "-"
            pct_str = f"{r.pct_examined:5.1f}%"
            cands_str = f"{r.avg_candidates:8.0f}"
            print(
                f"| {r.name:<24} | {nprobe_str:<6} | {r.recall_at_10:<9.4f} | "
                f"{r.avg_latency_ms:<9.3f} | {r.p50_latency_ms:<8.3f} | "
                f"{r.p95_latency_ms:<8.3f} | {r.p99_latency_ms:<8.3f} | "
                f"{cands_str:<10} | {pct_str:<8} | {r.speedup:<6.2f}x |"
            )

        print("=" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector DB from Scratch - Benchmark Suite")
    parser.add_argument("--num_vectors", type=int, default=50000, help="Number of vectors (e.g. 10000 or 50000)")
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50], help="List of nprobe values to test")
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff")
    parser.add_argument("--nlist", type=int, default=100, help="Number of Voronoi clusters (K-Means)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Directory to save results")
    args = parser.parse_args()

    print(f"Loading / preparing {args.num_vectors:,} vector dataset...")
    vectors, queries = prepare_50k_dataset(total_vectors=args.num_vectors, seed=args.seed)
    print(f"Vectors: shape={vectors.shape}, dtype={vectors.dtype}")
    print(f"Queries: shape={queries.shape}, dtype={queries.dtype}")

    runner = BenchmarkRunner(
        vectors=vectors,
        queries=queries,
        nlist=args.nlist,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    runner.run_full_suite(nprobes=args.nprobes, k=args.k)
