"""
Comprehensive unit tests for IVFFlatIndex.
Compares IVF approximate nearest neighbor search directly against ExactIndex ground truth,
and verifies lazy deletion, nprobe tradeoff, rebuild compaction, and cluster metrics.
"""

import numpy as np
import pytest
from app.core.exact_index import ExactIndex
from app.core.ivf_index import IVFFlatIndex


class TestIVFBasics:
    """Tests for initialization, training, insertion, and stats."""

    def test_empty_ivf_search(self):
        ivf = IVFFlatIndex(dim=4, nlist=5, nprobe=2)
        res = ivf.search(np.array([1.0, 0.0, 0.0, 0.0]), k=5)
        assert len(res) == 0
        assert res.num_compared == 0
        assert res.num_candidates == 0

    def test_auto_train_on_add(self):
        ivf = IVFFlatIndex(dim=3, nlist=3, nprobe=2, seed=42)
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.1, 0.0],
        ], dtype=np.float32)
        ids = ["v1", "v2", "v3", "v4"]
        ivf.add(ids, vectors)

        assert ivf.is_trained
        assert ivf.centroids is not None
        assert ivf.centroids.shape == (3, 3)
        assert ivf.size() == 4

        stats = ivf.get_cluster_stats()
        assert stats["nlist"] == 3
        assert stats["active_vectors"] == 4
        assert stats["total_vectors"] == 4


class TestIVFVsExactGroundTruth:
    """Tests comparing IVF search results against ExactIndex."""

    def test_ivf_high_nprobe_matches_exact_index(self):
        """When nprobe == nlist, IVF scans all posting lists and must equal ExactIndex."""
        np.random.seed(42)
        n_vectors = 200
        dim = 16
        k = 10
        nlist = 10

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        ids = [f"doc_{i}" for i in range(n_vectors)]

        # 1. ExactIndex Ground Truth
        exact = ExactIndex(dim=dim)
        exact.add(ids, vectors)

        # 2. IVF-Flat Index
        ivf = IVFFlatIndex(dim=dim, nlist=nlist, nprobe=nlist, seed=42)
        ivf.add(ids, vectors)

        # Query with 5 different random vectors
        for _ in range(5):
            q = np.random.randn(dim).astype(np.float32)
            exact_res = exact.search(q, k=k)
            ivf_res = ivf.search(q, k=k, nprobe=nlist)

            assert len(ivf_res) == k
            # With nprobe == nlist, all active vectors are compared
            assert ivf_res.num_compared == n_vectors
            # Result IDs must match exactly
            assert ivf_res.ids == exact_res.ids
            assert np.allclose(ivf_res.scores, exact_res.scores, atol=1e-5)

    def test_nprobe_tradeoff_scans_fewer_vectors(self):
        """Probing fewer clusters examines strictly fewer vectors."""
        np.random.seed(123)
        vectors = np.random.randn(300, 8).astype(np.float32)
        ids = [f"id_{i}" for i in range(300)]

        ivf = IVFFlatIndex(dim=8, nlist=15, nprobe=2, seed=42)
        ivf.add(ids, vectors)

        q = np.random.randn(8).astype(np.float32)

        res_p1 = ivf.search(q, k=5, nprobe=1)
        res_p5 = ivf.search(q, k=5, nprobe=5)
        res_p15 = ivf.search(q, k=5, nprobe=15)

        assert res_p1.num_compared < res_p5.num_compared < res_p15.num_compared
        assert res_p15.num_compared == 300


class TestIVFDeletionAndRebuild:
    """Tests verifying lazy deletion and index compaction."""

    def test_lazy_deletion_excludes_vectors(self):
        ivf = IVFFlatIndex(dim=3, nlist=2, nprobe=2, seed=42)
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        ivf.add(["target", "second", "other"], vectors)

        # Before deletion, query finds target
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        res1 = ivf.search(q, k=2)
        assert res1.ids[0] == "target"

        # Delete target
        deleted_count = ivf.delete("target")
        assert deleted_count == 1
        assert ivf.size() == 2
        assert ivf.total_size() == 3

        # Search again: target MUST NOT appear in results
        res2 = ivf.search(q, k=2)
        assert "target" not in res2.ids
        assert res2.ids[0] == "second"
        # Verify that candidate counting tracked the tombstone
        assert res2.num_candidates == 3
        assert res2.num_compared == 2

    def test_rebuild_purges_tombstones(self):
        ivf = IVFFlatIndex(dim=2, nlist=2, nprobe=2, seed=42)
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        ivf.add(["a", "b", "c"], vectors)

        ivf.delete("b")
        assert ivf.size() == 2
        assert ivf.total_size() == 3

        # Rebuild without retraining kmeans
        ivf.rebuild(retrain_kmeans=False)
        assert ivf.size() == 2
        assert ivf.total_size() == 2
        assert len(ivf.deleted_ids) == 0

        # Verify search
        res = ivf.search(np.array([0.0, 1.0], dtype=np.float32), k=2)
        assert "b" not in res.ids
        assert len(res) == 2

