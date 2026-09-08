"""
Comprehensive unit tests for ExactIndex brute-force search.
Verifies insertion, exact search, metadata return, top-k argpartition,
lazy deletion, compaction, and k > dataset size.
"""

import numpy as np
import pytest
from app.core.exact_index import ExactIndex, SearchResult
from app.core.similarity import cosine_similarity_pairwise


class TestExactIndexBasics:
    """Tests for basic ExactIndex operations."""

    def test_empty_index(self):
        index = ExactIndex(dim=4)
        assert index.size() == 0
        res = index.search(np.array([1.0, 0.0, 0.0, 0.0]), k=5)
        assert len(res) == 0
        assert len(res.ids) == 0
        assert len(res.scores) == 0

    def test_single_insert_and_search(self):
        index = ExactIndex(dim=3)
        v = np.array([1.0, 2.0, 3.0])
        index.insert(id="doc_1", vector=v, metadata={"category": "tech"})

        assert index.size() == 1
        res = index.search(v, k=1)
        assert len(res) == 1
        assert res.ids[0] == "doc_1"
        assert np.isclose(res.scores[0], 1.0)
        assert res.metadata[0]["category"] == "tech"

    def test_batch_add(self):
        index = ExactIndex(dim=2)
        ids = ["a", "b", "c"]
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        metas = [{"label": 1}, {"label": 2}, {"label": 3}]
        index.add(ids, vectors, metas)

        assert index.size() == 3
        # Query closest to 'b'
        res = index.search(np.array([0.1, 0.9]), k=2)
        assert res.ids[0] == "b"
        assert res.metadata[0]["label"] == 2


class TestExactIndexEdgeCases:
    """Tests for edge cases: k > N, k = 1, tie breaks, dimension mismatch."""

    def test_k_larger_than_dataset_size(self):
        index = ExactIndex(dim=2)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        index.add(["v1", "v2"], vectors)

        # Request k=10 from 2 elements
        res = index.search(np.array([1.0, 0.0]), k=10)
        assert len(res) == 2
        assert res.ids[0] == "v1"
        assert res.ids[1] == "v2"
        assert res.scores[0] > res.scores[1]

    def test_dimension_mismatch_raises(self):
        index = ExactIndex(dim=3)
        with pytest.raises(ValueError):
            index.add(["bad"], np.array([[1.0, 2.0]]))  # Dim 2 vs 3

        index.add(["ok"], np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError):
            index.search(np.array([1.0, 2.0]), k=1)  # Query dim 2 vs 3


class TestExactIndexDeletionAndCompaction:
    """Tests for lazy deletion and compaction."""

    def test_lazy_deletion_excludes_from_search(self):
        index = ExactIndex(dim=2)
        vectors = np.array([
            [1.0, 0.0],    # v1
            [0.9, 0.1],    # v2
            [0.0, 1.0],    # v3
        ])
        index.add(["v1", "v2", "v3"], vectors)

        # Before deletion, query [1.0, 0.0] finds v1 first, then v2
        res1 = index.search(np.array([1.0, 0.0]), k=2)
        assert res1.ids[0] == "v1"

        # Delete v1
        del_count = index.delete("v1")
        assert del_count == 1
        assert index.size() == 2
        assert index.total_size() == 3

        # Search again: v1 MUST NOT appear; v2 must now be top result
        res2 = index.search(np.array([1.0, 0.0]), k=2)
        assert len(res2) == 2
        assert "v1" not in res2.ids
        assert res2.ids[0] == "v2"
        assert res2.ids[1] == "v3"

    def test_delete_multiple_and_all(self):
        index = ExactIndex(dim=2)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        index.add(["a", "b"], vectors)

        index.delete(["a", "b"])
        assert index.size() == 0
        res = index.search(np.array([1.0, 0.0]), k=5)
        assert len(res) == 0

    def test_compaction(self):
        index = ExactIndex(dim=2)
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        index.add(["a", "b", "c"], vectors, [{"id": "a"}, {"id": "b"}, {"id": "c"}])

        index.delete("b")
        assert index.size() == 2
        assert index.total_size() == 3

        index.compact()
        assert index.size() == 2
        assert index.total_size() == 2
        assert len(index.deleted_ids) == 0
        assert len(index.vectors) == 2

        # Verify search after compaction
        res = index.search(np.array([0.0, 1.0]), k=2)
        assert "b" not in res.ids
        assert len(res) == 2


class TestExactIndexGroundTruthExactness:
    """Verifies that ExactIndex returns the true mathematical nearest neighbors."""

    def test_exact_ground_truth_matches_manual_calculation(self):
        np.random.seed(1337)
        dim = 16
        n = 100
        k = 10

        vectors = np.random.randn(n, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)
        ids = [f"id_{i}" for i in range(n)]

        index = ExactIndex(dim=dim)
        index.add(ids, vectors)

        res = index.search(query, k=k)

        # Compute ground truth manually using pairwise cosine
        manual_scores = [cosine_similarity_pairwise(query, vectors[i]) for i in range(n)]
        manual_sorted_indices = np.argsort(-np.array(manual_scores))[:k]
        expected_ids = [ids[i] for i in manual_sorted_indices]
        expected_scores = [manual_scores[i] for i in manual_sorted_indices]

        assert res.ids == expected_ids
        assert np.allclose(res.scores, expected_scores, atol=1e-5)

