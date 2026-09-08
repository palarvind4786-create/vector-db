"""
Unit tests for vector distance metrics, normalization, cosine similarity,
batch operations, zero-vector handling, and top-k selection.
"""

import numpy as np
import pytest
from app.core.similarity import (
    normalize_vectors,
    cosine_similarity_pairwise,
    cosine_similarity,
    cosine_similarity_batch,
    l2_distance,
    top_k,
)


class TestVectorNormalization:
    """Tests for normalize_vectors."""

    def test_1d_vector_normalization(self):
        v = np.array([3.0, 4.0])
        normed = normalize_vectors(v)
        expected = np.array([0.6, 0.8])
        assert np.allclose(normed, expected, atol=1e-7)
        assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-7)

    def test_2d_matrix_normalization(self):
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        normed = normalize_vectors(matrix, axis=1)
        row_norms = np.linalg.norm(normed, axis=1)
        assert np.allclose(row_norms, np.ones(3), atol=1e-7)

    def test_zero_vector_safety(self):
        zero_v = np.zeros(5)
        normed = normalize_vectors(zero_v)
        assert not np.isnan(normed).any()
        assert not np.isinf(normed).any()
        assert np.all(normed == 0.0)

    def test_mixed_zero_and_nonzero_rows(self):
        matrix = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        normed = normalize_vectors(matrix, axis=1)
        assert np.all(normed[0] == 0.0)
        assert np.allclose(normed[1], [1.0, 0.0, 0.0])
        assert np.all(normed[2] == 0.0)
        assert np.allclose(normed[3], [0.0, 0.0, 1.0])

    def test_near_zero_epsilon(self):
        near_zero = np.array([1e-12, 1e-12])
        normed = normalize_vectors(near_zero, eps=1e-10)
        assert np.all(normed == 0.0)


class TestCosineSimilarityPairwise:
    """Tests for pairwise cosine similarity."""

    def test_identical_vectors(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 2.0, 3.0])
        assert np.isclose(cosine_similarity_pairwise(u, v), 1.0)

    def test_collinear_scaled_vectors(self):
        u = np.array([1.0, -2.0, 3.0])
        v = 5.0 * u
        assert np.isclose(cosine_similarity_pairwise(u, v), 1.0)

    def test_opposite_vectors(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([-1.0, -2.0, -3.0])
        assert np.isclose(cosine_similarity_pairwise(u, v), -1.0)

    def test_orthogonal_vectors(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert np.isclose(cosine_similarity_pairwise(u, v), 0.0)

    def test_zero_vector_handling(self):
        u = np.zeros(4)
        v = np.array([1.0, 2.0, 3.0, 4.0])
        assert cosine_similarity_pairwise(u, v) == 0.0
        assert cosine_similarity_pairwise(v, u) == 0.0
        assert cosine_similarity_pairwise(u, u) == 0.0

    def test_dimension_mismatch(self):
        u = np.array([1.0, 2.0])
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cosine_similarity_pairwise(u, v)


class TestCosineSimilarityVectorMatrix:
    """Tests for 1 query vs N dataset vectors."""

    def test_single_query_against_matrix(self):
        q = np.array([1.0, 0.0])
        X = np.array([
            [1.0, 0.0],    # angle 0 -> 1.0
            [0.0, 1.0],    # angle 90 -> 0.0
            [-1.0, 0.0],   # angle 180 -> -1.0
            [1.0, 1.0],    # angle 45 -> 1/sqrt(2) ~ 0.7071
        ])
        scores = cosine_similarity(q, X)
        expected = np.array([1.0, 0.0, -1.0, 1.0 / np.sqrt(2.0)])
        assert np.allclose(scores, expected, atol=1e-7)

    def test_empty_vectors_matrix(self):
        q = np.array([1.0, 2.0])
        X = np.empty((0, 2))
        scores = cosine_similarity(q, X)
        assert scores.shape == (0,)

    def test_zero_query_vector(self):
        q = np.zeros(3)
        X = np.random.randn(10, 3)
        scores = cosine_similarity(q, X)
        assert np.all(scores == 0.0)
        assert not np.isnan(scores).any()

    def test_matrix_with_zero_vector_rows(self):
        q = np.array([1.0, 1.0, 1.0])
        X = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0],
        ])
        scores = cosine_similarity(q, X)
        assert np.isclose(scores[0], 1.0)
        assert np.isclose(scores[1], 0.0)
        assert np.isclose(scores[2], -1.0)

    def test_dimension_mismatch_raises(self):
        q = np.array([1.0, 2.0])
        X = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            cosine_similarity(q, X)


class TestCosineSimilarityBatch:
    """Tests for batch queries (M x D) vs vectors (N x D)."""

    def test_batch_against_matrix(self):
        Q = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        X = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        S = cosine_similarity_batch(Q, X)
        expected = np.array([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        assert np.allclose(S, expected, atol=1e-7)

    def test_batch_matches_individual_queries(self):
        np.random.seed(42)
        Q = np.random.randn(5, 8)
        X = np.random.randn(20, 8)
        batch_scores = cosine_similarity_batch(Q, X)
        for i in range(len(Q)):
            single_scores = cosine_similarity(Q[i], X)
            assert np.allclose(batch_scores[i], single_scores, atol=1e-7)

    def test_empty_batch(self):
        Q = np.empty((0, 4))
        X = np.random.randn(5, 4)
        assert cosine_similarity_batch(Q, X).shape == (0, 5)

        Q2 = np.random.randn(3, 4)
        X2 = np.empty((0, 4))
        assert cosine_similarity_batch(Q2, X2).shape == (3, 0)


class TestL2Distance:
    """Tests for Euclidean distance."""

    def test_l2_distance_calculation(self):
        q = np.array([0.0, 0.0])
        X = np.array([
            [3.0, 4.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ])
        dists = l2_distance(q, X)
        expected = np.array([5.0, np.sqrt(2.0), 0.0])
        assert np.allclose(dists, expected, atol=1e-7)

    def test_empty_vectors(self):
        q = np.array([1.0, 2.0])
        X = np.empty((0, 2))
        assert l2_distance(q, X).shape == (0,)


class TestTopK:
    """Tests for top-k selection."""

    def test_top_k_largest(self):
        scores = np.array([0.1, 0.9, 0.4, 0.7, 0.3])
        indices, top_scores = top_k(scores, k=3, largest=True)
        assert list(indices) == [1, 3, 2]
        assert np.allclose(top_scores, [0.9, 0.7, 0.4])

    def test_top_k_smallest(self):
        scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        indices, top_scores = top_k(scores, k=2, largest=False)
        assert list(indices) == [1, 3]
        assert np.allclose(top_scores, [1.0, 2.0])

    def test_top_k_greater_than_n(self):
        scores = np.array([0.2, 0.8])
        indices, top_scores = top_k(scores, k=5, largest=True)
        assert list(indices) == [1, 0]
        assert np.allclose(top_scores, [0.8, 0.2])

    def test_top_k_k_equals_1(self):
        scores = np.array([-0.5, 0.2, 0.9, -0.1])
        indices, top_scores = top_k(scores, k=1, largest=True)
        assert list(indices) == [2]
        assert np.allclose(top_scores, [0.9])

    def test_top_k_empty_scores(self):
        scores = np.array([])
        indices, top_scores = top_k(scores, k=5)
        assert len(indices) == 0
        assert len(top_scores) == 0

