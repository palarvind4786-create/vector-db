"""
Comprehensive unit tests for custom NumPy KMeans implementation.
Verifies convergence on known synthetic cluster structures,
K-Means++ initialization, empty cluster recovery, and parameter configurations.
"""

import numpy as np
import pytest
from app.core.kmeans import KMeans


def generate_synthetic_clusters(
    centers: list[list[float]],
    points_per_cluster: int = 100,
    cluster_std: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper to generate synthetic Gaussian clusters with known centers and true labels.
    """
    rng = np.random.default_rng(seed)
    X_parts = []
    y_parts = []
    for label, center in enumerate(centers):
        pts = rng.normal(loc=center, scale=cluster_std, size=(points_per_cluster, len(center)))
        X_parts.append(pts)
        y_parts.append(np.full(points_per_cluster, label))
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    return X, y


class TestKMeansSyntheticData:
    """Tests evaluating KMeans on known ground truth clusters."""

    def test_recovers_known_3_cluster_structure(self):
        # 3 well-separated 2D clusters
        centers = [[-5.0, -5.0], [0.0, 5.0], [5.0, -5.0]]
        X, true_labels = generate_synthetic_clusters(centers, points_per_cluster=150, cluster_std=0.3, seed=123)

        kmeans = KMeans(n_clusters=3, max_iter=30, init="k-means++", seed=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None
        assert kmeans.centroids.shape == (3, 2)
        assert kmeans.converged_
        assert kmeans.n_iter_ < 30
        assert kmeans.inertia_ > 0.0

        # Verify that each fitted centroid is close to one of the true centers
        fitted_centers = kmeans.centroids
        matched_centers = 0
        for true_c in centers:
            dists = np.linalg.norm(fitted_centers - np.array(true_c), axis=1)
            if np.min(dists) < 0.5:
                matched_centers += 1
        assert matched_centers == 3, "Failed to locate all 3 true cluster centers"

    def test_prediction_on_unseen_points(self):
        centers = [[10.0, 0.0], [0.0, 10.0]]
        X, _ = generate_synthetic_clusters(centers, points_per_cluster=100, cluster_std=0.5, seed=42)
        kmeans = KMeans(n_clusters=2, seed=42).fit(X)

        # Unseen test points near each center
        test_pts = np.array([[10.1, -0.1], [0.1, 9.9]], dtype=np.float32)
        preds = kmeans.predict(test_pts)
        # The two points should be assigned to distinct clusters
        assert preds[0] != preds[1]


class TestKMeansInitializationAndConvergence:
    """Tests for initialization methods, convergence metrics, and iteration caps."""

    def test_kmeans_plus_plus_distinct_centroids(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 10)).astype(np.float32)

        kmeans = KMeans(n_clusters=5, init="k-means++", seed=42)
        centroids = kmeans._init_centroids(X)

        assert centroids.shape == (5, 10)
        # Centroids should all be distinct
        for i in range(5):
            for j in range(i + 1, 5):
                assert not np.allclose(centroids[i], centroids[j])

    def test_max_iter_respected(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 4)).astype(np.float32)

        # Force stop at 2 iterations with very tight tolerance
        kmeans = KMeans(n_clusters=4, max_iter=2, tol=1e-12, seed=42)
        kmeans.fit(X)

        assert kmeans.n_iter_ == 2
        assert len(kmeans.history_) == 2

    def test_history_logging(self):
        centers = [[-3.0, 0.0], [3.0, 0.0]]
        X, _ = generate_synthetic_clusters(centers, points_per_cluster=50, seed=42)
        kmeans = KMeans(n_clusters=2, seed=42).fit(X)

        assert len(kmeans.history_) > 0
        first_step = kmeans.history_[0]
        assert "iteration" in first_step
        assert "inertia" in first_step
        assert "max_shift" in first_step
        assert "empty_clusters" in first_step


class TestKMeansEdgeCases:
    """Tests for empty cluster handling, input validation, and cosine metric."""

    def test_empty_cluster_recovery(self):
        # Create a dataset where points are heavily concentrated in only 2 groups,
        # but request K = 10 clusters (prone to empty clusters)
        rng = np.random.default_rng(99)
        c1 = rng.normal(loc=[10.0, 10.0], scale=0.1, size=(50, 2))
        c2 = rng.normal(loc=[-10.0, -10.0], scale=0.1, size=(50, 2))
        X = np.vstack([c1, c2]).astype(np.float32)

        kmeans = KMeans(n_clusters=10, max_iter=15, seed=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None
        assert not np.isnan(kmeans.centroids).any()
        assert not np.isinf(kmeans.centroids).any()
        # All 10 clusters should be non-empty after recovery
        unique_labels = np.unique(kmeans.labels_)
        assert len(unique_labels) == 10

    def test_cosine_spherical_clustering(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 8)).astype(np.float32)

        kmeans = KMeans(n_clusters=4, metric="cosine", seed=42)
        kmeans.fit(X)

        assert kmeans.centroids is not None
        # Centroids should be unit-normalized
        norms = np.linalg.norm(kmeans.centroids, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)
        with pytest.raises(ValueError):
            KMeans(max_iter=-1)
        with pytest.raises(ValueError):
            KMeans(init="invalid_init")
        with pytest.raises(ValueError):
            KMeans(metric="manhattan")

        # More clusters than data samples
        with pytest.raises(ValueError):
            KMeans(n_clusters=10).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

