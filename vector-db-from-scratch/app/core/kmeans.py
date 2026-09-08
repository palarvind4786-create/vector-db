"""
K-Means clustering from scratch using NumPy.
Includes K-Means++ initialization, vectorized Lloyd's updates,
and empty cluster handling.
Strictly ZERO sklearn.cluster.KMeans or external clustering libraries.
"""

from typing import Any
import numpy as np
from app.core.similarity import normalize_vectors


class KMeans:
    """
    Custom K-Means clustering implementation using pure NumPy.

    Features:
    - K-Means++ probabilistic initialization (D^2 weighting) and standard random init.
    - Vectorized Lloyd's algorithm using matrix distance expansion (||X - C||^2).
    - Automatic zero-member cluster recovery via worst-represented outlier reassignment.
    - Full convergence tracking: inertia curve, centroid shift, iteration counts.
    - Support for Euclidean and Cosine (spherical) clustering.
    """

    def __init__(
        self,
        n_clusters: int = 100,
        max_iter: int = 50,
        tol: float = 1e-4,
        init: str = "k-means++",
        metric: str = "euclidean",
        seed: int | None = 42,
    ):
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}.")
        if init not in ("k-means++", "random"):
            raise ValueError(f"Unknown init method '{init}'. Choose 'k-means++' or 'random'.")
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"Unsupported metric '{metric}'. Choose 'euclidean' or 'cosine'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.metric = metric
        self.seed = seed

        # Fitted attributes
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.history_: list[dict[str, Any]] = []

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize K centroids using K-Means++ or uniform random sampling."""
        n_samples, dim = X.shape
        rng = np.random.default_rng(self.seed)

        if self.init == "random":
            chosen_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[chosen_indices].copy()

        # K-Means++ Initialization
        centroids = np.empty((self.n_clusters, dim), dtype=X.dtype)
        # 1. Pick first centroid uniformly at random
        first_idx = rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        # 2. Maintain running minimum squared distances to chosen centroids
        min_sq_dists = np.sum((X - centroids[0]) ** 2, axis=1)

        # 3. Choose remaining K-1 centroids proportional to D(x)^2
        for k in range(1, self.n_clusters):
            total_dist = np.sum(min_sq_dists)
            if total_dist <= 1e-12:
                # All points identical or already matched; pick uniformly
                next_idx = rng.integers(0, n_samples)
            else:
                probs = min_sq_dists / total_dist
                # Ensure valid probability distribution (guard against floating drift)
                probs = np.clip(probs, 0.0, 1.0)
                probs /= np.sum(probs)
                next_idx = rng.choice(n_samples, p=probs)

            centroids[k] = X[next_idx]
            # Update min squared distances with new centroid
            new_sq_dists = np.sum((X - centroids[k]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, new_sq_dists)

        return centroids

    def _compute_squared_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of squared Euclidean distances between
        dataset X of shape (N, D) and centroids C of shape (K, D).

        Mathematical identity:
            ||X_i - C_j||^2 = ||X_i||^2 - 2 <X_i, C_j> + ||C_j||^2
        """
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)            # (N, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        dists = X_sq - 2.0 * np.dot(X, centroids.T) + C_sq      # (N, K)
        # Clip numerical precision artifacts (< 0.0)
        return np.maximum(dists, 0.0)

    def _assign_clusters(
        self, X: np.ndarray, centroids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Assign each vector in X to its nearest centroid.
        Returns (labels, min_squared_distances, total_inertia).
        """
        sq_dists = self._compute_squared_distances(X, centroids)
        labels = np.argmin(sq_dists, axis=1)
        min_dists = sq_dists[np.arange(len(X)), labels]
        inertia = float(np.sum(min_dists))
        return labels, min_dists, inertia

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        min_dists: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """
        Recalculate cluster centroids as mean of assigned vectors.
        Handles empty clusters by reassigning to the point with maximum residual distance.
        """
        n_samples, dim = X.shape
        new_centroids = np.zeros((self.n_clusters, dim), dtype=X.dtype)
        empty_count = 0

        # Working copy of residuals for assigning multiple empty clusters
        residuals = min_dists.copy()

        for j in range(self.n_clusters):
            mask = labels == j
            cluster_size = np.count_nonzero(mask)

            if cluster_size > 0:
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                # Empty cluster handling: assign to the furthest outlier
                empty_count += 1
                outlier_idx = int(np.argmax(residuals))
                new_centroids[j] = X[outlier_idx].copy()
                residuals[outlier_idx] = 0.0  # Prevent re-picking same point

        # For cosine clustering, re-normalize centroids to unit sphere
        if self.metric == "cosine":
            new_centroids = normalize_vectors(new_centroids, axis=1)

        return new_centroids, empty_count

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means clustering on vectors X using Lloyd's algorithm.

        Parameters:
            X: 2D NumPy array of shape (N, D).

        Returns:
            self
        """
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (N, D), got {arr.shape}.")

        n_samples, dim = arr.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})."
            )

        if self.metric == "cosine":
            arr = normalize_vectors(arr, axis=1)

        # 1. Initialize centroids
        centroids = self._init_centroids(arr)
        if self.metric == "cosine":
            centroids = normalize_vectors(centroids, axis=1)

        self.history_.clear()
        self.converged_ = False

        # 2. Lloyd's iteration loop
        for it in range(self.max_iter):
            # Assignment step
            labels, min_dists, inertia = self._assign_clusters(arr, centroids)

            # Update step with empty cluster recovery
            new_centroids, empty_count = self._update_centroids(arr, labels, min_dists)

            # Measure maximum centroid shift
            shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))

            self.history_.append({
                "iteration": it + 1,
                "inertia": inertia,
                "max_shift": shift,
                "empty_clusters": empty_count,
            })

            centroids = new_centroids

            # Check convergence
            if shift < self.tol:
                self.converged_ = True
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Final assignment pass
        final_labels, _, final_inertia = self._assign_clusters(arr, centroids)

        self.centroids = centroids
        self.labels_ = final_labels
        self.inertia_ = final_inertia

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new vectors X to nearest fitted cluster centroids.

        Parameters:
            X: 2D NumPy array of shape (M, D).

        Returns:
            1D integer array of shape (M,) with cluster labels in [0, K-1].
        """
        if self.centroids is None:
            raise RuntimeError("KMeans instance is not fitted yet. Call fit() before predict().")

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Dimension mismatch: expected (*, {self.centroids.shape[1]}), got {arr.shape}."
            )

        if self.metric == "cosine":
            arr = normalize_vectors(arr, axis=1)

        labels, _, _ = self._assign_clusters(arr, self.centroids)
        return labels

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance from each vector in X to all K centroids.
        Returns 2D array of shape (M, K).
        """
        if self.centroids is None:
            raise RuntimeError("KMeans instance is not fitted yet.")
        arr = np.asarray(X, dtype=np.float32)
        if self.metric == "cosine":
            arr = normalize_vectors(arr, axis=1)
        sq_dists = self._compute_squared_distances(arr, self.centroids)
        return np.sqrt(sq_dists)

