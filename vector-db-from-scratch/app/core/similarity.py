"""
Vector distance metrics, normalization, and similarity computations from scratch using NumPy.
Strictly zero external ANN or clustering libraries.

Mathematical Foundations:
------------------------
1. Vector L2 Normalization:
   Given a vector v in R^D, its Euclidean (L2) norm is:
       ||v||_2 = sqrt( sum_{i=1}^D v_i^2 )
   The unit-normalized vector is:
       v_hat = v / ||v||_2  (for ||v||_2 > epsilon)
       v_hat = 0            (for ||v||_2 <= epsilon, handling zero vectors safely)

2. Cosine Similarity:
   The cosine of the angle theta between two vectors u and v is:
       cos(theta) = <u, v> / (||u||_2 * ||v||_2) = <u_hat, v_hat>
   where <., .> denotes the inner (dot) product.
   The value lies in [-1.0, 1.0], where 1.0 indicates identical direction,
   0.0 indicates orthogonality, and -1.0 indicates opposite direction.

3. Vector-Matrix & Batch Operations:
   For a query matrix Q in R^(M x D) and dataset matrix X in R^(N x D):
       S = Q_hat @ X_hat.T  in R^(M x N)
   where S_{i, j} = cosine_similarity(Q_i, X_j).
   By leveraging BLAS GEMM (General Matrix Multiply), millions of similarity
   scores are calculated in parallel without Python loops.
"""

from typing import Union
import numpy as np


def normalize_vectors(
    x: np.ndarray,
    eps: float = 1e-10,
    axis: int = -1,
) -> np.ndarray:
    """
    L2-normalizes vectors along a specified axis.

    Mathematical formulation:
        ||x||_2 = sqrt( sum(x_i^2) )
        x_hat = x / ||x||_2  if ||x||_2 > eps else 0

    Zero-Vector Safety:
        If a vector has norm <= eps (e.g. all zeros or near-zero), it is mapped
        to an all-zero vector rather than producing NaN/Inf or throwing warnings.

    Parameters:
        x: Input array of shape (D,) or (..., D).
        eps: Small positive constant for numerical stability. Default: 1e-10.
        axis: Axis along which to compute the L2 norm. Default: -1.

    Returns:
        L2-normalized array with the same shape and float dtype as input.
    """
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)

    # Compute Euclidean norms along the target axis
    norms = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)

    # Boolean mask of strictly positive norms
    valid_mask = norms > eps

    # Allocate zero array matching arr shape and dtype
    normalized = np.zeros_like(arr)

    # Safe division only where norms > eps (prevents 0/0 -> NaN and avoids warnings)
    np.divide(arr, norms, out=normalized, where=valid_mask)

    return normalized


def cosine_similarity_pairwise(
    u: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Computes cosine similarity between two 1D vectors u and v.

    Mathematical formulation:
        sim(u, v) = <u, v> / (||u||_2 * ||v||_2)

    Edge Case Handling:
        If either vector is a zero vector (norm <= eps), the cosine similarity
        is geometrically undefined; by convention in vector retrieval, this returns 0.0.

    Parameters:
        u: 1D NumPy array of shape (D,).
        v: 1D NumPy array of shape (D,).
        eps: Epsilon threshold for zero-vector detection.

    Returns:
        Cosine similarity as a Python float in range [-1.0, 1.0].
    """
    u_arr = np.asarray(u, dtype=np.float64).ravel()
    v_arr = np.asarray(v, dtype=np.float64).ravel()

    if u_arr.shape != v_arr.shape:
        raise ValueError(
            f"Shape mismatch: vector u has shape {u_arr.shape} but vector v has shape {v_arr.shape}."
        )

    norm_u = float(np.linalg.norm(u_arr))
    norm_v = float(np.linalg.norm(v_arr))

    # Safe handling of zero vectors
    if norm_u <= eps or norm_v <= eps:
        return 0.0

    dot_product = float(np.dot(u_arr, v_arr))
    sim = dot_product / (norm_u * norm_v)

    # Clip for floating point precision boundary violations
    return float(np.clip(sim, -1.0, 1.0))


def cosine_similarity(
    query: np.ndarray,
    vectors: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Computes cosine similarity between a single query vector and a matrix of vectors.

    Mathematical formulation:
        For query q in R^D and dataset X in R^(N x D):
        sim(q, X_i) = <q_hat, X_hat_i>
        sim(q, X)   = X_hat @ q_hat in R^N

    Optimization:
        Converts the calculation into a single BLAS matrix-vector product (GEMV).
        Zero Python loops are used.

    Parameters:
        query: 1D array of shape (D,) or 2D array of shape (1, D).
        vectors: 2D array of shape (N, D).
        eps: Epsilon threshold for zero-vector safety.

    Returns:
        1D array of shape (N,) with cosine similarity scores in range [-1.0, 1.0].
    """
    q_arr = np.asarray(query, dtype=np.float64).ravel()
    v_arr = np.asarray(vectors, dtype=np.float64)

    if v_arr.ndim != 2:
        raise ValueError(f"Expected vectors to be a 2D matrix of shape (N, D), got {v_arr.shape}.")

    n_samples, dim = v_arr.shape
    if n_samples == 0:
        return np.empty(0, dtype=np.float64)

    if q_arr.shape[0] != dim:
        raise ValueError(
            f"Dimension mismatch: query dimension {q_arr.shape[0]} does not match vectors dimension {dim}."
        )

    # Normalize query and vectors safely
    q_norm = normalize_vectors(q_arr, eps=eps)
    v_norm = normalize_vectors(v_arr, eps=eps, axis=1)

    # Vectorized matrix-vector multiplication
    scores = np.dot(v_norm, q_norm)

    # Clip for numerical stability
    return np.clip(scores, -1.0, 1.0)


def cosine_similarity_batch(
    queries: np.ndarray,
    vectors: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Computes all pairwise cosine similarities between a batch of M queries
    and N dataset vectors.

    Mathematical formulation:
        For queries Q in R^(M x D) and vectors X in R^(N x D):
            S = Q_hat @ X_hat.T in R^(M x N)
        where S_{i, j} = cosine_similarity(Q_i, X_j).

    Optimization:
        Executes via highly optimized BLAS GEMM (General Matrix Multiply).
        Entirely vectorized in C/Fortran primitives without Python loops.

    Parameters:
        queries: 2D array of shape (M, D).
        vectors: 2D array of shape (N, D).
        eps: Epsilon threshold for zero-vector safety.

    Returns:
        2D array of shape (M, N) containing cosine similarity scores.
    """
    q_arr = np.asarray(queries, dtype=np.float64)
    v_arr = np.asarray(vectors, dtype=np.float64)

    if q_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got queries={q_arr.shape} and vectors={v_arr.shape}."
        )

    m_queries, q_dim = q_arr.shape
    n_vectors, v_dim = v_arr.shape

    if q_dim != v_dim:
        raise ValueError(
            f"Dimension mismatch: queries dimension {q_dim} != vectors dimension {v_dim}."
        )

    if m_queries == 0 or n_vectors == 0:
        return np.empty((m_queries, n_vectors), dtype=np.float64)

    # Vectorized normalization
    q_norm = normalize_vectors(q_arr, eps=eps, axis=1)
    v_norm = normalize_vectors(v_arr, eps=eps, axis=1)

    # BLAS GEMM: (M, D) @ (D, N) -> (M, N)
    scores = np.dot(q_norm, v_norm.T)

    return np.clip(scores, -1.0, 1.0)


def l2_distance(
    query: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    """
    Computes Euclidean (L2) distance between query vector and matrix of vectors.

    Mathematical formulation:
        dist(q, X_i) = ||q - X_i||_2 = sqrt( sum_{j=1}^D (q_j - X_{i, j})^2 )

    Parameters:
        query: 1D array of shape (D,).
        vectors: 2D array of shape (N, D).

    Returns:
        1D array of shape (N,) with non-negative Euclidean distances.
    """
    q_arr = np.asarray(query, dtype=np.float64).ravel()
    v_arr = np.asarray(vectors, dtype=np.float64)

    if v_arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for vectors, got {v_arr.shape}.")

    if v_arr.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    diff = v_arr - q_arr  # Broadcasting (N, D) - (D,)
    return np.linalg.norm(diff, ord=2, axis=1)


def top_k(
    scores: np.ndarray,
    k: int,
    largest: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selects top-k scores and their indices in O(N + k log k) time complexity.

    Algorithm:
        Instead of sorting the entire array (O(N log N)), uses `np.argpartition`
        to partition the array in O(N) linear time, then sorts only the k
        candidates in O(k log k).

    Parameters:
        scores: 1D array of shape (N,).
        k: Number of elements to retrieve.
        largest: If True, retrieves largest values (similarity). If False, smallest (distance).

    Returns:
        (top_indices, top_scores) where top_scores are sorted descending (if largest=True).
    """
    s = np.asarray(scores).ravel()
    n = len(s)

    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=s.dtype)

    k = min(k, n)

    if largest:
        # For largest, we want the largest k elements
        if k == n:
            sorted_order = np.argsort(-s)
            return sorted_order, s[sorted_order]
        partition_idx = np.argpartition(-s, k)[:k]
        sorted_top_k = partition_idx[np.argsort(-s[partition_idx])]
    else:
        # For smallest, we want the smallest k elements
        if k == n:
            sorted_order = np.argsort(s)
            return sorted_order, s[sorted_order]
        partition_idx = np.argpartition(s, k)[:k]
        sorted_top_k = partition_idx[np.argsort(s[partition_idx])]

    return sorted_top_k, s[sorted_top_k]

