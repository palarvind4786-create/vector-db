"""
Exact brute-force vector index using NumPy.
Scans all active vectors linearly to establish exact top-k ground truth.
Strictly zero external ANN or clustering libraries.
"""

from typing import Any, Sequence
import numpy as np
from app.core.similarity import normalize_vectors, top_k


class SearchResult:
    """
    Structured search results container.
    Supports attribute access (.ids, .scores, .metadata),
    iteration as list of result dictionaries, and indexing.
    """

    def __init__(
        self,
        ids: list[Any],
        scores: np.ndarray,
        metadata: list[dict[str, Any]],
    ):
        self.ids = ids
        self.scores = scores
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self.ids)):
            yield {
                "id": self.ids[i],
                "score": float(self.scores[i]),
                "metadata": self.metadata[i],
            }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "id": self.ids[idx],
            "score": float(self.scores[idx]),
            "metadata": self.metadata[idx],
        }

    def to_list(self) -> list[dict[str, Any]]:
        return list(self)

    def __repr__(self) -> str:
        return f"SearchResult(count={len(self.ids)}, top_id={self.ids[0] if self.ids else None})"


class ExactIndex:
    """
    Brute-force exact nearest neighbor index using NumPy.

    Features:
    - Contiguous float32 vector storage.
    - Pre-computed unit-normalized vectors for instant BLAS GEMV dot products.
    - Efficient O(N + k log k) top-k selection via np.argpartition.
    - Full lazy deletion with tombstone masks and optional compaction.
    - Associated document metadata tracking.
    - Handles k >= active dataset size gracefully.
    """

    def __init__(self, dim: int | None = None, metric: str = "cosine"):
        if metric != "cosine":
            raise ValueError(f"Currently only 'cosine' metric is supported, got '{metric}'.")

        self.dim = dim
        self.metric = metric

        # Storage structures
        self.vectors: np.ndarray = np.empty((0, dim if dim else 0), dtype=np.float32)
        self.vectors_norm: np.ndarray = np.empty((0, dim if dim else 0), dtype=np.float32)
        self.ids: list[Any] = []
        self.metadata: list[dict[str, Any]] = []

        # Fast ID lookup and tombstone management
        self.id_to_idx: dict[Any, int] = {}
        self.deleted_mask: np.ndarray = np.empty(0, dtype=bool)
        self.deleted_ids: set[Any] = set()

    def add(
        self,
        ids: Sequence[Any],
        vectors: np.ndarray,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add a batch of vectors, their IDs, and optional metadata to the index.

        Parameters:
            ids: Sequence of unique identifiers for each vector.
            vectors: 2D NumPy array of shape (N, D).
            metadata: Optional sequence of metadata dictionaries for each vector.
        """
        v_arr = np.asarray(vectors, dtype=np.float32)
        if v_arr.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (N, D), got shape {v_arr.shape}.")

        n_new, d = v_arr.shape
        if n_new != len(ids):
            raise ValueError(
                f"Mismatch: number of vectors ({n_new}) != number of IDs ({len(ids)})."
            )

        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {d}.")

        if metadata is None:
            meta_list = [{} for _ in range(n_new)]
        else:
            if len(metadata) != n_new:
                raise ValueError(
                    f"Mismatch: metadata length ({len(metadata)}) != vectors length ({n_new})."
                )
            meta_list = [dict(m) for m in metadata]

        # Normalize new vectors for cosine similarity
        v_norm = normalize_vectors(v_arr, axis=1).astype(np.float32)

        start_idx = len(self.ids)
        new_mask = np.zeros(n_new, dtype=bool)

        # Update or append
        if len(self.vectors) == 0:
            self.vectors = v_arr
            self.vectors_norm = v_norm
            self.deleted_mask = new_mask
        else:
            self.vectors = np.vstack([self.vectors, v_arr])
            self.vectors_norm = np.vstack([self.vectors_norm, v_norm])
            self.deleted_mask = np.concatenate([self.deleted_mask, new_mask])

        for i, doc_id in enumerate(ids):
            # If ID was previously added, tombstone the prior instance to prevent ghost duplicates
            if doc_id in self.id_to_idx:
                old_idx = self.id_to_idx[doc_id]
                self.deleted_mask[old_idx] = True

            if doc_id in self.deleted_ids:
                self.deleted_ids.remove(doc_id)

            current_pos = start_idx + i
            self.ids.append(doc_id)
            self.metadata.append(meta_list[i])
            self.id_to_idx[doc_id] = current_pos

    def insert(
        self,
        id: Any,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Convenience method to insert a single vector."""
        v_arr = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        meta = [metadata] if metadata is not None else None
        self.add([id], v_arr, meta)

    def search(self, query: np.ndarray, k: int = 10) -> SearchResult:
        """
        Scan all active vectors using cosine similarity and return top-k results.

        Parameters:
            query: Query vector of shape (D,) or (1, D).
            k: Maximum number of nearest neighbors to retrieve.

        Returns:
            SearchResult container with matching IDs, scores, and metadata.
        """
        k = max(1, int(k))

        if self.dim is None or len(self.ids) == 0:
            return SearchResult([], np.empty(0, dtype=np.float32), [])

        q_arr = np.asarray(query, dtype=np.float32).ravel()
        if q_arr.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {q_arr.shape[0]} does not match index dimension {self.dim}."
            )

        # Normalize query vector safely
        q_norm = normalize_vectors(q_arr).astype(np.float32)

        # Filter out tombstoned vectors directly against deleted_mask
        has_deletions = len(self.deleted_mask) > 0 and self.deleted_mask.any()
        if has_deletions:
            active_indices = np.where(~self.deleted_mask)[0]
            if len(active_indices) == 0:
                return SearchResult([], np.empty(0, dtype=np.float32), [])

            active_vectors = self.vectors_norm[active_indices]
            # BLAS matrix-vector product with clipping to [-1.0, 1.0]
            scores = np.clip(np.dot(active_vectors, q_norm), -1.0, 1.0)
            local_indices, top_scores = top_k(scores, k=min(k, len(active_indices)), largest=True)
            chosen_global_indices = active_indices[local_indices]
        else:
            if len(self.vectors_norm) == 0:
                return SearchResult([], np.empty(0, dtype=np.float32), [])

            # BLAS matrix-vector product across entire continuous buffer with clipping
            scores = np.clip(np.dot(self.vectors_norm, q_norm), -1.0, 1.0)
            chosen_global_indices, top_scores = top_k(scores, k=min(k, len(scores)), largest=True)

        res_ids = [self.ids[idx] for idx in chosen_global_indices]
        res_meta = [self.metadata[idx] for idx in chosen_global_indices]

        return SearchResult(res_ids, top_scores, res_meta)

    def delete(self, ids: Any | Sequence[Any]) -> int:
        """
        Mark one or more vector IDs as deleted (lazy deletion / tombstone).
        Deleted vectors are guaranteed never to appear in search results.

        Returns:
            Count of vectors newly marked as deleted.
        """
        if isinstance(ids, (str, int)) or not hasattr(ids, "__iter__"):
            target_ids = [ids]
        else:
            target_ids = list(ids)

        deleted_count = 0
        for doc_id in target_ids:
            if doc_id in self.id_to_idx and doc_id not in self.deleted_ids:
                idx = self.id_to_idx[doc_id]
                self.deleted_mask[idx] = True
                self.deleted_ids.add(doc_id)
                deleted_count += 1

        return deleted_count

    def compact(self) -> None:
        """
        Physically remove tombstoned vectors from memory and compact the storage array.
        """
        if len(self.deleted_ids) == 0:
            return

        active_indices = np.where(~self.deleted_mask)[0]

        self.vectors = self.vectors[active_indices]
        self.vectors_norm = self.vectors_norm[active_indices]
        self.ids = [self.ids[i] for i in active_indices]
        self.metadata = [self.metadata[i] for i in active_indices]

        # Rebuild ID lookup map
        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.deleted_mask = np.zeros(len(self.ids), dtype=bool)
        self.deleted_ids.clear()

    def save(self, directory: str) -> None:
        """Persist ExactIndex state to disk (.npy + .json)."""
        import os, json
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        np.save(os.path.join(directory, "vectors_norm.npy"), self.vectors_norm)
        np.save(os.path.join(directory, "deleted_mask.npy"), self.deleted_mask)
        meta_data = {
            "dim": self.dim,
            "metric": self.metric,
            "ids": self.ids,
            "metadata": self.metadata,
            "id_to_idx": self.id_to_idx,
            "deleted_ids": list(self.deleted_ids),
        }
        with open(os.path.join(directory, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

    def load(self, directory: str) -> "ExactIndex":
        """Load ExactIndex state from disk."""
        import os, json
        with open(os.path.join(directory, "index_meta.json"), "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.dim = meta_data["dim"]
        self.metric = meta_data["metric"]
        self.ids = meta_data["ids"]
        self.metadata = meta_data["metadata"]
        self.id_to_idx = meta_data["id_to_idx"]
        self.deleted_ids = set(meta_data["deleted_ids"])
        self.vectors = np.load(os.path.join(directory, "vectors.npy"))
        self.vectors_norm = np.load(os.path.join(directory, "vectors_norm.npy"))
        self.deleted_mask = np.load(os.path.join(directory, "deleted_mask.npy"))
        return self

    def size(self) -> int:
        """Return the count of active (non-deleted) vectors in the index."""
        return len(self.ids) - len(self.deleted_ids)

    def total_size(self) -> int:
        """Return total count of vectors including tombstoned entries."""
        return len(self.ids)

    def clear(self) -> None:
        """Reset the index and purge all stored data."""
        self.vectors = np.empty((0, self.dim if self.dim else 0), dtype=np.float32)
        self.vectors_norm = np.empty((0, self.dim if self.dim else 0), dtype=np.float32)
        self.ids.clear()
        self.metadata.clear()
        self.id_to_idx.clear()
        self.deleted_mask = np.empty(0, dtype=bool)
        self.deleted_ids.clear()


