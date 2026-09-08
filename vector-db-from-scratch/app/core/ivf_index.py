"""
IVF-Flat (Inverted File Flat) approximate nearest-neighbor index from scratch using NumPy.
Strictly ZERO external ANN or clustering libraries (NO FAISS, Annoy, HNSW, Pinecone, Chroma).

Mathematical & Architectural Design:
-----------------------------------
1. Coarse Quantization:
   The high-dimensional vector space is partitioned into K Voronoi cells using
   custom K-Means. Each Voronoi cell is identified by its centroid C_j (j in [0, K-1]).

2. Inverted Posting Lists:
   Instead of duplicating vectors across multiple structures, posting lists store
   only the integer indices into a single contiguous vector storage array.
   posting_lists[j] = [internal_id_1, internal_id_2, ...]

3. Multi-Probe Query Routing (nprobe):
   Given query vector q:
     a. Compute cosine similarity against all K centroids: S_c = centroids @ q_hat
     b. Select top nprobe centroids {C_p1, ..., C_pnprobe}
     c. Gather candidate vector IDs from the inverted lists of those nprobe centroids
     d. Filter out tombstoned (lazily deleted) vectors
     e. Calculate exact cosine similarity only for active candidate vectors
     f. Select top-k matches using O(M + k log k) argpartition
"""

import os
import json
import time
from typing import Any, Sequence
import numpy as np
from app.core.kmeans import KMeans
from app.core.similarity import normalize_vectors, top_k


class IVFSearchResult:
    """
    Structured search results container for IVF-Flat queries.
    Exposes matched items, latency, and detailed candidate examination metrics.
    """

    def __init__(
        self,
        ids: list[Any],
        scores: np.ndarray,
        metadata: list[dict[str, Any]],
        latency_ms: float,
        nprobe: int,
        num_candidates: int,
        num_compared: int,
        nlist: int,
        probed_clusters: list[int],
    ):
        self.ids = ids
        self.scores = scores
        self.metadata = metadata
        self.latency_ms = latency_ms
        self.nprobe = nprobe
        self.num_candidates = num_candidates
        self.num_compared = num_compared
        self.nlist = nlist
        self.probed_clusters = probed_clusters

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {"id": self.ids[i], "score": float(self.scores[i]), "metadata": self.metadata[i]}
                for i in range(len(self.ids))
            ],
            "latency_ms": self.latency_ms,
            "nprobe": self.nprobe,
            "num_candidates": self.num_candidates,
            "num_compared": self.num_compared,
            "nlist": self.nlist,
            "probed_clusters": self.probed_clusters,
        }

    def __repr__(self) -> str:
        return (
            f"IVFSearchResult(count={len(self.ids)}, latency={self.latency_ms:.3f}ms, "
            f"compared={self.num_compared}/{self.num_candidates}, nprobe={self.nprobe}/{self.nlist})"
        )


class IVFFlatIndex:
    """
    Inverted File Flat (IVF-Flat) Approximate Nearest Neighbor (ANN) Index.
    Built entirely from mathematical foundations using NumPy.
    """

    def __init__(
        self,
        dim: int | None = None,
        nlist: int = 100,
        nprobe: int = 8,
        metric: str = "cosine",
        seed: int = 42,
    ):
        if nlist <= 0:
            raise ValueError(f"nlist must be > 0, got {nlist}.")
        if nprobe <= 0:
            raise ValueError(f"nprobe must be > 0, got {nprobe}.")
        if metric != "cosine":
            raise ValueError(f"Only 'cosine' metric is currently supported, got '{metric}'.")

        self.dim = dim
        self.nlist = int(nlist)
        self.nprobe = int(min(nprobe, nlist))
        self.metric = metric
        self.seed = seed

        # Coarse quantizer state
        self.centroids: np.ndarray | None = None
        self.is_trained: bool = False

        # Inverted posting lists: cluster_id -> list of internal vector indices
        self.posting_lists: dict[int, list[int]] = {i: [] for i in range(self.nlist)}

        # Contiguous vector storage
        self.vectors: np.ndarray = np.empty((0, dim if dim else 0), dtype=np.float32)
        self.vectors_norm: np.ndarray = np.empty((0, dim if dim else 0), dtype=np.float32)

        # ID mapping & metadata
        self.ids: list[Any] = []
        self.metadata: list[dict[str, Any]] = []
        self.id_to_idx: dict[Any, int] = {}

        # Deletion state (tombstones)
        self.deleted_mask: np.ndarray = np.empty(0, dtype=bool)
        self.deleted_ids: set[Any] = set()

    def train(self, vectors: np.ndarray) -> None:
        """
        Train coarse quantizer centroids on input vectors using custom KMeans.

        Parameters:
            vectors: 2D NumPy array of shape (N, D).
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.shape}.")

        n_samples, d = arr.shape
        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {d}.")

        # Adjust effective nlist if training sample count < nlist
        effective_nlist = min(self.nlist, n_samples)
        if effective_nlist < self.nlist:
            self.nlist = effective_nlist
            self.nprobe = min(self.nprobe, self.nlist)
            self.posting_lists = {i: [] for i in range(self.nlist)}

        # Run custom KMeans from scratch with spherical normalization
        kmeans = KMeans(
            n_clusters=self.nlist,
            max_iter=30,
            metric="cosine",
            init="k-means++",
            seed=self.seed,
        )
        kmeans.fit(arr)

        self.centroids = kmeans.centroids  # Already unit-normalized
        self.is_trained = True

    def add(
        self,
        ids: Sequence[Any],
        vectors: np.ndarray,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add vectors, IDs, and metadata to the IVF index.
        Assigns each vector to its nearest centroid Voronoi cell.

        Parameters:
            ids: Unique external identifiers for each vector.
            vectors: 2D NumPy array of shape (N, D).
            metadata: Optional list of metadata dictionaries.
        """
        v_arr = np.asarray(vectors, dtype=np.float32)
        if v_arr.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (N, D), got {v_arr.shape}.")

        n_new, d = v_arr.shape
        if n_new != len(ids):
            raise ValueError(f"Mismatch: vectors count ({n_new}) != IDs count ({len(ids)}).")

        if self.dim is None:
            self.dim = d
        elif self.dim != d:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {d}.")

        # Train coarse quantizer if not already trained
        if not self.is_trained:
            self.train(v_arr)

        if metadata is None:
            meta_list = [{} for _ in range(n_new)]
        else:
            if len(metadata) != n_new:
                raise ValueError(f"Mismatch: metadata length ({len(metadata)}) != vectors ({n_new}).")
            meta_list = [dict(m) for m in metadata]

        # Unit-normalize new vectors for cosine similarity
        v_norm = normalize_vectors(v_arr, axis=1).astype(np.float32)

        start_idx = len(self.ids)
        new_mask = np.zeros(n_new, dtype=bool)

        # Append to contiguous storage
        if len(self.vectors) == 0:
            self.vectors = v_arr
            self.vectors_norm = v_norm
            self.deleted_mask = new_mask
        else:
            self.vectors = np.vstack([self.vectors, v_arr])
            self.vectors_norm = np.vstack([self.vectors_norm, v_norm])
            self.deleted_mask = np.concatenate([self.deleted_mask, new_mask])

        # Assign vectors to nearest centroid via dot product (cosine similarity)
        # S_c = v_norm @ centroids.T  in R^(N_new x K)
        centroid_scores = np.dot(v_norm, self.centroids.T)
        assigned_clusters = np.argmax(centroid_scores, axis=1)

        for i, doc_id in enumerate(ids):
            internal_idx = start_idx + i
            cluster_id = int(assigned_clusters[i])

            # If ID was previously added, tombstone the previous instance to prevent ghost duplicates
            if doc_id in self.id_to_idx:
                old_idx = self.id_to_idx[doc_id]
                self.deleted_mask[old_idx] = True

            if doc_id in self.deleted_ids:
                self.deleted_ids.remove(doc_id)

            self.ids.append(doc_id)
            self.metadata.append(meta_list[i])
            self.id_to_idx[doc_id] = internal_idx
            self.posting_lists[cluster_id].append(internal_idx)

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

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
    ) -> IVFSearchResult:
        """
        Execute IVF-Flat multi-probe approximate nearest neighbor search.

        Search Pipeline Stages:
        1. Validate parameters and normalize query vector.
        2. Compare query against all K coarse centroids.
        3. Select top nprobe centroids with highest similarity.
        4. Gather candidate internal IDs from those nprobe posting lists.
        5. Filter out tombstoned (lazily deleted) vectors.
        6. Calculate exact cosine similarity only for active candidate vectors.
        7. Extract top-k results via np.argpartition.
        """
        t_start = time.perf_counter_ns()

        k = max(1, int(k))
        probe_count = int(self.nprobe if nprobe is None else max(1, min(int(nprobe), self.nlist)))

        if not self.is_trained or len(self.ids) == 0 or self.centroids is None:
            return IVFSearchResult(
                ids=[],
                scores=np.empty(0, dtype=np.float32),
                metadata=[],
                latency_ms=0.0,
                nprobe=probe_count,
                num_candidates=0,
                num_compared=0,
                nlist=self.nlist,
                probed_clusters=[],
            )

        q_arr = np.asarray(query, dtype=np.float32).ravel()
        if q_arr.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension {q_arr.shape[0]} does not match index dimension {self.dim}."
            )

        # Stage 1: Normalize query safely
        q_norm = normalize_vectors(q_arr).astype(np.float32)

        # Stage 2: Coarse Quantization - Compare query against all K centroids
        centroid_scores = np.dot(self.centroids, q_norm)

        # Stage 3: Select nearest nprobe centroids
        top_centroid_indices, _ = top_k(centroid_scores, k=probe_count, largest=True)
        probed_clusters = [int(idx) for idx in top_centroid_indices]

        # Stage 4: Collect candidate internal IDs from those inverted lists
        raw_candidates: list[int] = []
        for c_id in probed_clusters:
            raw_candidates.extend(self.posting_lists[c_id])

        num_candidates = len(raw_candidates)

        # Stage 5: Filter out tombstoned (deleted) candidates directly against mask
        active_candidates = [idx for idx in raw_candidates if not self.deleted_mask[idx]]
        num_compared = len(active_candidates)

        # Stage 6: Calculate exact cosine similarity only for active candidate vectors
        if num_compared == 0:
            t_end = time.perf_counter_ns()
            return IVFSearchResult(
                ids=[],
                scores=np.empty(0, dtype=np.float32),
                metadata=[],
                latency_ms=(t_end - t_start) / 1e6,
                nprobe=probe_count,
                num_candidates=num_candidates,
                num_compared=0,
                nlist=self.nlist,
                probed_clusters=probed_clusters,
            )

        cand_indices_arr = np.array(active_candidates, dtype=np.int64)
        # Vectorized slice from contiguous vector store
        cand_vectors_norm = self.vectors_norm[cand_indices_arr]
        # Fine search: BLAS matrix-vector product with clipping to [-1.0, 1.0]
        cand_scores = np.clip(np.dot(cand_vectors_norm, q_norm), -1.0, 1.0)

        # Stage 7: Select top-k results
        k_target = min(k, num_compared)
        local_top, top_scores = top_k(cand_scores, k=k_target, largest=True)

        chosen_global_indices = cand_indices_arr[local_top]
        res_ids = [self.ids[idx] for idx in chosen_global_indices]
        res_meta = [self.metadata[idx] for idx in chosen_global_indices]

        t_end = time.perf_counter_ns()
        latency_ms = (t_end - t_start) / 1e6

        return IVFSearchResult(
            ids=res_ids,
            scores=top_scores,
            metadata=res_meta,
            latency_ms=latency_ms,
            nprobe=probe_count,
            num_candidates=num_candidates,
            num_compared=num_compared,
            nlist=self.nlist,
            probed_clusters=probed_clusters,
        )

    def delete(self, ids: Any | Sequence[Any]) -> int:
        """
        Mark one or more vector IDs as deleted (lazy deletion).
        Tombstoned vectors remain in posting lists but are ignored during search.

        Returns:
            Count of vectors newly marked as deleted.
        """
        if isinstance(ids, (str, int)) or not hasattr(ids, "__iter__"):
            target_ids = [ids]
        else:
            target_ids = list(ids)

        del_count = 0
        for doc_id in target_ids:
            if doc_id in self.id_to_idx and doc_id not in self.deleted_ids:
                idx = self.id_to_idx[doc_id]
                self.deleted_mask[idx] = True
                self.deleted_ids.add(doc_id)
                del_count += 1

        return del_count

    def rebuild(self, retrain_kmeans: bool = False) -> None:
        """
        Purge tombstoned vectors, compact storage, and rebuild clean inverted lists.

        Parameters:
            retrain_kmeans: If True, retrains centroids on surviving active vectors.
                            If False, reassigns active vectors to existing centroids.
        """
        if len(self.deleted_ids) == 0 and not retrain_kmeans:
            return

        active_indices = np.where(~self.deleted_mask)[0]

        active_vectors = self.vectors[active_indices]
        active_vectors_norm = self.vectors_norm[active_indices]
        active_ids = [self.ids[i] for i in active_indices]
        active_metadata = [self.metadata[i] for i in active_indices]

        # Reset storage
        self.vectors = active_vectors
        self.vectors_norm = active_vectors_norm
        self.ids = active_ids
        self.metadata = active_metadata
        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.deleted_mask = np.zeros(len(self.ids), dtype=bool)
        self.deleted_ids.clear()

        # Reset inverted lists
        self.posting_lists = {i: [] for i in range(self.nlist)}

        if len(self.vectors) == 0:
            return

        if retrain_kmeans:
            self.train(self.vectors)

        # Reassign all active vectors to centroids
        centroid_scores = np.dot(self.vectors_norm, self.centroids.T)
        assigned_clusters = np.argmax(centroid_scores, axis=1)

        for idx, c_id in enumerate(assigned_clusters):
            self.posting_lists[int(c_id)].append(idx)

    def save(self, directory: str) -> None:
        """
        Persist complete IVF-Flat state to disk without external vector DB dependencies.
        Saves centroids & vectors as .npy binary arrays, and inverted lists & metadata as .json.
        """
        os.makedirs(directory, exist_ok=True)

        # 1. Save binary arrays
        np.save(os.path.join(directory, "centroids.npy"), self.centroids if self.centroids is not None else np.empty(0))
        np.save(os.path.join(directory, "vectors.npy"), self.vectors)
        np.save(os.path.join(directory, "vectors_norm.npy"), self.vectors_norm)
        np.save(os.path.join(directory, "deleted_mask.npy"), self.deleted_mask)

        # 2. Save structured metadata and posting lists as JSON
        config_data = {
            "dim": self.dim,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "metric": self.metric,
            "seed": self.seed,
            "is_trained": self.is_trained,
            "ids": self.ids,
            "metadata": self.metadata,
            "id_to_idx": self.id_to_idx,
            "deleted_ids": list(self.deleted_ids),
            "posting_lists": {str(k): v for k, v in self.posting_lists.items()},
        }
        with open(os.path.join(directory, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def load(self, directory: str) -> "IVFFlatIndex":
        """
        Restore complete IVF-Flat state from disk.
        """
        with open(os.path.join(directory, "index_meta.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dim = config["dim"]
        self.nlist = config["nlist"]
        self.nprobe = config["nprobe"]
        self.metric = config["metric"]
        self.seed = config["seed"]
        self.is_trained = config["is_trained"]
        self.ids = config["ids"]
        self.metadata = config["metadata"]
        self.id_to_idx = config["id_to_idx"]
        self.deleted_ids = set(config["deleted_ids"])
        self.posting_lists = {int(k): v for k, v in config["posting_lists"].items()}

        centroids_path = os.path.join(directory, "centroids.npy")
        if os.path.exists(centroids_path):
            c_arr = np.load(centroids_path)
            self.centroids = c_arr if c_arr.size > 0 else None

        self.vectors = np.load(os.path.join(directory, "vectors.npy"))
        self.vectors_norm = np.load(os.path.join(directory, "vectors_norm.npy"))
        self.deleted_mask = np.load(os.path.join(directory, "deleted_mask.npy"))

        return self

    def get_cluster_stats(self) -> dict[str, Any]:
        """Return distribution metrics across all Voronoi inverted lists."""
        sizes = [len(self.posting_lists[i]) for i in range(self.nlist)]
        non_empty = [s for s in sizes if s > 0]

        return {
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "total_vectors": len(self.ids),
            "active_vectors": len(self.ids) - len(self.deleted_ids),
            "tombstones": len(self.deleted_ids),
            "empty_clusters": sizes.count(0),
            "min_cluster_size": min(sizes) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
            "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
            "cluster_sizes": {i: sizes[i] for i in range(self.nlist)},
        }

    def size(self) -> int:
        """Return active non-deleted vector count."""
        return len(self.ids) - len(self.deleted_ids)

    def total_size(self) -> int:
        """Return total vector count including tombstones."""
        return len(self.ids)


