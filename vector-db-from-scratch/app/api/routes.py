"""
FastAPI route handlers for documents, vector search, index management, and statistics.
Provides RESTful endpoints for Exact and IVF-Flat vector search engine.
"""

import os
import json
import time
import uuid
import threading
from typing import Any
import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.core.exact_index import ExactIndex
from app.core.ivf_index import IVFFlatIndex
from app.models.schemas import (
    InsertRequest,
    InsertResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    DeleteResponse,
    RebuildRequest,
    RebuildResponse,
    StatsResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkMetric,
)

router = APIRouter()


class IndexManager:
    """
    Thread-safe in-memory container managing ExactIndex and IVFFlatIndex singletons.
    Initializes automatically on startup using precomputed dataset if present.
    """

    def __init__(self, dim: int = 384, nlist: int = 100, nprobe: int = 8):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.exact_index = ExactIndex(dim=dim, metric="cosine")
        self.ivf_index = IVFFlatIndex(dim=dim, nlist=nlist, nprobe=nprobe, metric="cosine", seed=42)
        self.is_initialized = False
        self._lock = threading.Lock()
        self._model = None

    def get_embedding_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Generate 384-dimensional dense embedding for given text."""
        model = self.get_embedding_model()
        vec = model.encode([text], show_progress_bar=False)[0]
        return vec.astype(np.float32)

    def initialize_from_data(self, data_dir: str = "data") -> bool:
        """Load pre-computed embeddings and documents from disk into both indices."""
        with self._lock:
            if self.is_initialized:
                return True

            emb_path = os.path.join(data_dir, "embeddings.npy")
            doc_path = os.path.join(data_dir, "documents.json")

            if os.path.exists(emb_path) and os.path.exists(doc_path):
                embeddings = np.load(emb_path).astype(np.float32)
                with open(doc_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)

                ids = [d["id"] for d in documents]
                metas = [{"text": d.get("text", ""), **d.get("metadata", {})} for d in documents]

                self.dim = embeddings.shape[1]
                self.exact_index = ExactIndex(dim=self.dim, metric="cosine")
                self.ivf_index = IVFFlatIndex(
                    dim=self.dim,
                    nlist=self.nlist,
                    nprobe=self.nprobe,
                    metric="cosine",
                    seed=42,
                )

                self.exact_index.add(ids, embeddings, metas)
                self.ivf_index.add(ids, embeddings, metas)
                self.is_initialized = True
                return True
            else:
                self.is_initialized = True
                return False

    def ensure_initialized(self):
        """Ensure indexes are ready before servicing requests."""
        if not self.is_initialized:
            self.initialize_from_data()


# Global index manager singleton
index_manager = IndexManager()


@router.post("/vectors", response_model=InsertResponse, status_code=status.HTTP_201_CREATED)
@router.post("/documents", response_model=InsertResponse, status_code=status.HTTP_201_CREATED)
async def insert_vectors(payload: InsertRequest):
    """
    Insert a vector/document or a batch of vectors/documents.
    Accepts raw vector float arrays or raw text (automatically embedded).
    Synchronously updates both ExactIndex and IVFFlatIndex.
    """
    index_manager.ensure_initialized()

    # Determine items to insert (support single item or batch documents list)
    if payload.documents:
        items = payload.documents
    else:
        items = [payload]

    inserted_ids = []

    for item in items:
        # Determine vector or text
        vec = None
        if item.vector is not None:
            if len(item.vector) != index_manager.dim:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Dimension mismatch: expected {index_manager.dim} dimensions, "
                        f"got {len(item.vector)}."
                    ),
                )
            vec = np.asarray(item.vector, dtype=np.float32)
        elif item.text is not None and item.text.strip():
            try:
                vec = index_manager.embed_text(item.text.strip())
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate embedding: {str(e)}",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each vector item must provide either 'vector' (float list) or 'text' (string).",
            )

        # Determine doc_id
        doc_id = item.id or f"vec_{uuid.uuid4().hex[:10]}"

        # Assemble metadata
        meta = dict(item.metadata) if item.metadata else {}
        if item.text and "text" not in meta:
            meta["text"] = item.text

        # Synchronously insert into both ExactIndex and IVFFlatIndex
        with index_manager._lock:
            index_manager.exact_index.insert(doc_id, vec, meta)
            index_manager.ivf_index.insert(doc_id, vec, meta)

        inserted_ids.append(doc_id)

    return InsertResponse(
        id=inserted_ids[0] if len(inserted_ids) == 1 else None,
        inserted_count=len(inserted_ids),
        ids=inserted_ids,
        message=f"Successfully inserted {len(inserted_ids)} vector(s).",
        dim=index_manager.dim,
    )


@router.post("/search", response_model=SearchResponse)
async def search_vectors(payload: SearchRequest):
    """
    Perform semantic vector search using either 'exact' (brute-force) or 'ivf' (IVF-Flat) mode.
    Accepts text queries or dense vector float arrays.
    """
    index_manager.ensure_initialized()

    # 1. Resolve and validate query vector
    if isinstance(payload.query, str):
        q_text = payload.query.strip()
        if not q_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text cannot be empty.",
            )
        try:
            q_vec = index_manager.embed_text(q_text)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text embedding failed: {str(e)}",
            )
    elif isinstance(payload.query, (list, tuple)):
        if len(payload.query) != index_manager.dim:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Query vector dimension {len(payload.query)} does not match "
                    f"index dimension {index_manager.dim}."
                ),
            )
        q_vec = np.asarray(payload.query, dtype=np.float32)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'query' is required (must be text string or float vector).",
        )

    # 2. Validate parameters
    mode = payload.mode.lower() if payload.mode else "ivf"
    if mode not in ("exact", "ivf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{payload.mode}'. Supported modes are 'exact' and 'ivf'.",
        )

    k = max(1, payload.k)
    nprobe = max(1, payload.nprobe)

    # 3. Route query to appropriate index
    if mode == "exact":
        t0 = time.perf_counter()
        res = index_manager.exact_index.search(q_vec, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, item in enumerate(res, start=1):
            meta = dict(item["metadata"])
            text_snippet = meta.pop("text", None)
            results.append(
                SearchResultItem(
                    id=str(item["id"]),
                    score=float(item["score"]),
                    rank=rank,
                    text=text_snippet,
                    metadata=meta,
                )
            )

        return SearchResponse(
            results=results,
            count=len(results),
            mode="exact",
            k=k,
            nprobe=None,
            latency_ms=round(latency_ms, 3),
            candidates_examined=index_manager.exact_index.size(),
            probed_clusters=None,
        )

    else:  # mode == "ivf"
        t0 = time.perf_counter()
        ivf_res = index_manager.ivf_index.search(q_vec, k=k, nprobe=nprobe)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, item in enumerate(ivf_res, start=1):
            meta = dict(item["metadata"])
            text_snippet = meta.pop("text", None)
            results.append(
                SearchResultItem(
                    id=str(item["id"]),
                    score=float(item["score"]),
                    rank=rank,
                    text=text_snippet,
                    metadata=meta,
                )
            )

        reported_latency = ivf_res.latency_ms if ivf_res.latency_ms > 0 else latency_ms

        return SearchResponse(
            results=results,
            count=len(results),
            mode="ivf",
            k=k,
            nprobe=ivf_res.nprobe,
            latency_ms=round(reported_latency, 3),
            candidates_examined=ivf_res.num_compared,
            probed_clusters=ivf_res.probed_clusters,
        )


@router.delete("/vectors/{id}", response_model=DeleteResponse)
@router.delete("/documents/{id}", response_model=DeleteResponse)
async def delete_vector(id: str):
    """
    Delete a vector by ID (tombstone / lazy deletion).
    Deleted vectors will immediately be excluded from all subsequent searches.
    """
    index_manager.ensure_initialized()

    with index_manager._lock:
        del_exact = index_manager.exact_index.delete(id)
        del_ivf = index_manager.ivf_index.delete(id)

    if del_exact == 0 and del_ivf == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector with ID '{id}' not found or already deleted.",
        )

    return DeleteResponse(
        id=id,
        deleted=True,
        message=f"Vector '{id}' successfully marked as deleted.",
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Retrieve database statistics including vector counts, dimensions,
    metric, cluster counts, and posting list distributions.
    """
    index_manager.ensure_initialized()

    c_stats = index_manager.ivf_index.get_cluster_stats()
    exact_size = index_manager.exact_index.size()
    ivf_size = index_manager.ivf_index.size()

    return StatsResponse(
        total_vectors=c_stats["total_vectors"],
        active_vectors=c_stats["active_vectors"],
        num_vectors=c_stats["active_vectors"],
        tombstone_count=c_stats["tombstones"],
        dimension=index_manager.dim,
        dimensions=index_manager.dim,
        metric=index_manager.ivf_index.metric,
        num_clusters=c_stats["nlist"],
        nlist=c_stats["nlist"],
        nprobe_default=c_stats["nprobe"],
        is_trained=index_manager.ivf_index.is_trained,
        empty_clusters=c_stats["empty_clusters"],
        min_cluster_size=c_stats["min_cluster_size"],
        max_cluster_size=c_stats["max_cluster_size"],
        mean_cluster_size=c_stats["mean_cluster_size"],
        median_cluster_size=c_stats["median_cluster_size"],
        cluster_distribution=c_stats["cluster_sizes"],
        exact_index_size=exact_size,
        ivf_index_size=ivf_size,
    )


@router.post("/rebuild", response_model=RebuildResponse)
@router.post("/index/rebuild", response_model=RebuildResponse)
async def rebuild_index(payload: RebuildRequest = RebuildRequest()):
    """
    Rebuild the IVF index: purges tombstoned vectors, compacts contiguous storage arrays,
    and rebuilds posting lists. Optionally retrains K-Means cluster centroids.
    """
    index_manager.ensure_initialized()

    if index_manager.ivf_index.size() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot rebuild an index with zero active vectors.",
        )

    t0 = time.perf_counter()
    with index_manager._lock:
        index_manager.exact_index.compact()
        index_manager.ivf_index.rebuild(retrain_kmeans=payload.retrain_kmeans)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    return RebuildResponse(
        success=True,
        message="IVF index rebuilt and storage compacted successfully.",
        active_vectors=index_manager.ivf_index.size(),
        num_clusters=index_manager.ivf_index.nlist,
        retrain_kmeans=payload.retrain_kmeans,
        latency_ms=round(duration_ms, 3),
    )


@router.get("/benchmark/latest")
async def get_latest_benchmark():
    """Return the cached results of the latest benchmark run if available."""
    res_path = os.path.join("results", "benchmark_results.json")
    if not os.path.exists(res_path):
        res_path = os.path.join("experiments", "results", "benchmark_results.json")
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    raise HTTPException(status_code=404, detail="No cached benchmark results found.")
