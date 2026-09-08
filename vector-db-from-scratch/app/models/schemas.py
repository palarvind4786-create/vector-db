"""
Pydantic schemas for the Vector Database REST API.
Defines strict data models for vector insertion, semantic search,
vector deletion, statistics, index rebuilding, and benchmarking.
"""

from typing import Any
from pydantic import BaseModel, Field, model_validator


class DocumentItem(BaseModel):
    """Individual vector/document representation."""
    id: str | None = Field(default=None, alias="doc_id", description="Unique vector/document identifier")
    vector: list[float] | None = Field(default=None, description="Dense embedding vector")
    text: str | None = Field(default=None, description="Raw text representation (optional if vector provided)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary document metadata")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "doc_demo_1",
                "text": "NASA announces new Artemis lunar mission.",
                "metadata": {"category": "Sci/Tech", "source": "Reuters"}
            }
        }
    }


class InsertRequest(BaseModel):
    """
    Request model for POST /vectors.
    Supports single vector/document insertion or batch insertion.
    """
    id: str | None = Field(default=None, alias="doc_id", description="Unique vector ID")
    vector: list[float] | None = Field(default=None, description="Dense embedding vector")
    text: str | None = Field(default=None, description="Raw text to embed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary")
    documents: list[DocumentItem] | None = Field(
        default=None,
        description="Optional list of documents for batch ingestion"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "doc_demo_1",
                "text": "New breakthroughs in artificial intelligence algorithms.",
                "metadata": {"category": "Sci/Tech"}
            }
        }
    }


class InsertResponse(BaseModel):
    """Response model for POST /vectors."""
    id: str | None = Field(default=None, description="ID of the inserted vector (for single insert)")
    inserted_count: int = Field(..., description="Number of vectors successfully inserted")
    ids: list[str] = Field(..., description="IDs of all inserted vectors")
    message: str = Field(default="Vector(s) inserted successfully")
    dim: int = Field(..., description="Vector dimensionality")


class SearchRequest(BaseModel):
    """
    Request model for POST /search.
    Supports semantic query text or raw query vector, exact or IVF mode, and nprobe.
    """
    query: str | list[float] | None = Field(
        default=None,
        description="Text query for semantic search or dense float vector"
    )
    query_text: str | None = Field(
        default=None,
        description="Alternative alias for query text"
    )
    query_vector: list[float] | None = Field(
        default=None,
        description="Alternative alias for query vector"
    )
    k: int = Field(default=10, ge=1, le=1000, description="Top-k nearest neighbors to retrieve")
    mode: str = Field(
        default="ivf",
        description="Search algorithm mode: 'exact' (brute-force) or 'ivf' (approximate inverted file)"
    )
    index_type: str | None = Field(
        default=None,
        description="Alternative alias for search mode ('exact' or 'ivf')"
    )
    nprobe: int = Field(
        default=8,
        ge=1,
        description="Number of Voronoi clusters to probe (used when mode='ivf')"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Normalize query field from query_text / query_vector
            if values.get("query") is None:
                if values.get("query_text") is not None:
                    values["query"] = values["query_text"]
                elif values.get("query_vector") is not None:
                    values["query"] = values["query_vector"]

            # Normalize mode from index_type
            if values.get("index_type") is not None and (values.get("mode") is None or values.get("mode") == "ivf"):
                values["mode"] = values["index_type"]

            # Normalize lowercase mode
            if isinstance(values.get("mode"), str):
                values["mode"] = values["mode"].lower()

        return values

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "space exploration astronomy satellite",
                "k": 5,
                "mode": "ivf",
                "nprobe": 8
            }
        }
    }


class SearchResultItem(BaseModel):
    """Individual search match item."""
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Cosine similarity score [-1.0, 1.0]")
    rank: int = Field(..., description="1-indexed rank in results")
    text: str | None = Field(default=None, description="Document text snippet if available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Associated document metadata")


class SearchResponse(BaseModel):
    """Response model for POST /search."""
    results: list[SearchResultItem] = Field(..., description="List of nearest neighbors ranked by similarity")
    count: int = Field(..., description="Number of results returned")
    mode: str = Field(..., description="Search mode used ('exact' or 'ivf')")
    k: int = Field(..., description="Requested top-k")
    nprobe: int | None = Field(default=None, description="nprobe setting used for IVF search")
    latency_ms: float = Field(..., description="Search latency in milliseconds")
    candidates_examined: int = Field(..., description="Number of candidate vectors compared")
    probed_clusters: list[int] | None = Field(
        default=None,
        description="Cluster centroid indices probed (for IVF mode)"
    )


class DeleteResponse(BaseModel):
    """Response model for DELETE /vectors/{id}."""
    id: str = Field(..., description="Document/vector ID")
    deleted: bool = Field(..., description="Whether the deletion was successfully registered")
    message: str = Field(..., description="Status description message")


class StatsResponse(BaseModel):
    """Response model for GET /stats."""
    total_vectors: int = Field(..., description="Total vectors loaded including tombstones")
    active_vectors: int = Field(..., description="Active non-deleted vectors available for search")
    num_vectors: int = Field(..., description="Alias for active vectors")
    tombstone_count: int = Field(..., description="Count of lazily deleted tombstoned vectors")
    dimension: int | None = Field(..., description="Vector dimensionality")
    dimensions: int | None = Field(..., description="Alias for vector dimensionality")
    metric: str = Field(..., description="Distance/similarity metric ('cosine')")
    num_clusters: int = Field(..., description="Number of Voronoi coarse quantizer clusters (nlist)")
    nlist: int = Field(..., description="Alias for number of clusters")
    nprobe_default: int = Field(..., description="Default nprobe search parameter")
    is_trained: bool = Field(..., description="Whether IVF quantizer is trained")
    empty_clusters: int = Field(..., description="Number of empty inverted file posting lists")
    min_cluster_size: int = Field(..., description="Smallest posting list size")
    max_cluster_size: int = Field(..., description="Largest posting list size")
    mean_cluster_size: float = Field(..., description="Mean posting list size")
    median_cluster_size: float = Field(..., description="Median posting list size")
    cluster_distribution: dict[int, int] = Field(default_factory=dict, description="Map of cluster ID to vector count")
    exact_index_size: int = Field(..., description="Active vector count in ExactIndex")
    ivf_index_size: int = Field(..., description="Active vector count in IVFFlatIndex")


class RebuildRequest(BaseModel):
    """Request model for POST /rebuild."""
    retrain_kmeans: bool = Field(
        default=False,
        description="If True, re-runs custom K-Means on active vectors; if False, reassigns vectors to existing centroids"
    )


class RebuildResponse(BaseModel):
    """Response model for POST /rebuild."""
    success: bool = Field(default=True, description="Whether rebuild completed successfully")
    message: str = Field(..., description="Status message")
    active_vectors: int = Field(..., description="Number of active vectors remaining after compaction")
    num_clusters: int = Field(..., description="Number of clusters in rebuilt index")
    retrain_kmeans: bool = Field(..., description="Whether centroids were retrained")
    latency_ms: float = Field(..., description="Rebuild duration in milliseconds")


class BenchmarkRequest(BaseModel):
    """Request model for running 500-query benchmark."""
    num_queries: int = Field(default=500, ge=1, le=1000)
    k: int = Field(default=10, ge=1, le=100)
    nprobes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])


class BenchmarkMetric(BaseModel):
    """Metrics for a single nprobe configuration."""
    nprobe: int
    recall_at_10: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_vectors_examined: float
    speedup: float


class BenchmarkResponse(BaseModel):
    """Response for benchmark evaluation."""
    total_queries: int
    k: int
    exact_latency_avg_ms: float
    exact_latency_p50_ms: float
    exact_latency_p95_ms: float
    exact_latency_p99_ms: float
    ivf_metrics: list[BenchmarkMetric]
