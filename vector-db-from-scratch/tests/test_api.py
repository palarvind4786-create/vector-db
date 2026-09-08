"""
Comprehensive integration tests for the FastAPI Vector Database REST API.
Tests all required endpoints:
- POST /vectors
- POST /search (exact and ivf modes, vector and semantic text queries)
- DELETE /vectors/{id}
- GET /stats
- POST /rebuild
And verifies clean error handling and input validation.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import index_manager


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance with initialized indices."""
    with TestClient(app) as test_client:
        yield test_client


class TestStatsEndpoint:
    """Tests for GET /stats."""

    def test_get_stats_success(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()

        assert "total_vectors" in data
        assert "active_vectors" in data
        assert "dimension" in data
        assert data["dimension"] == 384
        assert data["metric"] == "cosine"
        assert "num_clusters" in data
        assert data["num_clusters"] == 100
        assert data["is_trained"] is True
        assert data["active_vectors"] > 0
        assert "cluster_distribution" in data

    def test_get_stats_via_api_prefix(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["active_vectors"] > 0


class TestSearchEndpoint:
    """Tests for POST /search across exact and IVF modes."""

    def test_search_ivf_text_query(self, client):
        response = client.post(
            "/search",
            json={
                "query": "space exploration astronomy satellite",
                "k": 5,
                "mode": "ivf",
                "nprobe": 8,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "ivf"
        assert data["k"] == 5
        assert data["count"] <= 5
        assert len(data["results"]) == data["count"]
        assert data["nprobe"] == 8
        assert data["latency_ms"] >= 0.0
        assert data["candidates_examined"] > 0
        assert isinstance(data["probed_clusters"], list)
        assert len(data["probed_clusters"]) == 8

        # Verify ranking order
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

        for res in data["results"]:
            assert "id" in res
            assert "score" in res
            assert "rank" in res
            assert "metadata" in res

    def test_search_exact_text_query(self, client):
        response = client.post(
            "/search",
            json={
                "query": "championship soccer match tournament",
                "k": 3,
                "mode": "exact",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "exact"
        assert data["k"] == 3
        assert data["count"] == 3
        assert data["nprobe"] is None
        assert data["candidates_examined"] >= data["count"]

    def test_search_with_vector_input(self, client):
        # Generate a random 384-d normalized vector
        rng = np.random.default_rng(42)
        vec = rng.normal(size=384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        response = client.post(
            "/search",
            json={
                "query": vec.tolist(),
                "k": 4,
                "mode": "ivf",
                "nprobe": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 4

    def test_search_validation_errors(self, client):
        # Invalid mode
        res_bad_mode = client.post(
            "/search",
            json={"query": "test", "mode": "unsupported_mode"},
        )
        assert res_bad_mode.status_code == 400

        # Empty string query
        res_empty_q = client.post(
            "/search",
            json={"query": "   ", "mode": "ivf"},
        )
        assert res_empty_q.status_code == 400

        # Wrong vector dimension
        res_bad_dim = client.post(
            "/search",
            json={"query": [0.1, 0.2, 0.3], "mode": "exact"},
        )
        assert res_bad_dim.status_code == 400


class TestVectorsCrudAndRebuild:
    """Tests for POST /vectors, DELETE /vectors/{id}, and POST /rebuild."""

    def test_insert_single_vector_with_array(self, client):
        rng = np.random.default_rng(123)
        vec = rng.normal(size=384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        test_id = "test_custom_vec_001"
        response = client.post(
            "/vectors",
            json={
                "id": test_id,
                "vector": vec.tolist(),
                "metadata": {"test_tag": "pytest_integration"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["inserted_count"] == 1
        assert data["id"] == test_id
        assert data["ids"] == [test_id]
        assert data["dim"] == 384

        # Search for it via exact search using identical vector
        search_res = client.post(
            "/search",
            json={"query": vec.tolist(), "k": 1, "mode": "exact"},
        )
        assert search_res.status_code == 200
        top_match = search_res.json()["results"][0]
        assert top_match["id"] == test_id
        assert np.isclose(top_match["score"], 1.0, atol=1e-4)

    def test_insert_single_vector_with_text(self, client):
        test_id = "test_text_doc_002"
        response = client.post(
            "/vectors",
            json={
                "id": test_id,
                "text": "Quantum computing algorithms utilizing topological qubits for error correction.",
                "metadata": {"category": "Sci/Tech", "field": "Quantum"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["inserted_count"] == 1
        assert data["id"] == test_id

        # Search semantically for quantum computing
        search_res = client.post(
            "/search",
            json={
                "query": "topological qubits quantum computing error correction",
                "k": 5,
                "mode": "ivf",
                "nprobe": 10,
            },
        )
        assert search_res.status_code == 200
        result_ids = [r["id"] for r in search_res.json()["results"]]
        assert test_id in result_ids

    def test_delete_vector_and_exclusion_from_search(self, client):
        del_id = "test_custom_vec_001"
        # Delete the vector
        del_res = client.delete(f"/vectors/{del_id}")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["deleted"] is True
        assert del_data["id"] == del_id

        # Stats should show tombstone count >= 1
        stats_res = client.get("/stats")
        assert stats_res.json()["tombstone_count"] >= 1

        # Search for it again - should NOT be returned
        rng = np.random.default_rng(123)
        vec = rng.normal(size=384).astype(np.float32)
        vec /= np.linalg.norm(vec)

        search_res = client.post(
            "/search",
            json={"query": vec.tolist(), "k": 10, "mode": "exact"},
        )
        assert search_res.status_code == 200
        res_ids = [r["id"] for r in search_res.json()["results"]]
        assert del_id not in res_ids

    def test_delete_nonexistent_vector_returns_404(self, client):
        res = client.delete("/vectors/non_existent_vector_id_xyz")
        assert res.status_code == 404

    def test_rebuild_index_purges_tombstones(self, client):
        rebuild_res = client.post("/rebuild", json={"retrain_kmeans": False})
        assert rebuild_res.status_code == 200
        data = rebuild_res.json()
        assert data["success"] is True
        assert data["active_vectors"] > 0
        assert data["num_clusters"] == 100

        # After compaction, tombstone count should be 0
        stats_res = client.get("/stats")
        assert stats_res.json()["tombstone_count"] == 0
