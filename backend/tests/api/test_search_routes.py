import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _payload(modality: str) -> dict:
    return {
        "records": [
            {
                "chunk_id": f"{modality}_exact",
                "modality": modality,
                "histogram": [1, 0],
                "metadata": {"source": "fixture"},
            },
            {
                "chunk_id": f"{modality}_other",
                "modality": modality,
                "histogram": [0, 1],
            },
        ],
        "query_histogram": [1, 0],
        "k": 1,
    }


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint should report that the API is ready."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("path", "modality", "expected_score"),
    [
        ("/search/text", "text", 1.0),
        ("/search/music", "audio", 0.0),
        ("/search/visual", "image", 0.0),
    ],
)
def test_search_endpoints_return_ranked_results(
    path: str,
    modality: str,
    expected_score: float,
) -> None:
    """Search endpoints should return the nearest matching chunk."""
    response = client.post(path, json=_payload(modality))

    assert response.status_code == 200
    result = response.json()[0]
    assert result["chunk_id"] == f"{modality}_exact"
    assert result["score"] == pytest.approx(expected_score)
    assert result["metadata"] == {"source": "fixture"}


def test_visual_endpoint_rejects_audio_records() -> None:
    """Endpoints should reject records from another modality."""
    response = client.post("/search/visual", json=_payload("audio"))

    assert response.status_code == 422
    assert "image histograms" in response.json()["detail"]


def test_search_request_rejects_invalid_top_k() -> None:
    """Request validation should reject non-positive top-k values."""
    payload = _payload("text")
    payload["k"] = 0

    response = client.post("/search/text", json=payload)

    assert response.status_code == 422
