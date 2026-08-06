from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

VALID_TEXT = (
    "The quick brown fox jumps over the lazy dog in the middle of the afternoon. "
    "It was a bright sunny day in the quiet and beautiful park. "
    "Everyone enjoyed the peaceful and serene atmosphere throughout the entire day."
)

def test_health_endpoint_returns_200(client):
    """Test GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_name" in data

def test_analyze_valid_text_returns_200_and_expected_structure(client, patch_model_loaded_true, monkeypatch):

    """Test POST /api/analyze with valid text returns HTTP 200 and full AnalysisResult schema."""
    import server
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 75.0)

    payload = {"text": VALID_TEXT}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "overall_ai_score" in data
    assert "burstiness_cv" in data
    assert "predictability_index" in data
    assert "phrase_count" in data
    assert "model_name" in data
    assert "analysis_method" in data
    assert "ollama_active" in data
    assert "sentence_scores" in data

    assert data["overall_ai_score"] == 75.0
    assert isinstance(data["sentence_scores"], list)
    assert len(data["sentence_scores"]) == 3

def test_analyze_exceeds_max_length_returns_413(client):
    """Test POST /api/analyze with text > 50,000 characters returns 413 payload too large."""
    long_payload = {"text": "a" * 50001}
    response = client.post("/api/analyze", json=long_payload)
    assert response.status_code == 413
    assert "Text length exceeds 50,000 character limit." in response.json()["detail"]

def test_analyze_short_text_abstention_response(client, patch_model_loaded_true):
    """Test POST /api/analyze with text < minimum length (150 chars) returns appropriate abstention response."""
    payload = {"text": "Hello world. This is too short."}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["overall_ai_score"] is None
    assert data["analysis_method"] == "insufficient_text"
    assert data["burstiness_cv"] == 0.0
    assert data["predictability_index"] == 0.0
    assert "below the minimum required length" in data["message"]

def test_analyze_empty_text_returns_400(client):
    """Test POST /api/analyze with empty text payload returns 400 Bad Request."""
    response = client.post("/api/analyze", json={"text": ""})
    assert response.status_code == 400
    assert "Text payload cannot be empty." in response.json()["detail"]

    response_spaces = client.post("/api/analyze", json={"text": "   \n\t   "})
    assert response_spaces.status_code == 400

def test_get_health_returns_expected_fields(client):
    """Test GET /api/health returns online status and health fields."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_loaded" in data
    assert "ollama_active" in data
    assert "ollama_model" in data
    assert "active_engine" in data

def test_get_root_serves_html(client):
    """Test GET / serves the index HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

def test_response_content_type_headers(client):
    """Test Content-Type headers for API endpoints and root route."""
    res_health = client.get("/api/health")
    assert "application/json" in res_health.headers.get("content-type", "")

    res_analyze = client.post("/api/analyze", json={"text": VALID_TEXT})
    assert "application/json" in res_analyze.headers.get("content-type", "")

    res_root = client.get("/")
    assert "text/html" in res_root.headers.get("content-type", "")
