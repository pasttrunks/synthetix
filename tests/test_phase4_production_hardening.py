import pytest
from fastapi.testclient import TestClient
from server import app, MODEL_LOADED, MODEL_NAME

client = TestClient(app)

def test_liveness_probe():
    response = client.get("/health/liveness")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

def test_readiness_probe():
    response = client.get("/health/readiness")
    if MODEL_LOADED:
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["model_name"] == MODEL_NAME
    else:
        assert response.status_code == 503

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
