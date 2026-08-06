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

def test_gate_ece_strict_threshold():
    from benchmark.gate import check_release_gates
    mock_report = {
        "overall_metrics": {
            "n_samples": 100,
            "auroc": 0.90,
            "ece": 0.12,  # > 0.05
            "tpr_at_1_fpr": 0.60,
            "confusion_matrix": {"fpr": 0.01}
        }
    }
    assert check_release_gates(mock_report) is False

def test_report_exporter_html_escaping():
    from synthetix.report_exporter import generate_html_review_report
    malicious_analysis = {
        "overall_ai_score": "<script>alert(1)</script>",
        "model_name": "<img src=x onerror=alert(2)>",
        "analysis_method": "Standard Evaluation",
        "sentence_scores": []
    }
    html_output = generate_html_review_report(malicious_analysis)
    assert "<script>" not in html_output
    assert "&lt;script&gt;" in html_output
    assert "<img" not in html_output
    assert "&lt;img" in html_output

