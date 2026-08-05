import pytest
from fastapi.testclient import TestClient
from server import app
from synthetix.ingest import filter_qualifying_prose, extract_text_from_bytes
from synthetix.report_exporter import generate_html_review_report

client = TestClient(app)

def test_filter_qualifying_prose():
    raw_text = """
    Page 1
    Header
    
    The collapse of the Roman Republic was a prolonged erosion of institutional norms that began decades before Julius Caesar crossed the Rubicon. Client-patron networks transformed into competitive instruments of political warfare.
    
    Footer 12
    """
    prose, stats = filter_qualifying_prose(raw_text)
    assert "Page 1" not in prose
    assert "Roman Republic" in prose
    assert stats["qualifying_word_count"] > 20

def test_extract_text_from_bytes():
    content = b"This is a test plain text document containing long form prose for analysis."
    prose, stats = extract_text_from_bytes(content, "test.txt")
    assert "plain text document" in prose
    assert stats["qualifying_word_count"] > 5

def test_generate_html_review_report():
    analysis = {
        "overall_ai_score": 15.5,
        "model_name": "RoBERTa-base Classifier",
        "analysis_method": "RoBERTa-base Classifier",
        "sentence_scores": [
            {"sentence": "Sample human sentence.", "ai_score": 5.0, "is_suspicious": False}
        ]
    }
    html_report = generate_html_review_report(analysis)
    assert "Synthetix Writing Integrity Evidence Report" in html_report
    assert "15.5%" in html_report
    assert "Sample human sentence." in html_report

def test_api_extract_text_endpoint():
    response = client.post(
        "/api/extract-text",
        files={"file": ("sample.txt", b"Long form academic prose extracted from uploaded file.", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.txt"
    assert "academic prose" in data["extracted_text"]

def test_api_export_report_endpoint():
    analysis = {
        "overall_ai_score": 85.0,
        "model_name": "RoBERTa-base Classifier",
        "sentence_scores": []
    }
    response = client.post("/api/export-report", json=analysis)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Synthetix Writing Integrity Evidence Report" in response.text
