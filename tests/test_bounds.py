import pytest
import server

VALID_TEXT = (
    "The quick brown fox jumps over the lazy dog in the middle of the afternoon. "
    "It was a bright sunny day in the quiet and beautiful park. "
    "Everyone enjoyed the peaceful and serene atmosphere throughout the entire day."
)

def test_no_score_field_exceeds_100_or_below_0(client, patch_model_loaded_true, monkeypatch):
    """Test extreme transformer output values to ensure all returned score fields are within [0.0, 100.0]."""
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 500.0)

    res = client.post("/api/analyze", json={"text": VALID_TEXT})
    assert res.status_code == 200
    data = res.json()

    assert 0.0 <= data["overall_ai_score"] <= 100.0
    assert 0.0 <= data["predictability_index"] <= 100.0
    for s_score in data["sentence_scores"]:
        assert 0.0 <= s_score["ai_score"] <= 100.0

    # Test negative extreme
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: -500.0)
    res_neg = client.post("/api/analyze", json={"text": VALID_TEXT})
    assert res_neg.status_code == 200
    data_neg = res_neg.json()

    assert 0.0 <= data_neg["overall_ai_score"] <= 100.0
    assert 0.0 <= data_neg["predictability_index"] <= 100.0
    for s_score in data_neg["sentence_scores"]:
        assert 0.0 <= s_score["ai_score"] <= 100.0

def test_null_handling_when_models_unavailable(client, patch_model_loaded_false):
    """Test null/None handling for overall_ai_score and analysis_method when models are unavailable."""
    res = client.post("/api/analyze", json={"text": VALID_TEXT})
    assert res.status_code == 200
    data = res.json()

    assert data["overall_ai_score"] is None
    assert data["analysis_method"] == "unavailable"
    assert data["model_name"] == "Unavailable"

def test_very_long_input_just_under_limit(client, patch_model_loaded_true, monkeypatch):
    """Test text input of 49,999 characters (just under the 50,000 limit)."""
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 45.0)

    # 49,999 chars
    text = "word " * 9999 + "a" * (49999 - (5 * 9999))
    assert len(text) == 49999

    res = client.post("/api/analyze", json={"text": text})
    assert res.status_code == 200
    data = res.json()
    assert data["overall_ai_score"] == 45.0

def test_input_with_only_whitespace(client):
    """Test input with only spaces, tabs, and newlines returns 400."""
    res = client.post("/api/analyze", json={"text": "   \n\t  \r\n   "})
    assert res.status_code == 400
    assert "Text payload cannot be empty." in res.json()["detail"]

def test_input_with_only_punctuation(client, patch_model_loaded_true, monkeypatch):
    """Test input containing only punctuation marks (< 150 chars) returns abstention response."""
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 10.0)

    res = client.post("/api/analyze", json={"text": "!...???!!!"})
    assert res.status_code == 200
    data = res.json()
    assert data["overall_ai_score"] is None
    assert data["analysis_method"] == "insufficient_text"
