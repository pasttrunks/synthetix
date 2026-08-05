from unittest.mock import patch, MagicMock
import pytest
import server
from server import TextPayload, analyze, score_text_with_transformer

VALID_LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog in the middle of the afternoon. "
    "It was a bright sunny day in the quiet and beautiful park. "
    "Everyone enjoyed the peaceful and serene atmosphere throughout the entire day."
)

def test_overall_ai_score_uses_only_transformer(monkeypatch):
    """Test overall_ai_score uses only transformer score and ignores Ollama score."""
    monkeypatch.setattr(server, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 82.5)
    
    # Even if score_text_with_ollama returns a different score, overall_ai_score must be 82.5
    monkeypatch.setattr(server, "score_text_with_ollama", lambda text: 99.0)
    monkeypatch.setattr(server, "check_ollama_alive", lambda: True)

    payload = TextPayload(text=VALID_LONG_TEXT)
    result = analyze(payload)

    assert result.overall_ai_score == 82.5
    assert "DeBERTa-v3 Transformer" in result.analysis_method

def test_score_clamping_upper_bound(monkeypatch):
    """Test score clamping when transformer returns > 100.0 (e.g. 150.0)."""
    monkeypatch.setattr(server, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 150.0)

    payload = TextPayload(text=VALID_LONG_TEXT)
    result = analyze(payload)

    assert result.overall_ai_score == 100.0

def test_score_clamping_lower_bound(monkeypatch):
    """Test score clamping when transformer returns < 0.0 (e.g. -20.0)."""
    monkeypatch.setattr(server, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: -20.0)

    payload = TextPayload(text=VALID_LONG_TEXT)
    result = analyze(payload)

    assert result.overall_ai_score == 0.0

def test_model_loaded_false_overall_ai_score_none(monkeypatch):
    """Test that when MODEL_LOADED is False, overall_ai_score is None."""
    monkeypatch.setattr(server, "MODEL_LOADED", False)

    payload = TextPayload(text=VALID_LONG_TEXT)
    result = analyze(payload)

    assert result.overall_ai_score is None
    assert result.analysis_method == "unavailable"
    assert result.model_name == "Unavailable"

def test_analysis_method_field_for_model_states(monkeypatch):
    """Test analysis_method field is correct for both loaded and unavailable model states."""
    payload = TextPayload(text=VALID_LONG_TEXT)

    # State 1: MODEL_LOADED = True
    monkeypatch.setattr(server, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "score_text_with_transformer", lambda text: 50.0)
    result_true = analyze(payload)
    assert "DeBERTa-v3 Transformer" in result_true.analysis_method

    # State 2: MODEL_LOADED = False
    monkeypatch.setattr(server, "MODEL_LOADED", False)
    result_false = analyze(payload)
    assert result_false.analysis_method == "unavailable"
    assert result_false.model_name == "Unavailable"
