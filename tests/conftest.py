import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path so server can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Mock transformers before importing server to prevent model download during import
mock_config = MagicMock()
mock_config.id2label = {0: "Fake", 1: "Real"}

mock_tokenizer_instance = MagicMock()
mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

mock_model_instance = MagicMock()
mock_model_instance.config = mock_config
mock_model_instance.eval.return_value = mock_model_instance

class MockOutput:
    def __init__(self, logits):
        self.logits = logits

# Default logits where index 0 (Fake/AI) gets ~88% probability
mock_model_instance.side_effect = lambda **kwargs: MockOutput(torch.tensor([[2.0, -2.0]]))

with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer_instance), \
     patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model_instance):
    import server

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

@pytest.fixture(autouse=True)
def disable_ollama_requests(monkeypatch):
    """Ensure tests never call out to a live Ollama server by default."""
    monkeypatch.setattr(server, "check_ollama_alive", lambda: False)

@pytest.fixture
def mock_transformer_model(monkeypatch):
    """Provides a stubbed transformer model and tokenizer for server testing."""
    mock_tok = MagicMock()
    mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    
    mock_mod = MagicMock()
    mock_mod.config = MagicMock()
    mock_mod.config.id2label = {0: "Fake", 1: "Real"}
    mock_mod.eval.return_value = mock_mod
    mock_mod.side_effect = lambda **kwargs: MockOutput(torch.tensor([[2.0, -2.0]]))

    monkeypatch.setattr(server, "tokenizer", mock_tok)
    monkeypatch.setattr(server, "model", mock_mod)
    monkeypatch.setattr(server, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "AI_LABEL_INDEX", 0)
    return mock_tok, mock_mod

@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(server.app)

@pytest.fixture
def short_text():
    """Sample short text fixture (< 150 characters)."""
    return "Hello world"

@pytest.fixture
def medium_text():
    """Sample medium text fixture (3 sentences, > 150 characters)."""
    return (
        "The quick brown fox jumps over the lazy dog in the middle of the afternoon. "
        "It was a bright sunny day in the quiet and beautiful park. "
        "Everyone enjoyed the peaceful and serene atmosphere throughout the entire day."
    )

@pytest.fixture
def long_text():
    """Sample long text fixture (1000+ words)."""
    word = "word "
    return word * 1050

@pytest.fixture
def no_terminal_punct_text():
    """Sample text fixture without terminal punctuation (> 150 chars)."""
    return "This is a sentence without any terminal punctuation at the end " * 3

@pytest.fixture
def equal_length_sentences_text():
    """Sample text fixture with equal-length sentences."""
    return "One two three. Four five six. Seven eight nine."

@pytest.fixture
def patch_model_loaded_true(monkeypatch):
    """Fixture that patches MODEL_LOADED to True with working stub transformer score."""
    monkeypatch.setattr(server, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "AI_LABEL_INDEX", 0)

@pytest.fixture
def patch_model_loaded_false(monkeypatch):
    """Fixture that patches MODEL_LOADED to False."""
    monkeypatch.setattr(server, "MODEL_LOADED", False)
