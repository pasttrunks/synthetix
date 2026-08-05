import pytest
from server import get_ai_label_index

class DummyConfig:
    def __init__(self, id2label):
        self.id2label = id2label

def test_binary_config_fake_real():
    """Test binary config {0: 'Fake', 1: 'Real'} and {0: 'Real', 1: 'Fake'}."""
    config1 = DummyConfig({0: "Fake", 1: "Real"})
    assert get_ai_label_index(config1) == 0

    config2 = DummyConfig({0: "Real", 1: "Fake"})
    assert get_ai_label_index(config2) == 1

def test_binary_config_human_ai():
    """Test binary config {0: 'Human', 1: 'AI'} and {0: 'AI', 1: 'Human'}."""
    config1 = DummyConfig({0: "Human", 1: "AI"})
    assert get_ai_label_index(config1) == 1

    config2 = DummyConfig({0: "AI", 1: "Human"})
    assert get_ai_label_index(config2) == 0

def test_binary_config_label_0_label_1():
    """Test default HuggingFace/DeBERTa config {0: 'LABEL_0', 1: 'LABEL_1'}."""
    config = DummyConfig({0: "LABEL_0", 1: "LABEL_1"})
    assert get_ai_label_index(config) == 1

def test_multiclass_config_human_other_ai_generated():
    """Test multi-class config {0: 'Human', 1: 'Other', 2: 'AI Generated'} returns index 2."""
    config = DummyConfig({0: "Human", 1: "Other", 2: "AI Generated"})
    assert get_ai_label_index(config) == 2

def test_config_only_ai_like_labels():
    """Test config with only AI-like labels (e.g. {0: 'Synthetic', 1: 'Generated'})."""
    config = DummyConfig({0: "Synthetic", 1: "Generated"})
    assert get_ai_label_index(config) == 0

def test_config_no_recognizable_labels():
    """Test config with no recognizable labels returns default index 0."""
    config1 = DummyConfig({0: "ClassA", 1: "ClassB"})
    assert get_ai_label_index(config1) == 0

    config2 = DummyConfig({})
    assert get_ai_label_index(config2) == 0

    class EmptyConfig:
        pass
    assert get_ai_label_index(EmptyConfig()) == 0
