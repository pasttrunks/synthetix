import pytest

from synthetix.model_backends import (
    BACKENDS,
    DEFAULT_BACKEND,
    DesklibAcademicBackend,
    HC3RobertaBackend,
    create_backend,
    get_ai_label_index,
    resolve_backend_name,
)


def test_default_backend_is_hc3_roberta():
    assert DEFAULT_BACKEND == "hc3_roberta"
    assert resolve_backend_name() == "hc3_roberta"


def test_registry_contains_both_backends():
    assert set(BACKENDS) == {"hc3_roberta", "desklib_academic"}


def test_create_backend_hc3():
    b = create_backend("hc3_roberta")
    assert isinstance(b, HC3RobertaBackend)
    assert b.name == "hc3_roberta"
    assert b.model_revision == "d2b342c61775d5dd0221808a79983ed3b86ffd86"
    assert b.tokenizer_revision == b.model_revision


def test_create_backend_desklib_pinned_revision():
    b = create_backend("desklib_academic")
    assert isinstance(b, DesklibAcademicBackend)
    assert b.name == "desklib_academic"
    assert b.model_revision == "fe9b4da50ee2cca5c877d607640681609170e363"
    assert b.tokenizer_revision == b.model_revision


def test_create_backend_unknown_raises():
    with pytest.raises(ValueError):
        create_backend("not_a_backend")


def test_backend_metadata_shape():
    b = create_backend("hc3_roberta")
    meta = b.metadata()
    assert meta["backend_name"] == "hc3_roberta"
    assert meta["model_name"] == "Hello-SimpleAI/chatgpt-detector-roberta"
    assert meta["model_revision"]
    assert meta["tokenizer_revision"]
    assert meta["inference_device"] in ("cpu", "cuda")


def test_label_mapping_via_backend_module():
    class DummyConfig:
        def __init__(self, id2label):
            self.id2label = id2label

    assert get_ai_label_index(DummyConfig({0: "Fake", 1: "Real"})) == 0
    assert get_ai_label_index(DummyConfig({0: "Real", 1: "Fake"})) == 1
    assert get_ai_label_index(DummyConfig({0: "Human", 1: "AI"})) == 1
    assert get_ai_label_index(DummyConfig({0: "LABEL_0", 1: "LABEL_1"})) == 1
    assert get_ai_label_index(DummyConfig({0: "Synthetic", 1: "Generated"})) == 0
    assert get_ai_label_index(DummyConfig({})) == 0
