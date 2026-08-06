import pytest
from pathlib import Path

from synthetix.model_backends import (
    BalancedReviewBackend,
    HC3_REVISION,
    DESKLIB_REVISION,
    create_backend,
)
from synthetix.report_exporter import generate_html_review_report

ROOT = Path(__file__).resolve().parent.parent


def read_ui():
    return (ROOT / "ai_detector.html").read_text(encoding="utf-8")


class FakeBackend:
    def __init__(self, score):
        self.score = score

    def score_text(self, text):
        return self.score


def make_review(hc3_score, desklib_score):
    backend = BalancedReviewBackend()
    backend.hc3 = FakeBackend(hc3_score)
    backend.desklib = FakeBackend(desklib_score)
    return backend.review_document("sample text")


def test_both_high_is_strong_signal():
    r = make_review(80.0, 90.0)
    assert r["review_outcome"] == "strong_ai_signal"
    assert r["agreement_status"] == "agree_high"


def test_both_low_is_low_signal():
    r = make_review(10.0, 20.0)
    assert r["review_outcome"] == "low_ai_signal"
    assert r["agreement_status"] == "agree_low"


def test_hc3_high_desklib_low_is_uncertain():
    r = make_review(70.0, 30.0)
    assert r["review_outcome"] == "uncertain_disagreement"
    assert r["agreement_status"] == "disagree"


def test_hc3_low_desklib_high_is_uncertain():
    r = make_review(30.0, 70.0)
    assert r["review_outcome"] == "uncertain_disagreement"
    assert r["agreement_status"] == "disagree"


def test_disagreement_never_converted_to_classification():
    r = make_review(80.0, 20.0)
    assert r["review_outcome"] == "uncertain_disagreement"
    # scores are preserved individually; they are never averaged
    assert r["hc3_score"] == 80.0
    assert r["desklib_score"] == 20.0
    assert r["hc3_score"] != r["desklib_score"]


def test_one_backend_failure_propagates():
    backend = BalancedReviewBackend()
    backend.hc3 = FakeBackend(50.0)

    class Boom:
        def score_text(self, text):
            raise RuntimeError("out of memory")

    backend.desklib = Boom()
    with pytest.raises(RuntimeError):
        backend.review_document("text")


def test_metadata_preserved():
    r = make_review(80.0, 90.0)
    required = [
        "hc3_score",
        "hc3_backend_name",
        "hc3_model_name",
        "hc3_model_revision",
        "desklib_score",
        "desklib_backend_name",
        "desklib_model_name",
        "desklib_model_revision",
        "agreement_status",
        "review_outcome",
        "hc3_elapsed_s",
        "desklib_elapsed_s",
        "total_elapsed_s",
    ]
    for key in required:
        assert key in r, key
    assert r["hc3_backend_name"] == "hc3_roberta"
    assert r["hc3_model_revision"] == HC3_REVISION
    assert r["desklib_backend_name"] == "desklib_academic"
    assert r["desklib_model_revision"] == DESKLIB_REVISION
    assert r["hc3_elapsed_s"] >= 0
    assert r["desklib_elapsed_s"] >= 0
    assert r["total_elapsed_s"] >= r["hc3_elapsed_s"] + r["desklib_elapsed_s"] - 0.001


def test_balanced_backend_in_registry():
    backend = create_backend("balanced_review")
    assert isinstance(backend, BalancedReviewBackend)
    assert backend.model_revision.startswith("hc3:")
    assert "desklib:" in backend.model_revision


def test_exported_disagreement_report_contains_evidence_and_warning():
    analysis = {
        "overall_ai_score": 30.0,
        "model_name": "Balanced Review",
        "model_revision": "hc3:abc;desklib:def",
        "balanced_review": {
            "hc3_score": 80.0,
            "hc3_backend_name": "hc3_roberta",
            "hc3_model_name": "Hello-SimpleAI/chatgpt-detector-roberta",
            "hc3_model_revision": "d2b342c61775d5dd0221808a79983ed3b86ffd86",
            "desklib_score": 20.0,
            "desklib_backend_name": "desklib_academic",
            "desklib_model_name": "desklib/ai-text-detector-academic-v1.01",
            "desklib_model_revision": "fe9b4da50ee2cca5c877d607640681609170e363",
            "agreement_status": "disagree",
            "review_outcome": "uncertain_disagreement",
        },
    }
    html = generate_html_review_report(analysis)
    assert "Balanced Review" in html
    assert "80.0%" in html
    assert "20.0%" in html
    assert "d2b342c61775d5dd0221808a79983ed3b86ffd86" in html
    assert "fe9b4da50ee2cca5c877d607640681609170e363" in html
    assert "disagree" in html
    assert "Disagreement is inconclusive" in html


def test_balanced_ui_hides_generic_single_score():
    html = read_ui()
    assert 'id="scoreHero"' in html
    assert "scoreHero.style.display = 'none'" in html


def test_balanced_ui_renders_both_model_scores_and_plain_outcome():
    html = read_ui()
    for text in [
        "HC3 Fast Baseline",
        "Desklib Academic Sensitive",
        "At/above the 0.50 threshold",
        "Below the 0.50 threshold",
        "Not a probability.",
        "Both detectors found an elevated AI-writing signal. This is not proof that the text was AI-written.",
        "Neither detector found an elevated AI-writing signal. This does not prove that the text was written by a human.",
        "The detectors disagree. This analysis is inconclusive.",
    ]:
        assert text in html, f"missing: {text}"
    # Ambiguous labels without the explanatory sentence must not be used.
    assert "Strong agreement" not in html
    assert "Low agreement" not in html
    assert "Uncertain disagreement" not in html


def test_sentence_color_cannot_contradict_displayed_score():
    html = read_ui()
    # Highlighting is driven by the displayed ai_score threshold, not a separate flag.
    assert "item.ai_score >= 50" in html
    assert "item.is_suspicious ? 'hl-ai'" not in html


def test_balanced_ui_hides_sentence_map():
    html = read_ui()
    assert "inspectorSection.style.display = 'none'" in html
    assert "per-sentence scores come from a single detector" in html
    assert "Exploratory sentence pattern" in html


def test_supplementary_metrics_are_labelled_non_authoritative():
    html = read_ui()
    assert "Burstiness (supplementary)" in html
    assert "Predictability (supplementary)" in html
    assert "do not determine AI authorship and do not control the Balanced outcome" in html


def test_mode_verification_banner():
    html = read_ui()
    assert '"Running: " + friendly' in html
    assert "Wrong detector mode is running. Close the existing Synthetix server and relaunch Balanced Review." in html
    assert "expected_backend" in html
