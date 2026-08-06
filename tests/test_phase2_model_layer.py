import pytest
from synthetix.signals.lexical_regularity_heuristic import compute_lexical_regularity_score
from synthetix.signals.span_detector import detect_mixed_authorship_spans

def test_compute_lexical_regularity_score_short():
    res = compute_lexical_regularity_score("Too short text")
    assert res["lexical_regularity_score"] is None
    assert res["lexical_regularity_flagged"] is False

def test_compute_lexical_regularity_score_long():
    text = "The collapse of the Roman Republic was a prolonged erosion of institutional norms that began decades before Julius Caesar crossed the Rubicon. Client-patron networks transformed into competitive instruments of political warfare."
    res = compute_lexical_regularity_score(text)
    assert res["lexical_regularity_score"] is not None
    assert 0.0 <= res["lexical_regularity_score"] <= 1.0

def test_detect_mixed_authorship_spans_no_shift():
    sentence_scores = [
        {"sentence": "Sentence one.", "ai_score": 10.0},
        {"sentence": "Sentence two.", "ai_score": 15.0},
        {"sentence": "Sentence three.", "ai_score": 12.0}
    ]
    res = detect_mixed_authorship_spans(sentence_scores)
    assert res["has_transitions"] is False
    assert len(res["transitions"]) == 0

def test_detect_mixed_authorship_spans_with_shift():
    sentence_scores = [
        {"sentence": "Human written sentence.", "ai_score": 10.0},
        {"sentence": "AI generated sentence with high probability.", "ai_score": 95.0}
    ]
    res = detect_mixed_authorship_spans(sentence_scores)
    assert res["has_transitions"] is True
    assert len(res["transitions"]) == 1
    assert res["transitions"][0]["shift_type"] == "human_to_ai"
