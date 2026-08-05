import re
from typing import List, Dict, Any

def detect_mixed_authorship_spans(sentence_scores: List[Dict[str, Any]], delta_threshold: float = 40.0) -> Dict[str, Any]:
    """
    Detect change-points and transitions between human and AI writing styles in a document.
    Identifies specific sentence indexes where AI score sharply shifts.
    """
    if len(sentence_scores) < 2:
        return {
            "has_transitions": False,
            "transitions": [],
            "span_summary": "Document too short for change-point analysis."
        }

    transitions = []
    for i in range(1, len(sentence_scores)):
        prev_score = sentence_scores[i-1].get("ai_score", 0.0)
        curr_score = sentence_scores[i].get("ai_score", 0.0)
        delta = curr_score - prev_score

        if abs(delta) >= delta_threshold:
            shift_type = "human_to_ai" if delta > 0 else "ai_to_human"
            transitions.append({
                "transition_index": i,
                "sentence_before": sentence_scores[i-1].get("sentence", "")[:50],
                "sentence_after": sentence_scores[i].get("sentence", "")[:50],
                "delta": round(delta, 1),
                "shift_type": shift_type
            })

    return {
        "has_transitions": len(transitions) > 0,
        "transition_count": len(transitions),
        "transitions": transitions,
        "span_summary": f"Detected {len(transitions)} authorship transition point(s)." if transitions else "Uniform authorship style detected throughout document."
    }
