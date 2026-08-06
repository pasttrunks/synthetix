import html
from typing import Dict, Any

def generate_html_review_report(analysis: Dict[str, Any]) -> str:
    """Generate self-contained HTML review report artifact for educator inspection."""
    score = analysis.get("overall_ai_score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"overall_ai_score must be numeric (int/float) or None, got {type(score).__name__}")
        score_str = html.escape(f"{score}%")
    else:
        score_str = "N/A (Uncertain)"
    model_name = html.escape(str(analysis.get("model_name", "Unknown Model")))
    model_revision = html.escape(str(analysis.get("model_revision") or "unknown"))
    method_desc = html.escape(str(analysis.get("analysis_method", "Standard Evaluation")))

    balanced = analysis.get("balanced_review") or {}
    balanced_html = ""
    if balanced:
        hc3_score = html.escape(str(balanced.get("hc3_score", "n/a")))
        dk_score = html.escape(str(balanced.get("desklib_score", "n/a")))
        hc3_rev = html.escape(str(balanced.get("hc3_model_revision", "unknown")))
        dk_rev = html.escape(str(balanced.get("desklib_model_revision", "unknown")))
        agreement = html.escape(str(balanced.get("agreement_status", "unknown")))
        outcome = html.escape(str(balanced.get("review_outcome", "unknown")))
        outcome_label = {
            "strong_ai_signal": "Strong agreement",
            "low_ai_signal": "Low agreement",
            "uncertain_disagreement": "Uncertain disagreement",
        }.get(outcome, outcome)
        balanced_html = f"""
    <div class="card">
        <h2>Balanced Review</h2>
        <div class="meta">Agreement: {agreement} | Outcome: {outcome_label}</div>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
            <div class="card" style="flex: 1; min-width: 220px;">
                <h3>HC3 Fast Baseline</h3>
                <div class="score">Signal: {hc3_score}%</div>
                <div class="meta">Model: {html.escape(str(balanced.get('hc3_model_name', 'unknown')))}<br>Revision: {hc3_rev}</div>
            </div>
            <div class="card" style="flex: 1; min-width: 220px;">
                <h3>Desklib Academic Sensitive</h3>
                <div class="score">Signal: {dk_score}%</div>
                <div class="meta">Model: {html.escape(str(balanced.get('desklib_model_name', 'unknown')))}<br>Revision: {dk_rev}</div>
            </div>
        </div>
        <div class="disclaimer">
            <strong>Disagreement is inconclusive.</strong> When the two detectors disagree,
            no reliable classification can be made from this analysis. These are experimental
            writing signals, not probabilities or proof of AI authorship.
        </div>
    </div>
"""


    
    sentences_html = []
    for s in analysis.get("sentence_scores", []):
        text = html.escape(s.get("sentence", ""))
        ai_s = s.get("ai_score", 0.0)
        is_suspicious = s.get("is_suspicious", False)
        
        style = "background: rgba(244, 63, 94, 0.15); border-bottom: 1px solid rgba(244, 63, 94, 0.4);" if is_suspicious else ""
        sentences_html.append(f'<span style="{style}">{text}</span>')

    sentences_body = " ".join(sentences_html)

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Synthetix Writing Integrity Evidence Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0c10; color: #c5c6c7; margin: 0; padding: 2rem; }}
        .card {{ background: #1f2833; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #45a29e; }}
        h1, h2 {{ color: #66fcf1; margin-top: 0; }}
        .score {{ font-size: 2.5rem; font-weight: bold; color: #66fcf1; }}
        .meta {{ font-size: 0.9rem; color: #c5c6c7; margin-bottom: 1rem; }}
        .prose {{ line-height: 1.6; font-size: 1rem; background: #0b0c10; padding: 1rem; border-radius: 6px; border: 1px solid #45a29e; }}
        .disclaimer {{ background: rgba(251, 191, 36, 0.1); border: 1px solid #fbbf24; color: #fde68a; padding: 1rem; border-radius: 6px; font-size: 0.85rem; margin-top: 1rem; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Synthetix Writing Integrity Evidence Report</h1>
        <div class="meta">Engine: {model_name} | Revision: {model_revision} | Method: {method_desc}</div>
        <div class="score">Experimental AI-Writing Signal: {score_str}</div>
        <div class="disclaimer">
            <strong>Important Review Notice:</strong> This value is an experimental AI-writing signal from a text classification model, not a probability that the text was AI-written. It requires human review and must be evaluated alongside writing-process history and source material. Do not use it as sole grounds for misconduct actions.
        </div>
    </div>
{balanced_html}

    <div class="card">
        <h2>Document Evidence Breakdown</h2>
        <div class="prose">
            {sentences_body}
        </div>
    </div>
</body>
</html>"""
    return report_html
