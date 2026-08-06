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
        <div class="score">AI Probability Score: {score_str}</div>
        <div class="disclaimer">
            <strong>Important Review Notice:</strong> This score represents statistical probability estimation from an AI text classification model. Scores must be evaluated as supporting evidence alongside writing process history and source material. Do not use as sole grounds for misconduct actions.
        </div>
    </div>

    <div class="card">
        <h2>Document Evidence Breakdown</h2>
        <div class="prose">
            {sentences_body}
        </div>
    </div>
</body>
</html>"""
    return report_html
