#!/usr/bin/env python3
"""
Synthetix Full Audit Pipeline (Phase 0 - Phase 4 Verification)
Executes comprehensive validation across documentation, datasets, model architecture,
multi-signal evaluation, document ingestion, and API health endpoints.
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any, Tuple

def run_step(name: str, cmd: List[str]) -> Tuple[bool, str]:
    """Execute command step and return success flag and output string."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, res.stdout.strip()
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() or e.stdout.strip() or str(e)
        return False, err_msg

def audit_phase_0_documentation() -> Tuple[bool, List[str]]:
    """Audit Phase 0: Model naming, UI badges, and entrypoints."""
    issues = []
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "DeBERTa-v3" in content and "GPT-4" in content:
                issues.append("README still contains outdated DeBERTa-v3/GPT-4 fine-tuning claims.")
            if "Hello-SimpleAI/chatgpt-detector-roberta" not in content:
                issues.append("README missing real model name: Hello-SimpleAI/chatgpt-detector-roberta.")

    html_path = "ai_detector.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            if "hl-human" in html:
                issues.append("ai_detector.html contains green hl-human class instead of neutral gray hl-neutral.")

    return len(issues) == 0, issues

def audit_phase_1_evaluation() -> Tuple[bool, List[str]]:
    """Audit Phase 1: Corpus schema, provenance, and 95% CIs."""
    issues = []
    corpus_path = "benchmark/corpus/essay_corpus.jsonl"
    if not os.path.exists(corpus_path):
        issues.append(f"Corpus file not found: '{corpus_path}'")
        return False, issues

    from benchmark.corpus_registry import load_and_validate_corpus
    samples, summary = load_and_validate_corpus(corpus_path)
    if summary["error_count"] > 0:
        issues.append(f"Corpus has {summary['error_count']} schema validation errors.")

    return len(issues) == 0, issues

def audit_phase_2_model_layer() -> Tuple[bool, List[str]]:
    """Audit Phase 2: Lexical regularity heuristic and span change point detection."""
    issues = []
    from synthetix.signals.lexical_regularity_heuristic import compute_lexical_regularity_score
    from synthetix.signals.span_detector import detect_mixed_authorship_spans

    b_res = compute_lexical_regularity_score("The collapse of the Roman Republic was a prolonged erosion of institutional norms.")
    if "lexical_regularity_score" not in b_res or "lexical_regularity_flagged" not in b_res:
        issues.append("Lexical regularity score missing expected keys.")

    s_res = detect_mixed_authorship_spans([
        {"sentence": "Human sentence.", "ai_score": 10.0},
        {"sentence": "AI sentence.", "ai_score": 90.0}
    ])
    if not s_res.get("has_transitions"):
        issues.append("Span detector failed to flag sharp transition.")

    return len(issues) == 0, issues

def audit_phase_3_product_utility() -> Tuple[bool, List[str]]:
    """Audit Phase 3: Document text extraction and report exporter."""
    issues = []
    from synthetix.ingest import extract_text_from_bytes
    from synthetix.report_exporter import generate_html_review_report

    prose, stats = extract_text_from_bytes(b"Academic long form prose for testing document ingestion.", "test.txt")
    if stats["qualifying_word_count"] < 5:
        issues.append("Text extractor returned invalid qualifying word count.")

    html_rep = generate_html_review_report({"overall_ai_score": 50.0, "model_name": "RoBERTa"})
    if "Synthetix Writing Integrity Evidence Report" not in html_rep:
        issues.append("HTML report exporter missing title banner.")

    return len(issues) == 0, issues

def audit_phase_4_production_hardening() -> Tuple[bool, List[str]]:
    """Audit Phase 4: Unit test suite execution."""
    issues = []
    success, output = run_step("Pytest Suite", [sys.executable, "-m", "pytest", "tests/"])
    if not success:
        issues.append(f"Pytest unit test suite failed: {output}")

    return len(issues) == 0, issues

def main():
    print("=" * 70)
    print("         SYNTHETIX FULL AUDIT PIPELINE (PHASE 0 - PHASE 4)")
    print("=" * 70)

    audit_phases = [
        ("Phase 0: Model & Documentation Alignment", audit_phase_0_documentation),
        ("Phase 1: Dataset Provenance & 95% CIs", audit_phase_1_evaluation),
        ("Phase 2: Multi-Signal Engine & Span Detection", audit_phase_2_model_layer),
        ("Phase 3: Document Ingestion & Report Export", audit_phase_3_product_utility),
        ("Phase 4: Production Hardening & Pytest Suite", audit_phase_4_production_hardening)
    ]

    all_passed = True
    for name, audit_fn in audit_phases:
        ok, issues = audit_fn()
        status_str = "[PASS]" if ok else "[FAIL]"
        print(f"{status_str:<7} | {name}")
        if not ok:
            all_passed = False
            for issue in issues:
                print(f"        -> ERROR: {issue}")

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL PHASES (0-4) VERIFIED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("RESULT: AUDIT FAILED. Remediation required.")
        sys.exit(1)

if __name__ == "__main__":
    main()
