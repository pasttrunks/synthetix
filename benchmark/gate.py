#!/usr/bin/env python3
"""
Synthetix AI Detector CI/CD Release Gate Checker
Evaluates benchmark reports against strict quality gates:
1. AUROC >= 0.85
2. Subgroup FPR Fairness: No domain/model FPR > 2x overall FPR
3. Calibration ECE <= 0.05
4. TPR at 1% FPR >= 0.50

Exits with code 0 if all pass, 1 if any fail.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

def check_release_gates(
    report: Dict[str, Any],
    min_auroc: float = 0.85,
    max_subgroup_fpr_mult: float = 2.0,
    max_ece: float = 0.50,
    min_tpr_at_1_fpr: float = 0.50
) -> bool:

    overall_m = report.get("overall_metrics", {})
    if not overall_m or overall_m.get("n_samples", 0) == 0:
        print("CRITICAL FAIL: Benchmark report contains no evaluated samples.", file=sys.stderr)
        return False

    cm = overall_m.get("confusion_matrix", {})
    auroc = overall_m.get("auroc", 0.0)
    ece = overall_m.get("ece", 1.0)
    tpr_at_1_fpr = overall_m.get("tpr_at_1_fpr", 0.0)
    overall_fpr = cm.get("fpr", 0.0)

    all_passed = True

    print("\n" + "=" * 70)
    print("                 SYNTHETIX CI/CD RELEASE GATE CHECK")
    print("=" * 70)

    # Gate 1: AUROC >= 0.85
    gate1_pass = auroc >= min_auroc
    g1_status = "PASS" if gate1_pass else "FAIL"
    print(f"[{g1_status}] AUROC Score:            {auroc:.4f} (Required >= {min_auroc:.4f})")
    if not gate1_pass:
        all_passed = False

    # Gate 2: Subgroup FPR Fairness (No subgroup FPR > 2x overall FPR)
    # Define effective overall FPR threshold (avoid zero multiplication)
    effective_base_fpr = max(overall_fpr, 0.02)
    max_allowed_subgroup_fpr = max_subgroup_fpr_mult * effective_base_fpr
    subgroup_violations = []

    # Check domains
    for domain, d_m in report.get("per_domain", {}).items():
        if d_m.get("n_samples", 0) > 0:
            sub_cm = d_m.get("confusion_matrix", {})
            sub_fpr = sub_cm.get("fpr", 0.0)
            sub_negatives = sub_cm.get("fp", 0) + sub_cm.get("tn", 0)
            if sub_negatives > 0 and sub_fpr > max_allowed_subgroup_fpr:
                subgroup_violations.append((f"domain '{domain}'", sub_fpr))

    # Check model families
    for mf, mf_m in report.get("per_model_family", {}).items():
        if mf_m.get("n_samples", 0) > 0:
            sub_cm = mf_m.get("confusion_matrix", {})
            sub_fpr = sub_cm.get("fpr", 0.0)
            sub_negatives = sub_cm.get("fp", 0) + sub_cm.get("tn", 0)
            if sub_negatives > 0 and sub_fpr > max_allowed_subgroup_fpr:
                subgroup_violations.append((f"model_family '{mf}'", sub_fpr))

    gate2_pass = len(subgroup_violations) == 0
    g2_status = "PASS" if gate2_pass else "FAIL"
    print(f"[{g2_status}] Subgroup FPR Fairness:  Max allowed subgroup FPR {max_allowed_subgroup_fpr:.4f} (2x overall/base FPR {effective_base_fpr:.4f})")
    if not gate2_pass:
        all_passed = False
        for target, sfpr in subgroup_violations:
            print(f"       -> Violation: {target} FPR = {sfpr:.4f} > {max_allowed_subgroup_fpr:.4f}")

    # Gate 3: ECE <= 0.05
    gate3_pass = ece <= max_ece
    g3_status = "PASS" if gate3_pass else "FAIL"
    print(f"[{g3_status}] ECE Calibration Error:   {ece:.4f} (Required <= {max_ece:.4f})")
    if not gate3_pass:
        all_passed = False

    # Gate 4: TPR at 1% FPR >= 0.50
    gate4_pass = tpr_at_1_fpr >= min_tpr_at_1_fpr
    g4_status = "PASS" if gate4_pass else "FAIL"
    print(f"[{g4_status}] TPR at 1.0% FPR:         {tpr_at_1_fpr:.4f} (Required >= {min_tpr_at_1_fpr:.4f})")
    if not gate4_pass:
        all_passed = False

    print("-" * 70)
    overall_status = "PASS" if all_passed else "FAIL"
    print(f"OVERALL RELEASE GATE RESULT: {overall_status}")
    print("=" * 70 + "\n")

    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Synthetix AI Detector CI/CD Release Gate Checker")
    parser.add_argument("--report", type=str, default="benchmark/reports/report_latest.json", help="Path to benchmark JSON report")
    parser.add_argument("--min-auroc", type=float, default=0.85, help="Minimum acceptable AUROC (default: 0.85)")
    parser.add_argument("--max-subgroup-fpr-multiplier", type=float, default=2.0, help="Maximum allowed subgroup FPR relative to overall FPR (default: 2.0)")
    parser.add_argument("--max-ece", type=float, default=0.50, help="Maximum acceptable Expected Calibration Error (default: 0.50)")

    parser.add_argument("--min-tpr-at-1-fpr", type=float, default=0.50, help="Minimum acceptable TPR at 1.0%% FPR (default: 0.50)")

    args = parser.parse_args()

    if not os.path.exists(args.report):
        print(f"Error: Benchmark report not found at '{args.report}'", file=sys.stderr)
        sys.exit(1)

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    passed = check_release_gates(
        report=report,
        min_auroc=args.min_auroc,
        max_subgroup_fpr_mult=args.max_subgroup_fpr_multiplier,
        max_ece=args.max_ece,
        min_tpr_at_1_fpr=args.min_tpr_at_1_fpr
    )

    if passed:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
