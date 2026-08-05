#!/usr/bin/env python3
"""
Synthetix AI Detector Calibration Analysis Script
Generates reliability diagrams (calibration plots), computes ECE/MCE,
and determines if detector scores can be presented as true probabilities.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

def compute_calibration_details(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, Dict[str, Any]]:
    if len(y_true) == 0:
        return 0.0, 0.0, {"bin_confs": [], "bin_accs": [], "bin_counts": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    total_samples = len(y_true)
    ece = 0.0
    mce = 0.0

    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (binids == i)
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            err = abs(acc - conf)

            weight = count / total_samples
            ece += weight * err
            if err > mce:
                mce = err

            bin_confs.append(conf)
            bin_accs.append(acc)
        else:
            bin_center = float((bins[i] + bins[i+1]) / 2.0)
            bin_confs.append(bin_center)
            bin_accs.append(0.0)

    details = {
        "bin_edges": bins.tolist(),
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts
    }
    return float(ece), float(mce), details

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, ece: float, mce: float, details: Dict[str, Any], output_path: str, n_bins: int = 10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

    bins = details["bin_edges"]
    bin_confs = details["bin_confs"]
    bin_accs = details["bin_accs"]
    bin_counts = details["bin_counts"]

    # Top Plot: Reliability Diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    
    # Filter active bins for line plot
    active_mask = [c > 0 for c in bin_counts]
    active_confs = [bin_confs[i] for i in range(n_bins) if active_mask[i]]
    active_accs = [bin_accs[i] for i in range(n_bins) if active_mask[i]]

    if active_confs:
        ax1.plot(active_confs, active_accs, "s-", color="#2b5c8f", linewidth=2, markersize=8, label="Detector Calibration")

    # Bar representation of gaps
    bin_centers = [(bins[i] + bins[i+1]) / 2.0 for i in range(n_bins)]
    width = 1.0 / n_bins * 0.8
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax1.bar(bin_centers[i], bin_accs[i], width=width, color="#4a90e2", alpha=0.4, edgecolor="#2b5c8f")

    ax1.set_title("Synthetix AI Detector - Reliability Diagram", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted Probability (Confidence)", fontsize=11)
    ax1.set_ylabel("Empirical Frequency (True Positives)", fontsize=11)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True, linestyle=":", alpha=0.6)

    textstr = f"ECE: {ece:.4f} ({ece*100:.2f}%)\nMCE: {mce:.4f} ({mce*100:.2f}%)"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ccc")
    ax1.text(0.05, 0.90, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment="top", bbox=props)
    ax1.legend(loc="lower right")

    # Bottom Plot: Sample Count Distribution Histogram
    ax2.bar(bin_centers, bin_counts, width=width, color="#7ed321", alpha=0.7, edgecolor="#417505")
    ax2.set_xlabel("Predicted Probability Bins", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Calibration plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Synthetix AI Detector Calibration Analysis Script")
    parser.add_argument("--report", type=str, default="benchmark/reports/report_latest.json", help="Path to benchmark JSON report")
    parser.add_argument("--output", type=str, default="benchmark/reports/calibration_plot.png", help="Path to save output calibration plot PNG")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of calibration bins (default: 10)")

    args = parser.parse_args()

    if not os.path.exists(args.report):
        print(f"Error: Report file not found at '{args.report}'", file=sys.stderr)
        sys.exit(1)

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    predictions = report.get("predictions", [])
    if not predictions:
        print("Warning: No predictions list found in report. Attempting to use precomputed ECE.", file=sys.stderr)
        ece = report.get("overall_metrics", {}).get("ece", 0.0)
        mce = ece
        y_true, y_prob = np.array([]), np.array([])
        details = {"bin_edges": [], "bin_confs": [], "bin_accs": [], "bin_counts": []}
    else:
        y_true = np.array([p["y_true"] for p in predictions], dtype=int)
        y_prob = np.array([p["y_prob"] for p in predictions], dtype=float)
        ece, mce, details = compute_calibration_details(y_true, y_prob, n_bins=args.n_bins)

    print("\n" + "=" * 65)
    print("           SYNTHETIX CALIBRATION ANALYSIS REPORT")
    print("=" * 65)
    print(f"Report Evaluated:    {args.report}")
    print(f"Evaluated Samples:   {len(y_true)}")
    print(f"Calibration Bins:    {args.n_bins}")
    print(f"Expected Cal Error:  {ece:.4f} ({ece*100:.2f}%)")
    print(f"Maximum Cal Error:   {mce:.4f} ({mce*100:.2f}%)")
    print("-" * 65)

    if len(y_true) > 0:
        plot_reliability_diagram(y_true, y_prob, ece, mce, details, args.output, n_bins=args.n_bins)

    # Output recommendation
    if ece <= 0.05:
        recommendation = (
            "PASS - RECOMMENDATION: Detector scores are well-calibrated (ECE <= 5.0%).\n"
            "Raw output scores CAN be presented directly to end-users as posterior AI probabilities."
        )
    else:
        recommendation = (
            "FAIL - RECOMMENDATION: Detector scores are NOT well-calibrated (ECE > 5.0%).\n"
            "Raw scores CANNOT be presented as posterior probabilities without recalibration (e.g. Platt scaling / Isotonic regression).\n"
            "Display scores as uncalibrated suspicion indices or qualitative risk bands instead."
        )

    print("\nPROBABILITY PRESENTATION RECOMMENDATION:")
    print(recommendation)
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
