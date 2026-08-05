#!/usr/bin/env python3
"""
Synthetix AI Detector Benchmark Evaluation Script
Loads JSONL benchmark corpus, evaluates against local API, computes performance metrics,
and saves timestamped report JSON.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
import requests
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix

def validate_sample(sample: Dict[str, Any], line_num: int) -> bool:
    required_fields = ["text", "label", "source", "domain", "model_family", "word_count", "language"]
    missing = [field for field in required_fields if field not in sample]
    if missing:
        raise ValueError(f"Line {line_num}: Missing required fields: {missing}")
    if sample["label"] not in ["human", "ai"]:
        raise ValueError(f"Line {line_num}: Invalid label '{sample['label']}' (must be 'human' or 'ai')")
    if not isinstance(sample["text"], str) or not sample["text"].strip():
        raise ValueError(f"Line {line_num}: 'text' must be a non-empty string")
    return True

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (binids == i)
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            weight = np.sum(mask) / total
            ece += weight * abs(bin_acc - bin_conf)
    return float(ece)

def compute_tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid_tprs = tpr[fpr <= target_fpr]
    if len(valid_tprs) > 0:
        return float(np.max(valid_tprs))
    return 0.0

def compute_metrics(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.array(y_prob, dtype=float)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return {"n_samples": 0}

    y_pred = (y_prob_arr >= threshold).astype(int)

    if len(np.unique(y_true_arr)) == 1:
        if y_true_arr[0] == 0:
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
            fn, tp = 0, 0
        else:
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
            tn, fp = 0, 0
    else:
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    accuracy = float((tp + tn) / n_samples)
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if len(np.unique(y_true_arr)) >= 2:
        auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
        auprc = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        auroc = 0.0
        auprc = 0.0

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    ece = compute_ece(y_true_arr, y_prob_arr, n_bins=10)
    tpr_at_01_fpr = compute_tpr_at_fpr(y_true_arr, y_prob_arr, target_fpr=0.001)
    tpr_at_1_fpr = compute_tpr_at_fpr(y_true_arr, y_prob_arr, target_fpr=0.01)

    return {
        "n_samples": n_samples,
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "tpr_at_0_1_fpr": round(tpr_at_01_fpr, 4),
        "tpr_at_1_fpr": round(tpr_at_1_fpr, 4),
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "fpr": round(fpr_val, 4),
            "f1": round(f1, 4)
        }
    }

def print_summary_table(report: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("           SYNTHETIX AI DETECTOR BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Timestamp:         {report['timestamp']}")
    print(f"Corpus Path:       {report['corpus_path']}")
    print(f"Total Samples:     {report['total_samples']} (Evaluated: {report['eval_samples']}, Errors: {report['error_samples']})")
    print(f"Threshold:         {report['threshold']}")
    print("-" * 70)
    
    m = report["overall_metrics"]
    if m.get("n_samples", 0) > 0:
        cm = m["confusion_matrix"]
        print("OVERALL METRICS:")
        print(f"  AUROC:            {m['auroc']:.4f}")
        print(f"  AUPRC:            {m['auprc']:.4f}")
        print(f"  Brier Score:      {m['brier_score']:.4f}")
        print(f"  ECE (10 bins):    {m['ece']:.4f}")
        print(f"  TPR @ 0.1% FPR:   {m['tpr_at_0_1_fpr']:.4f}")
        print(f"  TPR @ 1.0% FPR:   {m['tpr_at_1_fpr']:.4f}")
        print(f"  Accuracy:         {cm['accuracy']:.4f} (TP:{cm['tp']} FP:{cm['fp']} TN:{cm['tn']} FN:{cm['fn']})")
        print(f"  Precision / Rec:  {cm['precision']:.4f} / {cm['recall']:.4f} (F1: {cm['f1']:.4f})")
        print(f"  Overall FPR:      {cm['fpr']:.4f}")
    else:
        print("No valid evaluation samples recorded.")

    print("\nPER-DOMAIN BREAKDOWN:")
    print(f"{'Domain':<12} | {'Count':<6} | {'AUROC':<7} | {'ECE':<7} | {'TPR@1%FPR':<9} | {'FPR':<7}")
    print("-" * 60)
    for domain, d_m in report.get("per_domain", {}).items():
        if d_m.get("n_samples", 0) > 0:
            cm = d_m["confusion_matrix"]
            print(f"{domain:<12} | {d_m['n_samples']:<6} | {d_m['auroc']:<7.4f} | {d_m['ece']:<7.4f} | {d_m['tpr_at_1_fpr']:<9.4f} | {cm['fpr']:<7.4f}")

    print("\nPER-MODEL-FAMILY BREAKDOWN:")
    print(f"{'Model Family':<14} | {'Count':<6} | {'AUROC':<7} | {'ECE':<7} | {'TPR@1%FPR':<9} | {'FPR':<7}")
    print("-" * 62)
    for mf, mf_m in report.get("per_model_family", {}).items():
        if mf_m.get("n_samples", 0) > 0:
            cm = mf_m["confusion_matrix"]
            print(f"{mf:<14} | {mf_m['n_samples']:<6} | {mf_m['auroc']:<7.4f} | {mf_m['ece']:<7.4f} | {mf_m['tpr_at_1_fpr']:<9.4f} | {cm['fpr']:<7.4f}")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Synthetix AI Detector Benchmark Evaluation Script")
    parser.add_argument("--corpus", type=str, default="benchmark/corpus/sample_corpus.jsonl", help="Path to benchmark JSONL corpus file")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/api/analyze", help="Synthetix API analyze endpoint URL")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold (0.0 - 1.0)")
    parser.add_argument("--output-dir", type=str, default="benchmark/reports", help="Directory to save JSON evaluation report")
    parser.add_argument("--dry-run", action="store_true", help="Validate corpus format without making API requests")

    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file not found at '{args.corpus}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus from: {args.corpus}")
    samples = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                validate_sample(data, idx)
                samples.append(data)
            except Exception as e:
                print(f"Corpus Validation Error (line {idx}): {e}", file=sys.stderr)
                sys.exit(1)

    print(f"Successfully loaded and validated {len(samples)} samples from corpus.")

    if args.dry_run:
        print("\n[DRY RUN MODE] Corpus structure validation PASSED. No API calls performed.")
        sys.exit(0)

    # Evaluate samples via API
    results = []
    error_count = 0

    print(f"Evaluating {len(samples)} samples against Synthetix API at {args.api_url}...")

    for idx, sample in enumerate(samples, 1):
        payload = {"text": sample["text"]}
        try:
            res = requests.post(args.api_url, json=payload, timeout=30)
            if res.status_code == 200:
                res_data = res.json()
                raw_score = res_data.get("overall_ai_score")
                if raw_score is None:
                    y_prob = 0.0
                else:
                    y_prob = float(raw_score) / 100.0
                
                y_true = 1 if sample["label"] == "ai" else 0
                results.append({
                    "sample_index": idx,
                    "text_snippet": sample["text"][:80],
                    "label": sample["label"],
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "domain": sample["domain"],
                    "model_family": sample["model_family"],
                    "source": sample["source"],
                    "raw_api_response": res_data
                })
            else:
                print(f"Warning: Sample {idx} API request failed with HTTP {res.status_code}: {res.text[:100]}")
                error_count += 1
        except Exception as e:
            print(f"Warning: Sample {idx} API request failed with exception: {e}")
            error_count += 1

    eval_count = len(results)
    print(f"Completed evaluation: {eval_count} successful, {error_count} errors.")

    y_true_all = [r["y_true"] for r in results]
    y_prob_all = [r["y_prob"] for r in results]

    overall_metrics = compute_metrics(y_true_all, y_prob_all, threshold=args.threshold)

    # Per-domain metrics
    per_domain_metrics = {}
    domains = set(r["domain"] for r in results)
    for d in sorted(domains):
        d_sub = [r for r in results if r["domain"] == d]
        per_domain_metrics[d] = compute_metrics([r["y_true"] for r in d_sub], [r["y_prob"] for r in d_sub], threshold=args.threshold)

    # Per-model-family metrics
    per_model_metrics = {}
    model_families = set(r["model_family"] for r in results)
    for mf in sorted(model_families):
        mf_sub = [r for r in results if r["model_family"] == mf]
        per_model_metrics[mf] = compute_metrics([r["y_true"] for r in mf_sub], [r["y_prob"] for r in mf_sub], threshold=args.threshold)

    now_utc = datetime.now(timezone.utc).isoformat()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": now_utc,
        "corpus_path": os.path.abspath(args.corpus),
        "api_url": args.api_url,
        "total_samples": len(samples),
        "eval_samples": eval_count,
        "error_samples": error_count,
        "threshold": args.threshold,
        "overall_metrics": overall_metrics,
        "per_domain": per_domain_metrics,
        "per_model_family": per_model_metrics,
        "predictions": [
            {
                "y_true": r["y_true"],
                "y_prob": r["y_prob"],
                "domain": r["domain"],
                "model_family": r["model_family"]
            } for r in results
        ]
    }

    os.makedirs(args.output_dir, exist_ok=True)
    report_filename = f"report_{timestamp_str}.json"
    report_path = os.path.join(args.output_dir, report_filename)
    latest_path = os.path.join(args.output_dir, "report_latest.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary_table(report)
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
