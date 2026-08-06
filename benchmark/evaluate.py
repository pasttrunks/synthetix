#!/usr/bin/env python3
"""
Synthetix AI Detector Benchmark Evaluation Script
Loads JSONL benchmark corpus, evaluates against local API, computes performance metrics
(with 95% bootstrap confidence intervals, abstention tracking, and corpus provenance),
and saves timestamped report JSON.
"""

import os
import sys
import json
import argparse
import subprocess
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
import requests
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix

def compute_corpus_hash(corpus_path: str) -> str:
    """Compute SHA256 hash of corpus file for provenance tracking."""
    sha = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def validate_sample(sample: Dict[str, Any], line_num: int) -> bool:
    required_fields = ["text", "label"]
    missing = [field for field in required_fields if field not in sample]
    if missing:
        raise ValueError(f"Line {line_num}: Missing required fields: {missing}")
    
    raw_label = str(sample["label"]).lower()
    if raw_label not in ["human", "ai", "0", "1"]:
        raise ValueError(f"Line {line_num}: Invalid label '{sample['label']}'")
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

def compute_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 200, confidence: float = 0.95) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Compute 95% bootstrap confidence intervals for key metrics."""
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return {
            "auroc_ci": (None, None),
            "auprc_ci": (None, None),
            "ece_ci": (None, None)
        }

    aurocs, auprcs, eces = [], [], []
    n = len(y_true)
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        yt_b, yp_b = y_true[indices], y_prob[indices]
        if len(np.unique(yt_b)) >= 2:
            aurocs.append(roc_auc_score(yt_b, yp_b))
            auprcs.append(average_precision_score(yt_b, yp_b))
            eces.append(compute_ece(yt_b, yp_b))

    alpha = (1.0 - confidence) / 2.0
    
    def get_bounds(arr):
        if len(arr) < 20:
            return None, None
        return float(np.percentile(arr, alpha * 100.0)), float(np.percentile(arr, (1.0 - alpha) * 100.0))

    auroc_low, auroc_high = get_bounds(aurocs)
    auprc_low, auprc_high = get_bounds(auprcs)
    ece_low, ece_high = get_bounds(eces)

    return {
        "auroc_ci": (round(auroc_low, 4) if auroc_low is not None else None, round(auroc_high, 4) if auroc_high is not None else None),
        "auprc_ci": (round(auprc_low, 4) if auprc_low is not None else None, round(auprc_high, 4) if auprc_high is not None else None),
        "ece_ci": (round(ece_low, 4) if ece_low is not None else None, round(ece_high, 4) if ece_high is not None else None)
    }

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
        auroc = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
        auprc = round(float(average_precision_score(y_true_arr, y_prob_arr)), 4)
    else:
        auroc = None
        auprc = None

    brier = round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4)
    ece = round(compute_ece(y_true_arr, y_prob_arr, n_bins=10), 4)
    tpr_at_01_fpr = round(compute_tpr_at_fpr(y_true_arr, y_prob_arr, target_fpr=0.001), 4)
    tpr_at_1_fpr = round(compute_tpr_at_fpr(y_true_arr, y_prob_arr, target_fpr=0.01), 4)

    ci_bounds = compute_bootstrap_ci(y_true_arr, y_prob_arr)

    return {
        "n_samples": n_samples,
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "ece": ece,
        "tpr_at_0_1_fpr": tpr_at_01_fpr,
        "tpr_at_1_fpr": tpr_at_1_fpr,
        "bootstrap_ci_95": ci_bounds,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "fpr": round(fpr_val, 4),
            "f1": round(f1, 4)
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Synthetix AI Detector Benchmark Evaluation Script")
    parser.add_argument("--corpus", type=str, default="benchmark/corpus/essay_corpus.jsonl", help="Path to benchmark JSONL corpus file")
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

    corpus_sha256 = compute_corpus_hash(args.corpus)
    git_commit_sha = get_git_commit_hash()

    results = []
    predictions = []
    abstained_count = 0
    error_count = 0
    model_revision = None

    print(f"Evaluating {len(samples)} samples against Synthetix API at {args.api_url}...")

    for idx, sample in enumerate(samples, 1):
        payload = {"text": sample["text"]}
        try:
            res = requests.post(args.api_url, json=payload, timeout=30)
            if res.status_code == 200:
                res_data = res.json()
                raw_score = res_data.get("overall_ai_score")
                
                raw_label = str(sample["label"]).lower()
                y_true = 1 if raw_label in ("ai", "1", "chatgpt") else 0

                if raw_score is None:
                    abstained_count += 1
                    y_prob = None
                else:
                    y_prob = float(raw_score) / 100.0
                    predictions.append({
                        "sample_id": sample.get("sample_id", f"sample_{idx}"),
                        "label": sample.get("label"),
                        "y_true": y_true,
                        "y_prob": y_prob,
                        "raw_score": float(raw_score),
                        "domain": sample.get("domain", "general"),
                        "fixture_family": sample.get("fixture_family", sample.get("model_family", sample.get("source", "unknown"))),
                        "source_group_id": sample.get("source_group_id", "default"),
                        "text_snippet": sample["text"][:80]
                    })

                if model_revision is None:
                    model_revision = res_data.get("model_revision") or "unknown"

                results.append({
                    "sample_index": idx,
                    "sample_id": sample.get("sample_id", f"sample_{idx}"),
                    "source_group_id": sample.get("source_group_id", sample.get("domain", "default")),
                    "text_snippet": sample["text"][:80],
                    "label": sample["label"],
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "domain": sample.get("domain", "general"),
                    "fixture_family": sample.get("fixture_family", sample.get("model_family", sample.get("source", "unknown"))),
                    "raw_api_response": res_data
                })
            else:
                error_count += 1
        except Exception as e:
            error_count += 1

    valid_results = [r for r in results if r["y_prob"] is not None]
    coverage_rate = round((len(valid_results) / len(samples)) * 100.0, 2) if len(samples) > 0 else 0.0

    print(f"Completed evaluation: {len(valid_results)} analyzed, {abstained_count} abstained, {error_count} errors.")
    print(f"Coverage Rate: {coverage_rate}%")
    print(f"Model revision: {model_revision}")

    y_true_eval = [r["y_true"] for r in valid_results]
    y_prob_eval = [r["y_prob"] for r in valid_results]

    overall_metrics = compute_metrics(y_true_eval, y_prob_eval, threshold=args.threshold)

    # Per-domain metrics
    per_domain_metrics = {}
    domains = set(r["domain"] for r in valid_results)
    for d in sorted(domains):
        d_sub = [r for r in valid_results if r["domain"] == d]
        per_domain_metrics[d] = compute_metrics([r["y_true"] for r in d_sub], [r["y_prob"] for r in d_sub], threshold=args.threshold)

    # Per-fixture-family metrics
    per_fixture_metrics = {}
    fixture_families = set(r["fixture_family"] for r in valid_results)
    for ff in sorted(fixture_families):
        ff_sub = [r for r in valid_results if r["fixture_family"] == ff]
        per_fixture_metrics[ff] = compute_metrics([r["y_true"] for r in ff_sub], [r["y_prob"] for r in ff_sub], threshold=args.threshold)

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha,
        "corpus_path": os.path.abspath(args.corpus),
        "corpus_sha256": corpus_sha256,
        "model_info": {
            "model_name": "Hello-SimpleAI/chatgpt-detector-roberta",
            "model_revision": model_revision,
            "tokenizer_revision": model_revision
        },
        "total_samples": len(samples),
        "eval_samples": len(valid_results),
        "abstained_samples": abstained_count,
        "error_samples": error_count,
        "coverage_rate_pct": coverage_rate,
        "threshold": args.threshold,
        "overall_metrics": overall_metrics,
        "per_domain": per_domain_metrics,
        "per_fixture_family": per_fixture_metrics
    }


    os.makedirs(args.output_dir, exist_ok=True)
    report_file = os.path.join(args.output_dir, "report_latest.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    predictions_file = os.path.join(args.output_dir, "predictions.jsonl")
    with open(predictions_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    timestamp_file = os.path.join(args.output_dir, f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    with open(timestamp_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Evaluation report successfully saved to '{report_file}' and '{timestamp_file}'.")
    print(f"Per-sample predictions saved to '{predictions_file}'.")

if __name__ == "__main__":
    main()
