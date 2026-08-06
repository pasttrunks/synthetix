# Pilot v0 Evaluation Results

## Result: the detector failed on this distribution

Pilot v0 evaluated the frozen pilot set against the live pinned Synthetix API
(`Hello-SimpleAI/chatgpt-detector-roberta` @ `d2b342c61775d5dd0221808a79983ed3b86ffd86`,
threshold 0.50, unchanged scoring and gate).

**The detector assigned a low signal to every sample.** All 50 human passages
and all 50 AI passages scored in the 0-10 band. Every GPT-2-small passage was
therefore a false negative (50/50), and no human passage was a false positive.

| Metric | Value |
|---|---|
| AUROC | 0.381 |
| AUPRC | 0.4925 |
| Brier | 0.4999 |
| ECE | 0.4996 |
| Confusion matrix | TP 0, FP 0, TN 50, FN 50 |
| Precision / Recall / F1 | 0.0 / 0.0 / 0.0 |
| Coverage / abstentions / errors | 100% / 0 / 0 |

## Why this matters

> **Pilot v0 compares mixed public-domain Gutenberg passages against GPT-2-small
> output and is not evidence of modern AI-detection accuracy.**

The result is expected given the detector's training distribution: the
HC3-trained RoBERTa baseline recognizes formulaic modern AI prose, not
2020-era base-model generation. It does not transfer to either the
19th-century public-domain register or GPT-2-small output. This confirms the
project's standing caveat that the green release gate reflects synthetic
fixture behavior, not real-world detection performance.

## Artifacts

- `report.json` — full metrics, validation results, commit SHA, detector
  revision, and SHA-256 hashes of both pilot files
- `predictions.jsonl` — 100 per-sample predictions
- `false_positives.jsonl` — 0 rows
- `false_negatives.jsonl` — 50 rows
- `score_distribution.json` — score summaries by true label

## Freeze notes

- The pilot files were not modified, replaced, filtered, reordered, or
  regenerated for this evaluation.
- No model, threshold, scoring logic, calibration, or release gate was changed.
- No tuning was performed based on these results.
