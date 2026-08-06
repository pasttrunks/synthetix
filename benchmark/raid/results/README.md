# RAID Baseline Evaluation Results

## Result summary (unchanged artifacts)

The frozen RAID baseline (200 samples, 100 human + 100 AI, non-adversarial,
domains: abstracts, news, reviews, wiki) was evaluated through the live pinned
Synthetix API at the existing 0.50 threshold. The artifacts in this directory
are the unmodified outputs of that run.

| Metric | Value |
|---|---|
| AUROC | 0.7563 |
| AUPRC | 0.8154 |
| Confusion matrix | TP 51, FP 12, TN 88, FN 49 |
| Precision / Recall / F1 | 0.8095 / 0.51 / 0.6258 |
| Brier | 0.2635 |
| ECE | 0.2608 |
| Coverage / abstentions / errors | 100% / 0 / 0 |

## Status

- **Primary-model acceptance failed.** The pinned HC3 RoBERTa baseline does not
  meet an acceptance bar for real-world use: recall is 0.51 at the 0.50
  threshold, ECE is 0.26, and base-model generators (e.g., mistral, cohere,
  mpt, gpt4) are largely or entirely missed.
- **No threshold, scoring logic, model, calibration, or release gate was
  changed** for this evaluation. The detector and corpus are exactly the frozen
  artifacts recorded in `report.json`.
- **RAID is an evaluated development benchmark, not an independent holdout.**
  It is drawn from the labeled `liamdugan/raid` train split, and any future
  model selection must not treat it as an unseen test set.
- **These results are not part of the CI release gate.** The gate continues to
  evaluate only the synthetic regression fixtures under `benchmark/corpus/`.

## Artifacts

- `report.json` — full metrics, per-domain and per-generator results,
  validation results, commit SHA, detector revision, and corpus SHA-256
- `predictions.jsonl` — 200 per-sample predictions
- `false_positives.jsonl` — 12 rows
- `false_negatives.jsonl` — 49 rows
- `score_distribution.json` — human and AI score summaries
