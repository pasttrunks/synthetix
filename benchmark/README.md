# Synthetix AI Detector Benchmark Framework

This directory contains the automated evaluation and calibration benchmark framework for the Synthetix AI Detection Engine.

> [!WARNING]
> **WARNING: DATASET DEPENDENCY**  
> The detector requires a real, diverse, and representative benchmark corpus before any empirical accuracy, precision, or calibration claims can be made. Evaluating against synthetic, trivial, or placeholder data will yield invalid metrics that do not reflect real-world performance.

---

## Purpose

The benchmark framework provides standardized tools to:
1. **Quantify Detection Accuracy:** Measure overall performance using classification metrics (AUROC, AUPRC, TPR at low FPR thresholds).
2. **Assess Calibration Quality:** Evaluate whether predicted AI scores correspond to true posterior probabilities using Expected Calibration Error (ECE).
3. **Audit Bias & Fairness:** Break down performance across text domains (e.g., essays, emails, code) and generative model families (e.g., GPT-4, Claude, Llama).
4. **Enforce Automated Quality Gates:** Block deployments in CI/CD pipelines if detection or calibration quality fails established reliability thresholds.

---

## Corpus Format (JSONL Manifest)

Benchmark datasets must be stored as **JSON Lines (`.jsonl`)** files, where each line represents a single test sample adhering to `benchmark/corpus/manifest_schema.json`.

### Schema Structure
```json
{
  "text": "Full text payload to analyze",
  "label": "human|ai",
  "source": "Dataset identifier or origin",
  "domain": "essay|email|code|blog|report|news|other",
  "model_family": "gpt4|claude|human|llama|gemini|other",
  "word_count": 150,
  "language": "en"
}
```

---

## How to Add New Test Data

1. **Collect & Verify Samples:** Gather clean, untruncated text samples from verified human authors and generative AI models.
2. **Format as JSONL:** Format each sample into a single JSON line with accurate metadata tags (`label`, `domain`, `model_family`, `word_count`).
3. **Validate Schema:** Run `python benchmark/evaluate.py --corpus benchmark/corpus/your_corpus.jsonl --dry-run` to ensure all entries strictly comply with the schema format.
4. **Place in Corpus Dir:** Store your file in `benchmark/corpus/`.

---

## How to Run an Evaluation

Ensure the Synthetix API server is running (`python server.py`), then execute:

```bash
# Validate corpus without calling API
python benchmark/evaluate.py --corpus benchmark/corpus/sample_corpus.jsonl --dry-run

# Run full evaluation against local server
python benchmark/evaluate.py --corpus benchmark/corpus/sample_corpus.jsonl --api-url http://localhost:8000/api/analyze --threshold 0.50
```

### CLI Arguments (`evaluate.py`)
- `--corpus`: Path to the `.jsonl` benchmark manifest (default: `benchmark/corpus/sample_corpus.jsonl`).
- `--api-url`: Target Synthetix API endpoint (default: `http://localhost:8000/api/analyze`).
- `--threshold`: Classification decision threshold (default: `0.50`).
- `--output-dir`: Directory for generated JSON reports (default: `benchmark/reports`).
- `--dry-run`: Validate corpus schema without hitting the API.

---

## How to Interpret the Metrics Report

The evaluation script generates a timestamped report in `benchmark/reports/report_YYYYMMDD_HHMMSS.json` containing:

1. **AUROC (Area Under ROC Curve):** Measures overall ranking capability across all thresholds. Target: $\ge 0.85$.
2. **AUPRC (Area Under Precision-Recall Curve):** Reflects detection quality under class imbalance.
3. **Brier Score:** Mean squared difference between predicted probabilities and binary labels (lower is better).
4. **ECE (Expected Calibration Error):** Difference between predicted confidence and actual accuracy across 10 probability bins. Target: $\le 0.05$ (5%).
5. **TPR at 0.1% FPR / 1.0% FPR:** True Positive Rate at strict False Positive Rate limits. Crucial for avoiding false accusations in academic/professional settings.
6. **Per-Domain & Per-Model Breakdown:** Pinpoints specific domains (e.g. code) or LLM models (e.g. Claude) where the detector degraded.

---

## Calibration Analysis & Release Gating

- **Run Calibration Analysis:**
  ```bash
  python benchmark/calibration.py --report benchmark/reports/report_latest.json
  ```
  Generates a reliability diagram (`calibration_plot.png`) and outputs whether raw scores can be presented as true probabilities.

- **Run CI/CD Release Gate:**
  ```bash
  python benchmark/gate.py --report benchmark/reports/report_latest.json
  ```
  Evaluates strict quality gates (AUROC $\ge 0.85$, Subgroup FPR $\le 2\times$ Overall, ECE $\le 0.05$, TPR@1% FPR $\ge 0.50$) and exits with code 0 (PASS) or code 1 (FAIL).
