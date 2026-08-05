# Synthetix AI Text Detection Engine

Research prototype for AI text detection built on sequence classification, metric analysis, and score calibration benchmarking.

> [!WARNING]
> **Experimental Research Prototype**
>
> Synthetix is an exploratory research prototype. Raw detection scores represent uncalibrated baseline outputs from the `Hello-SimpleAI/chatgpt-detector-roberta` sequence classifier.
>
> **Do not use for adverse actions**, including academic grading, disciplinary accusations, hiring evaluations, or publishing decisions.

## Features

- **RoBERTa Sequence Classification**: Baseline model using `Hello-SimpleAI/chatgpt-detector-roberta` fine-tuned on HC3 question-answer pairs.
- **Metric Analysis**: Computes sentence burstiness (CV) and predictability indices.
- **Score Calibration Framework**: Measures Expected Calibration Error (ECE) and plots reliability diagrams.
- **FastAPI Engine**: Standardized REST endpoints (`/api/analyze`, `/health`).
- **Interactive Web Interface**: Single-page UI with sentence breakdown and neutral highlighting.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e ".[dev,benchmark]"
   ```

2. Start the API server:
   ```bash
   python server.py
   ```
   Or run the CLI command:
   ```bash
   synthetix-serve
   ```

3. Health Check:
   ```bash
   curl http://127.0.0.1:8000/health
   ```

4. Web Interface:
   Navigate to `http://localhost:8000` in your web browser.

## Benchmark & Calibration

Validate corpus schema and baseline evaluation metrics:

```bash
# Validate corpus schema
python benchmark/evaluate.py --corpus benchmark/corpus/essay_corpus.jsonl --dry-run

# Run full evaluation against local server
python benchmark/evaluate.py --corpus benchmark/corpus/essay_corpus.jsonl --api-url http://localhost:8000/api/analyze

# Generate calibration plot
python benchmark/calibration.py --report benchmark/reports/report_latest.json
```

## Testing

Run unit tests:
```bash
pytest tests/
```

## License

[MIT License](LICENSE)
