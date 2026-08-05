# Synthetix AI Detector Engine

Production-oriented AI text detection engine built on DeBERTa-v3 sequence classification, probability calibration, and metric analysis.

> [!IMPORTANT]
> **Production Guidance & Usage Limitations**
>
> AI detection probabilities are statistical estimations. Scores must be evaluated alongside domain-specific benchmark calibration (ECE <= 5%) before being relied upon for automated decision workflows.

## Features

- **DeBERTa-v3 Classification**: Modern transformer fine-tuned on GPT-4, Claude 3, Llama 3, and Mistral outputs.
- **Metric Analysis**: Computes sentence burstiness (CV) and predictability indices.
- **Score Calibration**: Evaluates Expected Calibration Error (ECE) and reliability diagrams.
- **FastAPI Engine**: Standardized REST endpoints (`/api/analyze`, `/health`).
- **Interactive UI**: Web interface for sentence breakdown and probability visualization.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e ".[dev,benchmark]"
   ```

2. Start the API server:
   ```bash
   python server.py
   ```
   Or using Uvicorn:
   ```bash
   uvicorn server:app --host 127.0.0.1 --port 8000
   ```

3. Health Check:
   ```bash
   curl http://127.0.0.1:8000/health
   ```

4. Web Interface:
   Navigate to `http://localhost:8000` in your web browser.

## Benchmark & Calibration

Validate accuracy and score calibration against JSONL datasets:

```bash
# Validate corpus schema
python benchmark/evaluate.py --corpus benchmark/corpus/sample_corpus.jsonl --dry-run

# Run full evaluation
python benchmark/evaluate.py --corpus benchmark/corpus/sample_corpus.jsonl --api-url http://localhost:8000/api/analyze

# Generate calibration reliability plot
python benchmark/calibration.py --report benchmark/reports/report_latest.json
```

## Testing

Run unit tests:
```bash
pytest
```

## License

[MIT License](LICENSE)
