# Synthetix AI Detector Benchmark Corpus

This directory contains the dataset files, schema definitions, and workflow scripts for constructing, generating, and rebuilding the Synthetix AI Detector benchmark corpus.

---

## 1. Corpus Files Overview

| File | Description | Count | Sources / Labels |
|---|---|---|---|
| `manifest_schema.json` | JSON Schema for validating corpus sample objects | - | Schema Definition |
| `human_samples.jsonl` | Original human-written text samples (150-500 words) | 20 | Human-authored (`human`) |
| `ai_samples.jsonl` | Synthetic AI-generated text samples via local Ollama API | 20 | Llama 3 (`ai`) |
| `train.jsonl` | Training dataset split (60% of merged corpus) | 24 | Shuffled Human + AI |
| `test.jsonl` | Evaluation & testing dataset split (40% of merged corpus) | 16 | Shuffled Human + AI |
| `sample_corpus.jsonl` | Initial infrastructure validation placeholder corpus | 10 | Placeholder Samples |

---

## 2. Text Domains & Balance

The benchmark dataset maintains domain parity between human and synthetic AI samples to prevent topic confounding bias:

- **Academic Essays (`essay`):** History, plant biology, literature critique, political history, quantum physics.
- **Casual Blog Posts (`blog`):** Home coffee brewing, remote office setup, weekend hiking trips, mindfulness routines.
- **Business Emails (`email`):** Project timeline updates, Q4 infrastructure budget requests, design system rollout notes.
- **Personal Narratives (`narrative`):** Childhood memoirs, moving to a new city, car restoration projects.
- **Technical Documentation (`technical`):** Redis caching patterns, Docker multi-stage builds, PostgreSQL GIN indexing.
- **News Articles (`news`):** Public transit expansion votes, regional manufacturing supply chains, coastal wind farm energy records.

---

## 3. Generating AI Text Samples

AI samples are generated using `benchmark/generate_ai_samples.py`, which communicates directly with a local [Ollama](https://ollama.com/) instance.

### Prerequisites
Ensure Ollama is running locally:
```bash
ollama serve
```

### Usage & Arguments
```bash
# Generate 20 AI samples using default model (llama3)
python benchmark/generate_ai_samples.py --model llama3 --count 20 --output benchmark/corpus/ai_samples.jsonl

# Dry-run mode: View prompts and target domain distribution without calling Ollama API
python benchmark/generate_ai_samples.py --dry-run
```

### Options
- `--model`: Ollama model tag (default: `llama3`).
- `--count`: Number of AI samples to generate (default: `20`).
- `--output`: File path to save generated JSONL (default: `benchmark/corpus/ai_samples.jsonl`).
- `--ollama-url`: URL of the local Ollama instance (default: `http://localhost:11434`).
- `--dry-run`: Display prompt strategy without making API requests.

---

## 4. Rebuilding the Benchmark Corpus Splits

Once `human_samples.jsonl` and `ai_samples.jsonl` are present, use `benchmark/build_corpus.py` to merge, shuffle, split, and compute summary metrics.

### Command
```bash
python benchmark/build_corpus.py --human benchmark/corpus/human_samples.jsonl --ai benchmark/corpus/ai_samples.jsonl --output-dir benchmark/corpus --train-ratio 0.6 --seed 42
```

### Process
1. Loads human and AI JSONL records.
2. Shuffles the dataset using a fixed random seed (`--seed 42`) for reproducibility.
3. Splits data into `train.jsonl` (60%) and `test.jsonl` (40%).
4. Displays formatted summary statistics breaking down label, domain, and model family distribution across splits.

---

## 5. Validating & Benchmarking the Corpus

To validate corpus formatting prior to evaluation, run `evaluate.py` in dry-run mode:

```bash
# Validate dataset format & schema compliance
python benchmark/evaluate.py --corpus benchmark/corpus/test.jsonl --dry-run

# Run full evaluation against local Synthetix API
python benchmark/evaluate.py --corpus benchmark/corpus/test.jsonl --api-url http://localhost:8000/api/analyze
```
