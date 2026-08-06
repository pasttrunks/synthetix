# Synthetix AI Detector Regression Fixture Corpus

This directory contains a **synthetic regression fixture suite** used to exercise the Synthetix pipeline and release gates. The passages are written by project authors to resemble academic prose. They are **not** genuine output from GPT-4, Claude, Llama, or Gemini, and they are **not** verified human academic sources. Scores on these fixtures do not establish real-world detection performance.

---

## 1. Corpus Files Overview

| File | Description | Count | Sources / Labels |
|---|---|---|---|
| `manifest_schema.json` | JSON Schema for validating corpus sample objects | - | Schema Definition |
| `human_samples.jsonl` | Human-style regression fixtures | 50 | `human_style` (`label=0`) |
| `ai_samples.jsonl` | AI-style regression fixtures | 50 | `gpt4_style`, `claude_style`, `llama_style`, `gemini_style` (`label=1`) |
| `train.jsonl` | Train split (60% of groups) | 60 | Group-isolated Human + AI-style |
| `test.jsonl` | Test split (40% of groups) | 40 | Group-isolated Human + AI-style |
| `sample_corpus.jsonl` | Initial infrastructure validation placeholder corpus | 10 | Placeholder Samples |

---

## 2. Text Domains & Balance

The fixture suite maintains domain parity between human-style and AI-style samples to prevent topic confounding in the regression gate:

- **Academic Essays (`essay`):** History, plant biology, literature critique, political history, quantum physics.

---

## 3. Regenerating the Fixture Suite

Regenerate the human-style and AI-style fixtures deterministically:
```bash
python -m benchmark.expand_essay_corpus
python -m benchmark.build_corpus
```

## 4. Validating & Benchmarking the Corpus

To validate corpus formatting prior to evaluation, run `evaluate.py` in dry-run mode:

```bash
# Validate dataset format & schema compliance
python benchmark/evaluate.py --corpus benchmark/corpus/test.jsonl --dry-run

# Run full evaluation against local Synthetix API
python benchmark/evaluate.py --corpus benchmark/corpus/test.jsonl --api-url http://localhost:8000/api/analyze
```
