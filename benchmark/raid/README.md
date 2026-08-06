# RAID Baseline Subset (Frozen)

A frozen, non-adversarial baseline drawn from the labeled
[`liamdugan/raid`](https://huggingface.co/datasets/liamdugan/raid) dataset
(config `raid`, split `train`, file `train.csv`, source revision
`865cac74188466cb0c3b7574a10204007b57a459`).

> Note: RAID-test is unlabeled (it contains only `id` and `generation`), so
> this baseline uses the **labeled train split**. The subset is intended for
> honest out-of-fixture evaluation of the Synthetix detector, not for training.

## Composition

- 200 samples: 100 human + 100 AI
- Domains: `abstracts`, `news`, `reviews`, `wiki` — exactly 25 human and
  25 AI per domain
- No adversarial transformations: only `attack == "none"` rows
- AI samples balanced across all 11 generator families available per domain
  (`chatgpt`, `cohere`, `cohere-chat`, `gpt2`, `gpt3`, `gpt4`, `llama-chat`,
  `mistral`, `mistral-chat`, `mpt`, `mpt-chat`) — 3 families contribute 3
  samples each and 8 contribute 2 each per domain
- Deterministic seed: `42`

## Preserved fields

Each row keeps the RAID `id` (as `raid_id`), `source_id`, `domain`, `model`
(generator), `decoding`, `repetition_penalty`, `attack`, `label` (0 = human,
1 = AI, derived from `model == "human"`), the full `text` (RAID `generation`),
and a SHA-256 `content_hash` of the text.

## Freeze rules

1. The subset was frozen (committed) before any detector evaluation.
2. No samples will be replaced, filtered, reordered, or regenerated based on
   detector results.
3. No tuning of the model, threshold, scoring logic, calibration, or release
   gate is performed based on results from this baseline.

## Files

- `raid_baseline.jsonl` — the 200 frozen samples
- `manifest.json` — selection rules, composition, validation results, source
  revision, and the corpus SHA-256

Validation before freezing: 200 unique RAID IDs, 200 unique content hashes,
every stored hash matches its text, 100 human / 100 AI labels, and 25/25 per
domain.
