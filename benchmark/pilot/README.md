# Honest Pilot Set (Untouched)

This directory holds a small **untouched pilot set** intended to give the first
useful indication of whether the detector works beyond its tuned regression
fixtures. It is deliberately kept outside `benchmark/corpus/` and is **not**
used by the CI release gate.

## Composition

| File | Count | Content |
|---|---|---|
| `human_pilot.jsonl` | 50 | Public-domain human passages, one per distinct Project Gutenberg work (41 distinct authors) |
| `ai_pilot.jsonl` | 50 | Genuine GPT-2 generated passages (one model family) |
| `manifest.json` | - | Freeze metadata and generation provenance |

## Freeze rules

These rules were applied when the set was created and remain in force:

1. **No detector scores were inspected before the files were frozen.** Only
   word count, language, and boilerplate contamination were checked.
2. **No sample will be replaced or edited based on detector output.** If the
   detector performs poorly on a pilot sample, that is evidence, not a reason
   to curate the sample away.
3. **The pilot set is not part of the tuned fixture suite** and must not be
   used to tune the model, thresholds, or fixtures.

## Human passages

Sourced from Project Gutenberg public-domain works. Every record stores the
Gutenberg ID, title, author, license, collection date, and a content hash.
The Gutenberg header was verified to match the recorded title and author
before the passage was accepted. Passages are 313-418 words.

## AI passages

Generated locally with `openai-community/gpt2` pinned to revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` using `transformers` on CPU.
Every record stores the prompt, temperature, top-p, seed, token limits,
generation date, and the exact model revision. Text is genuine model output;
it has not been hand-edited. One model family is used initially by design.

## Known limitations

- GPT-2 is a small base model; passage quality and topical coherence vary.
- Human passages come from a mix of literary, philosophical, political, and
  scientific public-domain works, not a single academic register.
- This set is a first signal, not a validated benchmark.
