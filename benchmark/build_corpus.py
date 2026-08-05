#!/usr/bin/env python3
"""
Synthetix Corpus Builder
Reads human_samples.jsonl and ai_samples.jsonl, shuffles them deterministically,
splits them into train (60%) and test (40%) sets, and prints summary statistics.
"""

import os
import sys
import json
import random
import argparse
from collections import Counter
from typing import List, Dict, Any

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corpus file not found: '{filepath}'")
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(data)
            except Exception as e:
                raise ValueError(f"Error reading line {idx} in '{filepath}': {e}")
    return samples

def print_dataset_summary(title: str, samples: List[Dict[str, Any]]):
    print(f"\n{'=' * 60}")
    print(f"  {title.upper()} (Total: {len(samples)} samples)")
    print(f"{'=' * 60}")
    
    label_counts = Counter(s.get("label", "unknown") for s in samples)
    domain_counts = Counter(s.get("domain", "unknown") for s in samples)
    model_family_counts = Counter(s.get("model_family", "unknown") for s in samples)

    print("LABEL DISTRIBUTION:")
    for label, count in sorted(label_counts.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {label:<10}: {count:>4} ({pct:>5.1f}%)")

    print("\nDOMAIN DISTRIBUTION:")
    for domain, count in sorted(domain_counts.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {domain:<10}: {count:>4} ({pct:>5.1f}%)")

    print("\nMODEL FAMILY DISTRIBUTION:")
    for mf, count in sorted(model_family_counts.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {mf:<10}: {count:>4} ({pct:>5.1f}%)")
    print(f"{'=' * 60}")

def main():
    parser = argparse.ArgumentParser(description="Merge human and AI samples into train and test splits")
    parser.add_argument("--human", type=str, default="benchmark/corpus/human_samples.jsonl", help="Path to human_samples.jsonl")
    parser.add_argument("--ai", type=str, default="benchmark/corpus/ai_samples.jsonl", help="Path to ai_samples.jsonl")
    parser.add_argument("--output-dir", type=str, default="benchmark/corpus", help="Directory to save train.jsonl and test.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Ratio of dataset allocated to training (default: 0.6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling (default: 42)")

    args = parser.parse_args()

    print(f"Loading samples from:\n  Human: {args.human}\n  AI:    {args.ai}")
    human_samples = load_jsonl(args.human)
    ai_samples = load_jsonl(args.ai)

    print(f"Loaded {len(human_samples)} human samples and {len(ai_samples)} AI samples.")

    combined = human_samples + ai_samples
    if not combined:
        print("Error: No samples found to merge.", file=sys.stderr)
        sys.exit(1)

    # Deterministic shuffle
    random.seed(args.seed)
    random.shuffle(combined)

    # Train / Test split calculation
    n_train = int(len(combined) * args.train_ratio)
    train_samples = combined[:n_train]
    test_samples = combined[n_train:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nWrote {len(train_samples)} samples to {train_path}")
    print(f"Wrote {len(test_samples)} samples to {test_path}")

    # Summary statistics
    print_dataset_summary("Combined Corpus Overall", combined)
    print_dataset_summary("Train Set (60%)", train_samples)
    print_dataset_summary("Test Set (40%)", test_samples)

if __name__ == "__main__":
    main()
