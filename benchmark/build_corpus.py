#!/usr/bin/env python3
"""
Synthetix Corpus Builder
Reads human_samples.jsonl and ai_samples.jsonl, groups samples by source_group_id,
splits groups deterministically into train (60%) and test (40%) sets without data leakage,
and writes full essay_corpus.jsonl, train.jsonl, and test.jsonl manifests.
"""

import os
import sys
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any
from benchmark.corpus_registry import load_and_validate_corpus

def print_dataset_summary(title: str, samples: List[Dict[str, Any]]):
    print(f"\n{'=' * 60}")
    print(f"  {title.upper()} (Total: {len(samples)} samples)")
    print(f"{'=' * 60}")
    
    label_counts = Counter(s.get("label", "unknown") for s in samples)
    domain_counts = Counter(s.get("domain", "unknown") for s in samples)

    print("LABEL DISTRIBUTION:")
    for label, count in sorted(label_counts.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {str(label):<10}: {count:>4} ({pct:>5.1f}%)")

    print("\nDOMAIN DISTRIBUTION:")
    for domain, count in sorted(domain_counts.items()):
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {domain:<10}: {count:>4} ({pct:>5.1f}%)")
    print(f"{'=' * 60}")

def main():
    parser = argparse.ArgumentParser(description="Merge human and AI samples into group-isolated train and test splits")
    parser.add_argument("--human", type=str, default="benchmark/corpus/human_samples.jsonl", help="Path to human_samples.jsonl")
    parser.add_argument("--ai", type=str, default="benchmark/corpus/ai_samples.jsonl", help="Path to ai_samples.jsonl")
    parser.add_argument("--output-dir", type=str, default="benchmark/corpus", help="Output directory for merged datasets")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Ratio of source groups to allocate to train set (default 0.6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group assignment")

    args = parser.parse_args()

    human_samples, h_summary = load_and_validate_corpus(args.human)
    ai_samples, a_summary = load_and_validate_corpus(args.ai)

    all_samples = human_samples + ai_samples
    print(f"Loaded {len(human_samples)} human and {len(ai_samples)} AI samples (Total: {len(all_samples)}).")

    # Group by source_group_id to prevent topic/source leakage
    grouped = defaultdict(list)
    for sample in all_samples:
        group_id = sample.get("source_group_id", sample.get("domain", "default"))
        grouped[group_id].append(sample)

    group_keys = sorted(list(grouped.keys()))
    random.seed(args.seed)
    random.shuffle(group_keys)

    n_train_groups = int(len(group_keys) * args.train_ratio)
    train_groups = set(group_keys[:n_train_groups])
    test_groups = set(group_keys[n_train_groups:])

    train_samples = [s for g in train_groups for s in grouped[g]]
    test_samples = [s for g in test_groups for s in grouped[g]]

    print(f"Group Split: {len(train_groups)} train groups ({len(train_samples)} samples), {len(test_groups)} test groups ({len(test_samples)} samples).")

    os.makedirs(args.output_dir, exist_ok=True)
    
    full_path = os.path.join(args.output_dir, "essay_corpus.jsonl")
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    with open(full_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")

    print_dataset_summary("Combined Essay Corpus", all_samples)
    print(f"Saved full corpus to '{full_path}', train split to '{train_path}', test split to '{test_path}'.")

if __name__ == "__main__":
    main()
