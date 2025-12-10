#!/usr/bin/env python3
"""
Merge HumanEval and MBPP supervised data, then split into train/val.

Input:
    data/instruction_tuning/v2_raw/humaneval_supervised.jsonl
    data/instruction_tuning/v2_raw/mbpp_supervised.jsonl

Output:
    data/instruction_tuning/v2_train.jsonl (500 samples)
    data/instruction_tuning/v2_val.jsonl (50 samples)

Strategy:
    - Target: 500 train + 50 val = 550 total
    - HumanEval: Use all 164 tasks
    - MBPP: Sample 386 from 879 tasks (keep it diverse)
    - Shuffle and split 90/10

Usage:
    python nas/tools/merge_and_split_data.py
"""

import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    """Load JSONL file."""
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def save_jsonl(tasks, path: Path):
    """Save tasks to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')


def main():
    """Merge and split datasets."""
    random.seed(42)  # Reproducible

    humaneval_path = Path("data/instruction_tuning/v2_raw/humaneval_supervised.jsonl")
    mbpp_path = Path("data/instruction_tuning/v2_raw/mbpp_supervised.jsonl")
    train_path = Path("data/instruction_tuning/v2_train.jsonl")
    val_path = Path("data/instruction_tuning/v2_val.jsonl")

    # Load datasets
    print("="*70)
    print("MERGING DATASETS")
    print("="*70)

    humaneval_tasks = load_jsonl(humaneval_path)
    print(f"HumanEval: {len(humaneval_tasks)} tasks")

    mbpp_tasks = load_jsonl(mbpp_path)
    print(f"MBPP (raw): {len(mbpp_tasks)} tasks")

    # Target: 550 total (500 train + 50 val)
    target_total = 550
    humaneval_count = len(humaneval_tasks)
    mbpp_sample_count = target_total - humaneval_count

    print(f"\nTarget: {target_total} total samples")
    print(f"  - HumanEval: {humaneval_count} (use all)")
    print(f"  - MBPP: {mbpp_sample_count} (sample from {len(mbpp_tasks)})")

    # Sample MBPP tasks
    if len(mbpp_tasks) > mbpp_sample_count:
        mbpp_sampled = random.sample(mbpp_tasks, mbpp_sample_count)
        print(f"\n[SAMPLE] Selected {len(mbpp_sampled)} MBPP tasks")
    else:
        mbpp_sampled = mbpp_tasks
        print(f"\n[WARN] MBPP has fewer tasks than needed, using all {len(mbpp_sampled)}")

    # Merge
    all_tasks = humaneval_tasks + mbpp_sampled
    print(f"\nMerged: {len(all_tasks)} tasks")

    # Shuffle
    random.shuffle(all_tasks)
    print("[SHUFFLE] Randomized task order")

    # Split 90/10
    val_count = 50
    train_count = len(all_tasks) - val_count

    train_tasks = all_tasks[:train_count]
    val_tasks = all_tasks[train_count:]

    print(f"\n[SPLIT] Train: {len(train_tasks)}, Val: {len(val_tasks)}")

    # Save
    save_jsonl(train_tasks, train_path)
    save_jsonl(val_tasks, val_path)

    print("="*70)
    print("SAVE COMPLETE")
    print("="*70)
    print(f"Train: {train_path} ({len(train_tasks)} samples)")
    print(f"Val: {val_path} ({len(val_tasks)} samples)")

    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    # Count source distribution
    humaneval_in_train = sum(1 for t in train_tasks if t['id'].startswith('he_'))
    mbpp_in_train = sum(1 for t in train_tasks if t['id'].startswith('mbpp_'))
    humaneval_in_val = sum(1 for t in val_tasks if t['id'].startswith('he_'))
    mbpp_in_val = sum(1 for t in val_tasks if t['id'].startswith('mbpp_'))

    print(f"Train split:")
    print(f"  - HumanEval: {humaneval_in_train}")
    print(f"  - MBPP: {mbpp_in_train}")

    print(f"\nVal split:")
    print(f"  - HumanEval: {humaneval_in_val}")
    print(f"  - MBPP: {mbpp_in_val}")

    # Average lengths
    avg_prompt_len = sum(len(t['prompt']) for t in all_tasks) / len(all_tasks)
    avg_solution_len = sum(len(t['solution']) for t in all_tasks) / len(all_tasks)

    print(f"\nAverage lengths:")
    print(f"  - Prompt: {avg_prompt_len:.1f} chars")
    print(f"  - Solution: {avg_solution_len:.1f} chars")

    return 0


if __name__ == "__main__":
    exit(main())
