#!/usr/bin/env python3
"""
Convert HumanEval to supervised JSONL for v2 instruction tuning.

Input:
    data/external/HumanEval.jsonl (original format)

Output:
    data/instruction_tuning/v2_raw/humaneval_supervised.jsonl

Format:
    Each line: {"id": str, "prompt": str, "solution": str}

Usage:
    python nas/tools/convert_humaneval_to_supervised.py
"""

import json
from pathlib import Path


def convert_humaneval_task(task: dict) -> dict:
    """
    Convert HumanEval task to our supervised format.

    Args:
        task: HumanEval task dict with keys:
            - task_id: "HumanEval/0"
            - prompt: Function signature + docstring
            - canonical_solution: Reference implementation
            - test: Test cases
            - entry_point: Function name

    Returns:
        Supervised task dict with keys:
            - id: "he_0" (shortened from HumanEval/0)
            - prompt: Function signature + docstring
            - solution: Implementation only (indented)
    """
    task_id = task['task_id']
    prompt = task['prompt']
    solution = task['canonical_solution']

    # Convert task_id: "HumanEval/0" -> "he_0"
    short_id = task_id.replace('HumanEval/', 'he_')

    # prompt already contains signature + docstring
    # solution needs proper indentation (should already have it)

    return {
        'id': short_id,
        'prompt': prompt,
        'solution': solution
    }


def main():
    """Convert HumanEval dataset to supervised format."""
    src = Path("data/external/HumanEval.jsonl")
    dst = Path("data/instruction_tuning/v2_raw/humaneval_supervised.jsonl")

    if not src.exists():
        print(f"ERROR: {src} not found.")
        print("Download HumanEval first:")
        print("  wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz")
        print("  gunzip HumanEval.jsonl.gz")
        print("  mv HumanEval.jsonl data/external/")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Convert all tasks
    tasks = []
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line.strip())
            supervised_task = convert_humaneval_task(task)
            tasks.append(supervised_task)

    # Write to output
    with open(dst, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"✅ Converted {len(tasks)} HumanEval tasks")
    print(f"   Output: {dst}")

    # Show sample
    if tasks:
        print("\nSample (first task):")
        sample = tasks[0]
        print(f"  ID: {sample['id']}")
        print(f"  Prompt length: {len(sample['prompt'])} chars")
        print(f"  Solution length: {len(sample['solution'])} chars")

    return 0


if __name__ == "__main__":
    exit(main())
