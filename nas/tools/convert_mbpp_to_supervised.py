#!/usr/bin/env python3
"""
Convert MBPP to supervised JSONL for v2 instruction tuning.

Input:
    data/external/mbpp.jsonl (original format)

Output:
    data/instruction_tuning/v2_raw/mbpp_supervised.jsonl

Filters:
    - Only tasks with <= 30 lines of code
    - Only tasks marked as simple difficulty (if available)

Format:
    Each line: {"id": str, "prompt": str, "solution": str}

Usage:
    python nas/tools/convert_mbpp_to_supervised.py
"""

import json
from pathlib import Path


def convert_mbpp_task(task: dict) -> dict:
    """
    Convert MBPP task to our supervised format.

    Args:
        task: MBPP task dict with keys:
            - task_id: Integer ID
            - text: Problem description
            - code: Reference solution
            - test_list: List of test cases
            - (optional) difficulty: "simple", "medium", "hard"

    Returns:
        Supervised task dict with keys:
            - id: "mbpp_123"
            - prompt: Function signature + docstring (extracted from code)
            - solution: Implementation only

        Returns None if task should be filtered out.
    """
    task_id = task['task_id']
    text = task['text']
    code = task['code']

    # Filter: Skip if code is too long
    lines = code.strip().split('\n')
    if len(lines) > 30:
        return None

    # Filter: Skip if marked as difficult (if field exists)
    if 'difficulty' in task and task['difficulty'] not in ['simple', 'easy']:
        return None

    # Extract function signature and body
    # MBPP code typically includes full function definition
    # We need to split it into prompt (signature + docstring) and solution (body)

    # TODO: Proper parsing (for now, use full code as both prompt and solution)
    # This is a placeholder - needs proper implementation

    short_id = f"mbpp_{task_id}"

    # Placeholder: Extract first line as prompt, rest as solution
    # In real implementation, parse the function properly
    if 'def ' in code:
        # Find where docstring ends or first statement begins
        # For now, just use the whole code as prompt and solution separately
        # TODO: Implement proper splitting
        prompt = code  # Placeholder
        solution = code  # Placeholder
    else:
        # Skip non-function code
        return None

    return {
        'id': short_id,
        'prompt': prompt,
        'solution': solution,
        'description': text  # Keep original description for reference
    }


def main():
    """Convert MBPP dataset to supervised format."""
    src = Path("data/external/mbpp.jsonl")
    dst = Path("data/instruction_tuning/v2_raw/mbpp_supervised.jsonl")

    if not src.exists():
        print(f"ERROR: {src} not found.")
        print("Download MBPP first:")
        print("  wget https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl")
        print("  mv mbpp.jsonl data/external/")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Convert all tasks (with filtering)
    tasks = []
    skipped = 0
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line.strip())
            supervised_task = convert_mbpp_task(task)
            if supervised_task:
                tasks.append(supervised_task)
            else:
                skipped += 1

    # Write to output
    with open(dst, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"✅ Converted {len(tasks)} MBPP tasks (skipped {skipped})")
    print(f"   Output: {dst}")

    # Show sample
    if tasks:
        print("\nSample (first task):")
        sample = tasks[0]
        print(f"  ID: {sample['id']}")
        print(f"  Description: {sample.get('description', 'N/A')[:50]}...")
        print(f"  Prompt length: {len(sample['prompt'])} chars")
        print(f"  Solution length: {len(sample['solution'])} chars")

    print("\n⚠️  NOTE: This script needs proper parsing implementation.")
    print("   Currently uses placeholder logic for prompt/solution splitting.")

    return 0


if __name__ == "__main__":
    exit(main())
