#!/usr/bin/env python3
"""
Convert MBPP to supervised JSONL for v2 instruction tuning.

Input:
    data/external/mbpp.jsonl (original format)

Output:
    data/instruction_tuning/v2_raw/mbpp_supervised.jsonl

Filters:
    - Only tasks with <= 30 lines of code in function body
    - Only tasks with simple function definitions (single def)
    - Skip tasks with complex setup code

Format:
    Each line: {"id": str, "prompt": str, "solution": str}

Usage:
    python nas/tools/convert_mbpp_to_supervised.py
"""

import json
import re
from pathlib import Path


def extract_function_definition(code: str):
    """
    Extract function definition from MBPP code.

    Returns:
        (func_name, signature, body) or None if not extractable
    """
    # Normalize line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = code.split('\n')

    # Find the first 'def' line
    func_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            func_start = i
            break

    if func_start is None:
        return None

    # Extract function signature
    signature_line = lines[func_start].strip()

    # Extract function body (everything after the def line that is indented)
    body_lines = []
    base_indent = None

    for i in range(func_start + 1, len(lines)):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            continue

        # Detect base indentation from first non-empty line
        if base_indent is None:
            if line.startswith('\t') or line.startswith(' '):
                base_indent = len(line) - len(line.lstrip())
            else:
                # No indentation - function body ended
                break

        # Check if line belongs to function body
        if line.startswith('\t') or line.startswith(' '):
            current_indent = len(line) - len(line.lstrip())
            if current_indent >= base_indent:
                body_lines.append(line)
            else:
                # Dedented - function ended
                break
        else:
            # No indentation - function ended
            break

    # Extract function name from signature
    match = re.search(r'def\s+(\w+)\s*\(', signature_line)
    if not match:
        return None

    func_name = match.group(1)
    body = '\n'.join(body_lines) + '\n' if body_lines else ''

    return func_name, signature_line, body


def convert_mbpp_task(task: dict) -> dict:
    """
    Convert MBPP task to our supervised format.

    Args:
        task: MBPP task dict

    Returns:
        Supervised task dict or None if task should be filtered out.
    """
    task_id = task['task_id']
    text = task['text']
    code = task['code']

    # Extract function definition
    result = extract_function_definition(code)
    if result is None:
        return None

    func_name, signature, body = result

    # Filter: Skip if body is too long
    body_lines = [line for line in body.split('\n') if line.strip()]
    if len(body_lines) > 30:
        return None

    # Filter: Skip if body is too short (likely incomplete)
    if len(body_lines) < 2:
        return None

    # Build prompt: signature + docstring
    # MBPP doesn't have docstrings in code, so create from text
    docstring = f'    """{text}"""'
    prompt = f"{signature}\n{docstring}\n"

    # Solution is the body
    solution = body

    short_id = f"mbpp_{task_id}"

    return {
        'id': short_id,
        'prompt': prompt,
        'solution': solution
    }


def main():
    """Convert MBPP dataset to supervised format."""
    src = Path("data/external/mbpp.jsonl")
    dst = Path("data/instruction_tuning/v2_raw/mbpp_supervised.jsonl")

    if not src.exists():
        print(f"ERROR: {src} not found.")
        print("Download MBPP first:")
        print("  python nas/tools/download_datasets.py")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Convert all tasks (with filtering)
    tasks = []
    skipped_no_func = 0
    skipped_too_long = 0
    skipped_too_short = 0

    with open(src, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                task = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}")
                continue

            supervised_task = convert_mbpp_task(task)
            if supervised_task:
                tasks.append(supervised_task)
            else:
                # Count skip reasons (heuristic)
                code = task.get('code', '')
                if 'def ' not in code:
                    skipped_no_func += 1
                elif len(code.split('\n')) > 35:
                    skipped_too_long += 1
                else:
                    skipped_too_short += 1

    # Write to output
    with open(dst, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"[OK] Converted {len(tasks)} MBPP tasks")
    print(f"   Skipped: {skipped_no_func + skipped_too_long + skipped_too_short}")
    print(f"     - No function definition: {skipped_no_func}")
    print(f"     - Too long (>30 lines): {skipped_too_long}")
    print(f"     - Too short (<2 lines): {skipped_too_short}")
    print(f"   Output: {dst}")

    # Show sample
    if tasks:
        print("\nSample (first task):")
        sample = tasks[0]
        print(f"  ID: {sample['id']}")
        print(f"  Prompt ({len(sample['prompt'])} chars):")
        print(f"    {sample['prompt'][:80]}...")
        print(f"  Solution ({len(sample['solution'])} chars):")
        print(f"    {sample['solution'][:80]}...")

    return 0


if __name__ == "__main__":
    exit(main())
