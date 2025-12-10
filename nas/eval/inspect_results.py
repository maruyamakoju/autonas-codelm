#!/usr/bin/env python3
"""
inspect_results.py

Analyze batch evaluation results from eval_playground.py.

Usage:
    python inspect_results.py eval/results.jsonl
    python inspect_results.py eval/results.jsonl --show_examples 10
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import argparse


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL results."""
    results = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_completion(completion: str) -> Dict[str, Any]:
    """Analyze a single completion."""
    # Count unique characters
    unique_chars = len(set(completion))

    # Check for repetition (simple heuristic: most common char ratio)
    if len(completion) > 0:
        char_counts = Counter(completion)
        most_common_char, most_common_count = char_counts.most_common(1)[0]
        repetition_ratio = most_common_count / len(completion)
    else:
        most_common_char = None
        repetition_ratio = 0.0

    # Check for valid Python tokens
    has_keywords = any(kw in completion for kw in [
        'def', 'class', 'if', 'for', 'while', 'import', 'return', 'self'
    ])
    has_operators = any(op in completion for op in ['=', '+', '-', '*', '/', '>', '<'])
    has_parens = '(' in completion and ')' in completion

    # Entropy (simple measure of randomness)
    char_counts = Counter(completion)
    total = len(completion)
    entropy = 0.0
    if total > 0:
        for count in char_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p ** 0.5)  # simplified entropy-like measure

    return {
        'length': len(completion),
        'unique_chars': unique_chars,
        'repetition_ratio': repetition_ratio,
        'most_common_char': repr(most_common_char) if most_common_char else None,
        'has_keywords': has_keywords,
        'has_operators': has_operators,
        'has_parens': has_parens,
        'entropy': entropy,
    }


def categorize_quality(analysis: Dict[str, Any]) -> str:
    """Categorize completion quality."""
    # High repetition = mode collapse
    if analysis['repetition_ratio'] > 0.8:
        return 'mode_collapse'

    # Very low unique chars = repetitive
    if analysis['unique_chars'] < 5:
        return 'repetitive'

    # Has Python-like structure
    if analysis['has_keywords'] and analysis['has_operators']:
        return 'good'

    # Has some structure but incomplete
    if analysis['has_parens'] or analysis['has_operators']:
        return 'partial'

    # Random noise
    return 'noise'


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    print("=" * 70)
    print(" Evaluation Results Summary")
    print("=" * 70)
    print(f"Total prompts: {len(results)}")
    print()

    # Timing stats
    elapsed_times = [r['elapsed_sec'] for r in results]
    print(f"Total time: {sum(elapsed_times):.2f}s")
    print(f"Avg time/prompt: {sum(elapsed_times)/len(elapsed_times):.3f}s")
    print(f"Min time: {min(elapsed_times):.3f}s")
    print(f"Max time: {max(elapsed_times):.3f}s")
    print()

    # Analyze completions
    analyses = [analyze_completion(r['completion']) for r in results]
    qualities = [categorize_quality(a) for a in analyses]
    quality_counts = Counter(qualities)

    print("-" * 70)
    print(" Quality Distribution")
    print("-" * 70)
    for quality, count in quality_counts.most_common():
        pct = count / len(results) * 100
        print(f"{quality:20s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Completion statistics
    print("-" * 70)
    print(" Completion Statistics")
    print("-" * 70)
    avg_length = sum(a['length'] for a in analyses) / len(analyses)
    avg_unique = sum(a['unique_chars'] for a in analyses) / len(analyses)
    avg_repetition = sum(a['repetition_ratio'] for a in analyses) / len(analyses)

    print(f"Avg completion length: {avg_length:.1f} chars")
    print(f"Avg unique chars: {avg_unique:.1f}")
    print(f"Avg repetition ratio: {avg_repetition:.2%}")
    print()

    has_keywords_count = sum(1 for a in analyses if a['has_keywords'])
    has_operators_count = sum(1 for a in analyses if a['has_operators'])
    has_parens_count = sum(1 for a in analyses if a['has_parens'])

    print(f"Completions with keywords: {has_keywords_count}/{len(results)} ({has_keywords_count/len(results)*100:.1f}%)")
    print(f"Completions with operators: {has_operators_count}/{len(results)} ({has_operators_count/len(results)*100:.1f}%)")
    print(f"Completions with parens: {has_parens_count}/{len(results)} ({has_parens_count/len(results)*100:.1f}%)")
    print()

    # Most common repeating patterns
    print("-" * 70)
    print(" Most Common Repeating Characters")
    print("-" * 70)
    char_counter = Counter()
    for a in analyses:
        if a['most_common_char']:
            char_counter[a['most_common_char']] += 1

    for char, count in char_counter.most_common(10):
        print(f"{char:10s}: {count:3d} completions")
    print()


def print_examples(results: List[Dict[str, Any]], n: int = 10) -> None:
    """Print example completions."""
    print("=" * 70)
    print(f" Example Completions (showing {min(n, len(results))})")
    print("=" * 70)
    print()

    for i, r in enumerate(results[:n], 1):
        analysis = analyze_completion(r['completion'])
        quality = categorize_quality(analysis)

        print(f"[{i}] {quality.upper()}")
        print(f"Prompt: {r['prompt']}")
        print(f"Completion: {repr(r['completion'][:80])}")
        if len(r['completion']) > 80:
            print(f"            ... ({len(r['completion'])} chars total)")
        print(f"Analysis: {analysis['unique_chars']} unique chars, {analysis['repetition_ratio']:.1%} repetition")
        print(f"Time: {r['elapsed_sec']:.3f}s")
        print()


def print_quality_examples(results: List[Dict[str, Any]]) -> None:
    """Print examples from each quality category."""
    print("=" * 70)
    print(" Examples by Quality Category")
    print("=" * 70)
    print()

    # Categorize all results
    categorized = {}
    for r in results:
        analysis = analyze_completion(r['completion'])
        quality = categorize_quality(analysis)
        if quality not in categorized:
            categorized[quality] = []
        categorized[quality].append((r, analysis))

    # Show one example from each category
    for quality in ['good', 'partial', 'noise', 'repetitive', 'mode_collapse']:
        if quality in categorized:
            print(f"[{quality.upper()}]")
            r, analysis = categorized[quality][0]
            print(f"Prompt: {r['prompt']}")
            print(f"Completion: {repr(r['completion'][:100])}")
            if len(r['completion']) > 100:
                print(f"            ... ({len(r['completion'])} chars total)")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze batch evaluation results")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSONL file (e.g., eval/results.jsonl)"
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="Show N example completions (default: 0)"
    )
    parser.add_argument(
        "--show_quality_examples",
        action="store_true",
        help="Show one example from each quality category"
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    print()

    results = load_results(results_path)
    if not results:
        print("Error: No results found in file")
        sys.exit(1)

    # Print summary
    print_summary(results)

    # Print quality examples
    if args.show_quality_examples:
        print_quality_examples(results)

    # Print examples
    if args.show_examples > 0:
        print_examples(results, args.show_examples)


if __name__ == "__main__":
    main()
