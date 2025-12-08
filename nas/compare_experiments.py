#!/usr/bin/env python3
"""
NAS Experiment Comparison Tool

Compare results from two NAS experiments (e.g., v1 single-stage vs v2 two-stage).

Usage:
    python compare_experiments.py logs/code_nas_v1_single logs/code_nas_v2_two_stage
    python compare_experiments.py --exp1 logs/code_nas_v1_single --exp2 logs/code_nas_v2_two_stage
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def load_best_architecture(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load best_architecture.json from experiment directory."""
    best_arch_path = exp_dir / "evolution" / "best_architecture.json"
    if not best_arch_path.exists():
        return None
    with open(best_arch_path) as f:
        return json.load(f)


def load_fitness_history(exp_dir: Path) -> Optional[list]:
    """Load fitness_history.json from experiment directory."""
    history_path = exp_dir / "evolution" / "fitness_history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        return json.load(f)


def format_arch(arch: Dict[str, Any]) -> str:
    """Format architecture config as compact string."""
    a = arch.get("architecture", {})
    return (
        f"{a.get('arch_type', 'unknown')} "
        f"L{a.get('num_layers', '?')} "
        f"H{a.get('hidden_dim', '?')} "
        f"Heads={a.get('num_heads', '?')} "
        f"FFN×{a.get('ffn_multiplier', '?')} "
        f"{a.get('activation', '?')} "
        f"{a.get('position_encoding', '?')}"
    )


def print_comparison(
    exp1_name: str,
    exp1_data: Dict[str, Any],
    exp2_name: str,
    exp2_data: Dict[str, Any],
    hist1: Optional[list],
    hist2: Optional[list]
):
    """Print side-by-side comparison of two experiments."""

    print("=" * 70)
    print("NAS Experiment Comparison")
    print("=" * 70)
    print()

    # Header
    col1 = exp1_name[:25]
    col2 = exp2_name[:25]
    print(f"{'Metric':<20} {col1:>22} {col2:>22}")
    print("-" * 70)

    # Raw metrics
    m1 = exp1_data.get("raw_metrics", {})
    m2 = exp2_data.get("raw_metrics", {})

    metrics = [
        ("Fitness", exp1_data.get("fitness"), exp2_data.get("fitness"), "{:.4f}"),
        ("Val Loss", m1.get("val_loss"), m2.get("val_loss"), "{:.4f}"),
        ("Val PPL", m1.get("val_ppl"), m2.get("val_ppl"), "{:.2f}"),
        ("Accuracy", m1.get("accuracy"), m2.get("accuracy"), "{:.2%}"),
        ("Params", m1.get("num_params"), m2.get("num_params"), "{:,.0f}"),
        ("Model Size (MB)", m1.get("model_size_mb"), m2.get("model_size_mb"), "{:.2f}"),
        ("Latency (ms)", m1.get("latency_ms"), m2.get("latency_ms"), "{:.2f}"),
        ("Train Time (s)", m1.get("train_time_s"), m2.get("train_time_s"), "{:.2f}"),
    ]

    for name, v1, v2, fmt in metrics:
        s1 = fmt.format(v1) if v1 is not None else "N/A"
        s2 = fmt.format(v2) if v2 is not None else "N/A"

        # Highlight winner
        winner = ""
        if v1 is not None and v2 is not None:
            if name in ["Val Loss", "Model Size (MB)", "Latency (ms)", "Train Time (s)"]:
                # Lower is better
                if v1 < v2:
                    winner = " <--"
                elif v2 < v1:
                    winner = "     -->"
            else:
                # Higher is better
                if v1 > v2:
                    winner = " <--"
                elif v2 > v1:
                    winner = "     -->"

        print(f"{name:<20} {s1:>22} {s2:>22}{winner}")

    print("-" * 70)

    # Architecture comparison
    print()
    print("Architecture:")
    print(f"  {exp1_name}: {format_arch(exp1_data)}")
    print(f"  {exp2_name}: {format_arch(exp2_data)}")

    # Evolution stats
    if hist1 and hist2:
        print()
        print("Evolution Progress:")
        print(f"  {exp1_name}: {len(hist1)} generations")
        print(f"  {exp2_name}: {len(hist2)} generations")

        # Find generation where fitness=1.0 was first achieved
        def find_first_perfect(hist):
            for entry in hist:
                if entry.get("best_fitness", 0) >= 1.0:
                    return entry.get("generation", "?")
            return None

        gen1 = find_first_perfect(hist1)
        gen2 = find_first_perfect(hist2)

        if gen1 is not None or gen2 is not None:
            print()
            print("First Generation with Fitness=1.0:")
            print(f"  {exp1_name}: Gen {gen1}" if gen1 is not None else f"  {exp1_name}: Not achieved")
            print(f"  {exp2_name}: Gen {gen2}" if gen2 is not None else f"  {exp2_name}: Not achieved")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare two NAS experiments")
    parser.add_argument("exp1", nargs="?", help="Path to first experiment directory")
    parser.add_argument("exp2", nargs="?", help="Path to second experiment directory")
    parser.add_argument("--exp1", dest="exp1_alt", help="Alternative: path to first experiment")
    parser.add_argument("--exp2", dest="exp2_alt", help="Alternative: path to second experiment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Resolve paths
    exp1_path = Path(args.exp1 or args.exp1_alt or "logs/code_nas_v1_single")
    exp2_path = Path(args.exp2 or args.exp2_alt or "logs/code_nas_v2_two_stage")

    # Load data
    exp1_data = load_best_architecture(exp1_path)
    exp2_data = load_best_architecture(exp2_path)

    if exp1_data is None:
        print(f"Error: Could not load {exp1_path}/evolution/best_architecture.json")
        sys.exit(1)
    if exp2_data is None:
        print(f"Error: Could not load {exp2_path}/evolution/best_architecture.json")
        sys.exit(1)

    hist1 = load_fitness_history(exp1_path)
    hist2 = load_fitness_history(exp2_path)

    if args.json:
        result = {
            "exp1": {"name": exp1_path.name, "data": exp1_data, "history": hist1},
            "exp2": {"name": exp2_path.name, "data": exp2_data, "history": hist2},
        }
        print(json.dumps(result, indent=2))
    else:
        print_comparison(
            exp1_path.name, exp1_data,
            exp2_path.name, exp2_data,
            hist1, hist2
        )


if __name__ == "__main__":
    main()
