#!/usr/bin/env python3
"""
Sanity Check Comparison Tool

Compares Sequential vs Parallel NAS runs to verify consistency.

Usage:
    python compare_sanity.py
    python compare_sanity.py --seq_exp sanity_seq --par_exp sanity_par
"""

from pathlib import Path
import json
import argparse
from typing import Dict, List, Optional, Tuple


def load_best(exp_name: str) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
    """Load best architecture and fitness history for an experiment."""
    base = Path("logs") / exp_name / "evolution"
    best_path = base / "best_architecture.json"
    hist_path = base / "fitness_history.json"

    best = None
    history = None

    if best_path.exists():
        best = json.loads(best_path.read_text())

    if hist_path.exists():
        history = json.loads(hist_path.read_text())

    return best, history


def summarize_history(name: str, history: Optional[List[Dict]]) -> None:
    """Print fitness history summary."""
    print(f"[{name}]")
    if not history:
        print("  No history found")
        return

    for h in history:
        print(
            f"  Gen {h['generation']}: "
            f"best={h['best_fitness']:.4f}, "
            f"mean={h['mean_fitness']:.4f} +/- {h.get('std_fitness', 0):.4f}"
        )
    print()


def compare_architectures(seq_arch: Dict, par_arch: Dict) -> Dict:
    """Compare two architectures and return differences."""
    diffs = {}
    all_keys = set(seq_arch.keys()) | set(par_arch.keys())

    for key in all_keys:
        seq_val = seq_arch.get(key)
        par_val = par_arch.get(key)
        if seq_val != par_val:
            diffs[key] = {"seq": seq_val, "par": par_val}

    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sequential vs Parallel NAS sanity check results"
    )
    parser.add_argument(
        "--seq_exp",
        default="sanity_seq",
        help="Sequential experiment name (default: sanity_seq)"
    )
    parser.add_argument(
        "--par_exp",
        default="sanity_par",
        help="Parallel experiment name (default: sanity_par)"
    )
    args = parser.parse_args()

    # Load results
    seq_best, seq_hist = load_best(args.seq_exp)
    par_best, par_hist = load_best(args.par_exp)

    print("\n" + "=" * 60)
    print("SANITY CHECK: Sequential vs Parallel Comparison")
    print("=" * 60 + "\n")

    # Check if experiments exist
    if seq_best is None:
        print(f"[ERROR] Sequential experiment '{args.seq_exp}' not found")
        print(f"        Expected: logs/{args.seq_exp}/evolution/best_architecture.json")
        return

    if par_best is None:
        print(f"[WARNING] Parallel experiment '{args.par_exp}' not found")
        print(f"          Expected: logs/{args.par_exp}/evolution/best_architecture.json")
        print(f"          Run parallel sanity check first.")
        print("\n[Sequential Results Only]")
        summarize_history("Sequential", seq_hist)
        print(f"Best Fitness: {seq_best['fitness']:.4f}")
        return

    # Print fitness history
    print("[Fitness History]")
    print("-" * 40)
    summarize_history("Sequential", seq_hist)
    summarize_history("Parallel", par_hist)

    # Compare best fitness
    print("[Best Fitness Comparison]")
    print("-" * 40)
    seq_fitness = seq_best['fitness']
    par_fitness = par_best['fitness']
    diff = abs(seq_fitness - par_fitness)

    print(f"  Sequential: {seq_fitness:.4f}")
    print(f"  Parallel  : {par_fitness:.4f}")
    print(f"  Difference: {diff:.4f}")

    if diff < 0.01:
        print("  Status: PASS (difference < 0.01)")
    elif diff < 0.05:
        print("  Status: ACCEPTABLE (difference < 0.05)")
    else:
        print("  Status: WARNING (difference >= 0.05)")

    print()

    # Compare best architectures
    print("[Best Architecture Comparison]")
    print("-" * 40)

    seq_arch = seq_best.get('architecture', {})
    par_arch = par_best.get('architecture', {})

    # Key parameters to compare
    key_params = ['arch_type', 'num_layers', 'hidden_dim', 'num_heads', 'ffn_multiplier']

    print("\n  Parameter       | Sequential | Parallel")
    print("  " + "-" * 45)

    for param in key_params:
        seq_val = seq_arch.get(param, 'N/A')
        par_val = par_arch.get(param, 'N/A')
        match = "[OK]" if seq_val == par_val else "[DIFF]"
        print(f"  {param:16} | {str(seq_val):10} | {str(par_val):10} {match}")

    # Full architecture diff
    diffs = compare_architectures(seq_arch, par_arch)
    if diffs:
        print("\n  [Differences Found]")
        for key, vals in diffs.items():
            print(f"    {key}: seq={vals['seq']}, par={vals['par']}")
    else:
        print("\n  [No differences in architecture parameters]")

    # Val loss comparison
    print("\n[Validation Metrics]")
    print("-" * 40)
    seq_metrics = seq_best.get('raw_metrics', {})
    par_metrics = par_best.get('raw_metrics', {})

    print(f"  Val Loss  - Sequential: {seq_metrics.get('val_loss', 'N/A'):.4f}, "
          f"Parallel: {par_metrics.get('val_loss', 'N/A'):.4f}")
    print(f"  Val PPL   - Sequential: {seq_metrics.get('val_ppl', 'N/A'):.4f}, "
          f"Parallel: {par_metrics.get('val_ppl', 'N/A'):.4f}")
    print(f"  Accuracy  - Sequential: {seq_metrics.get('accuracy', 'N/A'):.4f}, "
          f"Parallel: {par_metrics.get('accuracy', 'N/A'):.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if diff < 0.01:
        print("  Sanity check PASSED!")
        print("  Sequential and Parallel modes produce consistent results.")
    elif diff < 0.05:
        print("  Sanity check ACCEPTABLE")
        print("  Minor differences detected, but within tolerance.")
    else:
        print("  Sanity check WARNING")
        print("  Significant differences detected. Please investigate.")

    print()


if __name__ == "__main__":
    main()
