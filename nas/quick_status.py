#!/usr/bin/env python3
"""
Quick NAS Experiment Status Tool

Display status summary of all experiments in logs/ directory.

Usage:
    python quick_status.py
    python quick_status.py --detailed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_best_architecture(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load best_architecture.json from experiment directory."""
    best_arch_path = exp_dir / "evolution" / "best_architecture.json"
    if not best_arch_path.exists():
        return None
    try:
        with open(best_arch_path) as f:
            return json.load(f)
    except:
        return None


def load_latest_checkpoint(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load latest checkpoint from experiment directory."""
    evo_dir = exp_dir / "evolution"
    if not evo_dir.exists():
        return None

    # Find all checkpoint files
    checkpoints = list(evo_dir.glob("checkpoint_gen*.json"))
    if not checkpoints:
        return None

    # Sort by generation number (extract from filename)
    def extract_gen(path: Path) -> int:
        try:
            return int(path.stem.split("_gen")[-1])
        except:
            return -1

    checkpoints.sort(key=extract_gen, reverse=True)

    try:
        with open(checkpoints[0]) as f:
            return json.load(f)
    except:
        return None


def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate unit (K, M)."""
    if num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def get_experiment_status(exp_dir: Path) -> Dict[str, Any]:
    """Get status summary for a single experiment."""
    status = {
        "name": exp_dir.name,
        "status": "UNKNOWN",
        "fitness": None,
        "val_loss": None,
        "accuracy": None,
        "params": None,
        "size_mb": None,
        "latency_ms": None,
        "architecture": None,
        "generation": None,
    }

    # Try to load best_architecture.json
    best_data = load_best_architecture(exp_dir)
    if best_data:
        status["status"] = "DONE"

        # Extract metrics (handle different key structures)
        raw_metrics = best_data.get("raw_metrics", best_data)
        arch = best_data.get("architecture", {})

        status["fitness"] = best_data.get("fitness")
        status["val_loss"] = raw_metrics.get("val_loss")
        status["accuracy"] = raw_metrics.get("accuracy")
        status["params"] = raw_metrics.get("num_params")
        status["size_mb"] = raw_metrics.get("model_size_mb")
        status["latency_ms"] = raw_metrics.get("latency_ms")

        # Format architecture string
        if arch:
            status["architecture"] = (
                f"{arch.get('arch_type', '?')} "
                f"L{arch.get('num_layers', '?')} "
                f"H{arch.get('hidden_dim', '?')}"
            )
    else:
        # Try to load latest checkpoint
        checkpoint = load_latest_checkpoint(exp_dir)
        if checkpoint:
            status["status"] = "RUNNING"

            # Get generation number
            status["generation"] = checkpoint.get("generation")

            # Get best architecture from checkpoint
            best_arch = checkpoint.get("best_architecture", {})
            raw_metrics = best_arch.get("raw_metrics", best_arch)
            arch = best_arch.get("architecture", {})

            status["fitness"] = best_arch.get("fitness")
            status["val_loss"] = raw_metrics.get("val_loss")
            status["accuracy"] = raw_metrics.get("accuracy")
            status["params"] = raw_metrics.get("num_params")
            status["size_mb"] = raw_metrics.get("model_size_mb")
            status["latency_ms"] = raw_metrics.get("latency_ms")

            if arch:
                status["architecture"] = (
                    f"{arch.get('arch_type', '?')} "
                    f"L{arch.get('num_layers', '?')} "
                    f"H{arch.get('hidden_dim', '?')}"
                )
        else:
            status["status"] = "NOT_FOUND"

    return status


def print_status_table(statuses: List[Dict[str, Any]], detailed: bool = False):
    """Print experiment status as a formatted table."""
    print("=" * 70)
    print(" NAS Experiment Status")
    print("=" * 70)
    print()

    for status in statuses:
        name = status["name"]
        state = status["status"]

        # Status indicator
        if state == "DONE":
            indicator = "[DONE]   "
        elif state == "RUNNING":
            indicator = "[RUN]    "
        else:
            indicator = "[ERROR]  "

        print(f"{indicator} {name}")
        print(f"  {'status':<15}: {state}", end="")
        if state == "RUNNING" and status["generation"] is not None:
            print(f" (Gen {status['generation']})")
        else:
            print()

        # Metrics
        if status["fitness"] is not None:
            print(f"  {'fitness':<15}: {status['fitness']:.4f}")

        if status["val_loss"] is not None:
            print(f"  {'val_loss':<15}: {status['val_loss']:.4f}")

        if status["accuracy"] is not None:
            acc_pct = status["accuracy"] * 100 if status["accuracy"] <= 1.0 else status["accuracy"]
            print(f"  {'accuracy':<15}: {acc_pct:.2f}%")

        if status["params"] is not None:
            print(f"  {'params':<15}: {format_number(status['params'])}")

        if status["size_mb"] is not None:
            print(f"  {'size_mb':<15}: {status['size_mb']:.2f}")

        if status["latency_ms"] is not None:
            print(f"  {'latency_ms':<15}: {status['latency_ms']:.2f}")

        if status["architecture"]:
            print(f"  {'architecture':<15}: {status['architecture']}")

        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Display NAS experiment status")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed information")
    parser.add_argument("--logs", default="logs", help="Path to logs directory (default: logs)")

    args = parser.parse_args()

    logs_dir = Path(args.logs)
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        sys.exit(1)

    # Find all experiment directories (those with evolution/ subdirectory)
    experiment_dirs = [
        d for d in logs_dir.iterdir()
        if d.is_dir() and (d / "evolution").exists()
    ]

    if not experiment_dirs:
        print(f"No experiments found in {logs_dir}")
        sys.exit(0)

    # Sort by name
    experiment_dirs.sort(key=lambda d: d.name)

    # Get status for each experiment
    statuses = [get_experiment_status(d) for d in experiment_dirs]

    # Print status table
    print_status_table(statuses, detailed=args.detailed)


if __name__ == "__main__":
    main()
