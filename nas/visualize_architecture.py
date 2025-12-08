#!/usr/bin/env python3
"""
visualize_architecture.py

Show a human-readable summary of a NAS architecture JSON.

Usage:
    python visualize_architecture.py path/to/best_architecture.json
    python visualize_architecture.py models/codenas_best_current.json
    python visualize_architecture.py logs/code_nas_v1_single/evolution/best_architecture.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def load_arch(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load architecture and metrics from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))

    # Flexible key extraction to handle different JSON structures
    arch = (
        data.get("arch_config")
        or data.get("architecture")
        or data.get("config", {}).get("arch_config")
        or data
    )
    metrics = data.get("raw_metrics", {}) or data.get("metrics", {})

    return arch, metrics, data


def fmt_number(x: Optional[float]) -> str:
    """Format number with appropriate unit (K, M)."""
    if x is None:
        return "N/A"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    elif x >= 1e3:
        return f"{x/1e3:.2f}K"
    else:
        return f"{x:.0f}"


def fmt_mb(x: Optional[float]) -> str:
    """Format size in MB."""
    if x is None:
        return "N/A"
    return f"{x:.2f} MB"


def visualize(path: Path) -> None:
    """Visualize architecture from JSON file."""
    arch, metrics, full_data = load_arch(path)

    # Extract architecture parameters
    arch_type = arch.get("arch_type", "unknown")
    num_layers = arch.get("num_layers", "?")
    hidden_dim = arch.get("hidden_dim", "?")
    num_heads = arch.get("num_heads", "?")
    ffn_mult = arch.get("ffn_multiplier", "?")
    norm = arch.get("normalization", "?")
    act = arch.get("activation", "?")
    pos = arch.get("position_encoding") or arch.get("pos_encoding", "?")
    vocab_size = arch.get("vocab_size", "?")
    max_seq_len = arch.get("max_seq_length", "?")
    quant = arch.get("quantization", "?")
    prune = arch.get("pruning_ratio", 0.0)

    # Extract metrics
    num_params = metrics.get("num_params")
    size_mb = metrics.get("model_size_mb")
    val_loss = metrics.get("val_loss")
    val_ppl = metrics.get("val_ppl")
    acc = metrics.get("accuracy")
    latency_ms = metrics.get("latency_ms") or metrics.get("latency")
    train_time_s = metrics.get("train_time_s")
    fitness = full_data.get("fitness")

    # Print header
    print("=" * 70)
    print(" Architecture Summary")
    print("=" * 70)
    print(f"JSON file    : {path.name}")
    print(f"Source       : {path.parent.name if path.parent.name != 'models' else 'models/'}")

    # Version info if available
    if "version" in full_data:
        print(f"Version      : {full_data['version']}")
    if "description" in full_data:
        print(f"Description  : {full_data['description']}")

    print()
    print("-" * 70)
    print(" Configuration")
    print("-" * 70)
    print(f"Type         : {arch_type}")
    print(f"Layers       : {num_layers}")
    print(f"Hidden dim   : {hidden_dim}")
    print(f"Heads        : {num_heads}")
    print(f"FFN mult     : {ffn_mult}")
    print(f"Normalization: {norm}")
    print(f"Activation   : {act}")
    print(f"Pos encoding : {pos}")
    print(f"Vocab size   : {vocab_size}")
    print(f"Max seq len  : {max_seq_len}")
    print(f"Quantization : {quant}")
    if prune > 0:
        print(f"Pruning      : {prune:.0%}")

    print()
    print("-" * 70)
    print(" Performance Metrics")
    print("-" * 70)

    if fitness is not None:
        print(f"Fitness      : {fitness:.4f}")
    if num_params is not None:
        print(f"Parameters   : {fmt_number(num_params)}")
    if size_mb is not None:
        print(f"Model size   : {fmt_mb(size_mb)}")
    if val_loss is not None:
        print(f"Val loss     : {val_loss:.4f}")
    if val_ppl is not None:
        print(f"Val PPL      : {val_ppl:.3f}")
    if acc is not None:
        acc_pct = acc * 100 if acc <= 1.0 else acc
        print(f"Accuracy     : {acc_pct:.2f}%")
    if latency_ms is not None:
        print(f"Latency      : {latency_ms:.2f} ms")
    if train_time_s is not None:
        print(f"Train time   : {train_time_s:.2f} s")

    print()
    print("-" * 70)
    print(" Model Architecture")
    print("-" * 70)
    print()

    # Calculate FFN hidden size
    ffn_hidden = f"{hidden_dim} × {ffn_mult}" if isinstance(hidden_dim, int) and isinstance(ffn_mult, (int, float)) else "?"

    # Block diagram
    print("Input (code tokens)")
    print("  |")
    print(f"  +--> Embedding Layer (vocab={vocab_size}, dim={hidden_dim})")
    print("  |")

    if arch_type == "transformer":
        print(f"  +--> [×{num_layers}] Transformer Block")
        print("  |      |")
        print(f"  |      +--> Multi-Head Attention")
        print(f"  |      |      heads={num_heads}, dim={hidden_dim}, pos={pos}")
        print("  |      |")
        print(f"  |      +--> Feed-Forward Network")
        print(f"  |      |      input={hidden_dim} -> hidden={ffn_hidden} -> output={hidden_dim}")
        print(f"  |      |      activation={act}")
        print("  |      |")
        print(f"  |      +--> Normalization: {norm}")
        print("  |")
    elif arch_type == "linear_transformer":
        print(f"  +--> [×{num_layers}] Linear Transformer Block")
        print("  |      |")
        print(f"  |      +--> Linear Attention (O(n) complexity)")
        print(f"  |      |      heads={num_heads}, dim={hidden_dim}")
        print("  |      |")
        print(f"  |      +--> Feed-Forward Network")
        print(f"  |      |      input={hidden_dim} -> hidden={ffn_hidden} -> output={hidden_dim}")
        print(f"  |      |      activation={act}")
        print("  |      |")
        print(f"  |      +--> Normalization: {norm}")
        print("  |")
    else:
        print(f"  +--> [×{num_layers}] {arch_type.upper()} Block")
        print("  |      [Architecture-specific details]")
        print("  |")

    print(f"  +--> Language Model Head (vocab projection)")
    print(f"  |      dim={hidden_dim} -> vocab={vocab_size}")
    print("  |")
    print("Output (token probabilities)")

    print()
    print("=" * 70)

    # Print recommendation if available
    if "why_this_model" in full_data:
        print()
        print("Recommendation:")
        print(f"  {full_data['why_this_model']}")
        print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_architecture.py path/to/best_architecture.json")
        print()
        print("Examples:")
        print("  python visualize_architecture.py models/codenas_best_current.json")
        print("  python visualize_architecture.py logs/code_nas_v1_single/evolution/best_architecture.json")
        print("  python visualize_architecture.py logs/code_nas_v2_two_stage/evolution/best_architecture.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    try:
        visualize(path)
    except Exception as e:
        print(f"Error: Failed to visualize architecture: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
