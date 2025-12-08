"""
Multi-Objective Fitness Function for NAS

Research-grade implementation with:
- Normalized scoring for each objective
- Configurable weights
- Pareto-optimal tracking
- Full metrics logging

Objectives:
1. val_loss (lower is better) -> s_loss
2. model_size_mb (lower is better) -> s_size
3. latency_ms (lower is better) -> s_latency
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class FitnessConfig:
    """
    Configuration for fitness function

    Weights should sum to 1.0 for interpretability
    """
    # Weights for each objective
    w_loss: float = 0.5      # Weight for loss score
    w_size: float = 0.3      # Weight for size score
    w_latency: float = 0.2   # Weight for latency score

    # Reference values for normalization
    loss_ref: float = 2.0    # Reference val_loss (random baseline ~5.6)
    loss_min: float = 0.1    # Target val_loss (well-trained model)

    size_max: float = 200.0  # Maximum acceptable size (MB)
    size_target: float = 50.0  # Target size (MB)

    latency_max: float = 50.0  # Maximum acceptable latency (ms)
    latency_target: float = 5.0  # Target latency (ms)

    # Scaling factors
    loss_alpha: float = 1.0   # Exponential decay rate for loss

    def to_dict(self) -> Dict:
        return {
            "w_loss": self.w_loss,
            "w_size": self.w_size,
            "w_latency": self.w_latency,
            "loss_ref": self.loss_ref,
            "loss_min": self.loss_min,
            "size_max": self.size_max,
            "size_target": self.size_target,
            "latency_max": self.latency_max,
            "latency_target": self.latency_target,
            "loss_alpha": self.loss_alpha
        }


@dataclass
class FitnessResult:
    """
    Complete fitness evaluation result

    Contains both raw metrics and normalized scores
    """
    # Raw metrics
    val_loss: float
    val_ppl: float
    num_params: int
    model_size_mb: float
    latency_ms: float
    train_time_s: float

    # Normalized scores (0-1, higher is better)
    s_loss: float = 0.0
    s_size: float = 0.0
    s_latency: float = 0.0

    # Final fitness
    fitness: float = 0.0

    # Config used
    config: Optional[FitnessConfig] = None

    def to_dict(self) -> Dict:
        return {
            "raw": {
                "val_loss": self.val_loss,
                "val_ppl": self.val_ppl,
                "num_params": self.num_params,
                "model_size_mb": self.model_size_mb,
                "latency_ms": self.latency_ms,
                "train_time_s": self.train_time_s
            },
            "normalized": {
                "s_loss": self.s_loss,
                "s_size": self.s_size,
                "s_latency": self.s_latency
            },
            "fitness": self.fitness,
            "config": self.config.to_dict() if self.config else None
        }


class FitnessFunction:
    """
    Multi-objective fitness function

    Formulas:
    ---------
    1. Loss score (exponential decay):
       s_loss = exp(-alpha * max(0, val_loss - loss_min))

       - val_loss = loss_min -> s_loss = 1.0
       - val_loss = loss_ref -> s_loss = exp(-alpha * (loss_ref - loss_min))

    2. Size score (logarithmic):
       s_size = log(size_max / size) / log(size_max / size_target)
       Clamped to [0, 1]

       - size = size_target -> s_size = 1.0
       - size = size_max -> s_size = 0.0

    3. Latency score (logarithmic):
       s_latency = log(latency_max / latency) / log(latency_max / latency_target)
       Clamped to [0, 1]

       - latency = latency_target -> s_latency = 1.0
       - latency = latency_max -> s_latency = 0.0

    4. Final fitness:
       fitness = w_loss * s_loss + w_size * s_size + w_latency * s_latency
    """

    def __init__(self, config: FitnessConfig = None):
        self.config = config or FitnessConfig()

    def compute_loss_score(self, val_loss: float) -> float:
        """
        Compute normalized loss score

        s_loss = exp(-alpha * max(0, val_loss - loss_min))
        """
        cfg = self.config
        excess_loss = max(0, val_loss - cfg.loss_min)
        s_loss = math.exp(-cfg.loss_alpha * excess_loss)
        return min(1.0, max(0.0, s_loss))

    def compute_size_score(self, size_mb: float) -> float:
        """
        Compute normalized size score

        s_size = log(size_max / size) / log(size_max / size_target)
        """
        cfg = self.config

        if size_mb <= 0:
            return 0.0
        if size_mb >= cfg.size_max:
            return 0.0
        if size_mb <= cfg.size_target:
            return 1.0

        # Logarithmic scaling
        numerator = math.log(cfg.size_max / size_mb)
        denominator = math.log(cfg.size_max / cfg.size_target)

        if denominator == 0:
            return 1.0

        s_size = numerator / denominator
        return min(1.0, max(0.0, s_size))

    def compute_latency_score(self, latency_ms: float) -> float:
        """
        Compute normalized latency score

        s_latency = log(latency_max / latency) / log(latency_max / latency_target)
        """
        cfg = self.config

        if latency_ms <= 0:
            return 1.0
        if latency_ms >= cfg.latency_max:
            return 0.0
        if latency_ms <= cfg.latency_target:
            return 1.0

        # Logarithmic scaling
        numerator = math.log(cfg.latency_max / latency_ms)
        denominator = math.log(cfg.latency_max / cfg.latency_target)

        if denominator == 0:
            return 1.0

        s_latency = numerator / denominator
        return min(1.0, max(0.0, s_latency))

    def compute(self, metrics: Dict) -> FitnessResult:
        """
        Compute full fitness from raw metrics

        Args:
            metrics: Dict with keys:
                - val_loss
                - val_ppl (optional, will compute from val_loss)
                - num_params
                - model_size_mb
                - latency_ms
                - train_time_s

        Returns:
            FitnessResult with all scores
        """
        cfg = self.config

        # Extract metrics
        val_loss = metrics.get("val_loss", 5.0)
        val_ppl = metrics.get("val_ppl", math.exp(min(val_loss, 10)))
        num_params = int(metrics.get("num_params", 0))
        model_size_mb = metrics.get("model_size_mb", 100.0)
        latency_ms = metrics.get("latency_ms", 10.0)
        train_time_s = metrics.get("train_time_s", 0.0)

        # Compute normalized scores
        s_loss = self.compute_loss_score(val_loss)
        s_size = self.compute_size_score(model_size_mb)
        s_latency = self.compute_latency_score(latency_ms)

        # Weighted sum
        fitness = (
            cfg.w_loss * s_loss +
            cfg.w_size * s_size +
            cfg.w_latency * s_latency
        )

        return FitnessResult(
            val_loss=val_loss,
            val_ppl=val_ppl,
            num_params=num_params,
            model_size_mb=model_size_mb,
            latency_ms=latency_ms,
            train_time_s=train_time_s,
            s_loss=s_loss,
            s_size=s_size,
            s_latency=s_latency,
            fitness=fitness,
            config=cfg
        )


class ParetoTracker:
    """
    Track Pareto-optimal solutions

    A solution is Pareto-optimal if no other solution
    dominates it on all objectives.
    """

    def __init__(self):
        self.solutions: List[FitnessResult] = []

    def dominates(self, a: FitnessResult, b: FitnessResult) -> bool:
        """
        Check if solution A dominates solution B

        A dominates B if:
        - A is at least as good as B on all objectives
        - A is strictly better than B on at least one objective
        """
        # Lower val_loss is better
        # Lower size is better
        # Lower latency is better

        a_better_or_equal = (
            a.val_loss <= b.val_loss and
            a.model_size_mb <= b.model_size_mb and
            a.latency_ms <= b.latency_ms
        )

        a_strictly_better = (
            a.val_loss < b.val_loss or
            a.model_size_mb < b.model_size_mb or
            a.latency_ms < b.latency_ms
        )

        return a_better_or_equal and a_strictly_better

    def add(self, result: FitnessResult) -> bool:
        """
        Add solution and update Pareto front

        Returns:
            True if solution is Pareto-optimal
        """
        # Check if dominated by existing solutions
        for existing in self.solutions:
            if self.dominates(existing, result):
                return False  # Not Pareto-optimal

        # Remove solutions dominated by new one
        self.solutions = [
            s for s in self.solutions
            if not self.dominates(result, s)
        ]

        self.solutions.append(result)
        return True

    def get_front(self) -> List[FitnessResult]:
        """Get current Pareto front"""
        return self.solutions.copy()

    def save(self, path: Path):
        """Save Pareto front to JSON"""
        data = [s.to_dict() for s in self.solutions]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def __len__(self):
        return len(self.solutions)


def plot_pareto_front(results: List[FitnessResult], save_path: str = None):
    """
    Plot 2D Pareto fronts

    Creates plots for:
    - val_loss vs model_size
    - val_loss vs latency
    - model_size vs latency
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Extract data
    losses = [r.val_loss for r in results]
    sizes = [r.model_size_mb for r in results]
    latencies = [r.latency_ms for r in results]
    fitness = [r.fitness for r in results]

    # Normalize fitness for coloring
    fitness_norm = np.array(fitness)
    fitness_norm = (fitness_norm - fitness_norm.min()) / (fitness_norm.max() - fitness_norm.min() + 1e-8)

    # Plot 1: val_loss vs size
    ax = axes[0]
    scatter = ax.scatter(sizes, losses, c=fitness, cmap='RdYlGn', s=50)
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Val Loss')
    ax.set_title('Loss vs Size')
    ax.grid(True, alpha=0.3)

    # Plot 2: val_loss vs latency
    ax = axes[1]
    ax.scatter(latencies, losses, c=fitness, cmap='RdYlGn', s=50)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Val Loss')
    ax.set_title('Loss vs Latency')
    ax.grid(True, alpha=0.3)

    # Plot 3: size vs latency
    ax = axes[2]
    ax.scatter(latencies, sizes, c=fitness, cmap='RdYlGn', s=50)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Size vs Latency')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=axes, label='Fitness')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved Pareto front: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Fitness Function Test")
    print("="*60)

    # Create fitness function
    config = FitnessConfig(
        w_loss=0.5,
        w_size=0.3,
        w_latency=0.2
    )
    fitness_fn = FitnessFunction(config)

    # Test cases
    test_cases = [
        {"val_loss": 0.1, "model_size_mb": 50, "latency_ms": 5, "num_params": 10e6},
        {"val_loss": 0.5, "model_size_mb": 100, "latency_ms": 10, "num_params": 50e6},
        {"val_loss": 1.0, "model_size_mb": 150, "latency_ms": 20, "num_params": 100e6},
        {"val_loss": 2.0, "model_size_mb": 200, "latency_ms": 50, "num_params": 200e6},
        {"val_loss": 0.2, "model_size_mb": 27.6, "latency_ms": 4.64, "num_params": 18.4e6},  # NAS best
    ]

    print("\nTest Cases:")
    print("-"*70)

    pareto = ParetoTracker()

    for i, metrics in enumerate(test_cases, 1):
        result = fitness_fn.compute(metrics)
        is_pareto = pareto.add(result)

        print(f"\nCase {i}:")
        print(f"  Raw: loss={result.val_loss:.2f}, size={result.model_size_mb:.1f}MB, lat={result.latency_ms:.1f}ms")
        print(f"  Scores: s_loss={result.s_loss:.3f}, s_size={result.s_size:.3f}, s_lat={result.s_latency:.3f}")
        print(f"  Fitness: {result.fitness:.3f}")
        print(f"  Pareto-optimal: {is_pareto}")

    print(f"\n{'='*60}")
    print(f"Pareto Front: {len(pareto)} solutions")
    print("="*60)

    for r in pareto.get_front():
        print(f"  loss={r.val_loss:.2f}, size={r.model_size_mb:.1f}MB, lat={r.latency_ms:.1f}ms -> fitness={r.fitness:.3f}")
