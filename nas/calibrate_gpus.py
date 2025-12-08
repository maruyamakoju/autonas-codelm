#!/usr/bin/env python3
"""
GPU Performance Calibration for Heterogeneous NAS

Runs a mini-benchmark on each GPU to measure actual performance,
then saves calibrated weights to gpu_calibration.json.

Usage:
    python calibrate_gpus.py
    python calibrate_gpus.py --num_runs 5 --output logs/gpu_calibration.json

The calibration result is automatically loaded by detect_gpus_with_profiles()
to provide accurate heterogeneous scheduling.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import torch

from search_space import SearchSpace, ArchitectureConfig
from evaluator import Evaluator
from parallel_evaluator import detect_gpus_with_profiles, GPUProfile


@dataclass
class CalibrationResult:
    """Result of GPU calibration"""
    device: str
    gpu_name: str
    avg_eval_time_s: float
    min_eval_time_s: float
    max_eval_time_s: float
    relative_weight: float  # Normalized to slowest GPU = 1.0
    num_runs: int
    timestamp: float


def benchmark_gpu(
    device: str,
    architectures: List[ArchitectureConfig],
    num_runs: int = 3
) -> Tuple[float, float, float]:
    """
    Benchmark a single GPU with representative architectures.

    Returns:
        (avg_time, min_time, max_time) in seconds
    """
    print(f"\n[BENCHMARK] Testing {device}...")

    # Create evaluator for this GPU (simulated training for speed)
    evaluator = Evaluator(
        device=device,
        use_real_training=False,
        max_train_steps=50
    )

    times = []

    for run in range(num_runs):
        for i, arch in enumerate(architectures):
            t0 = time.time()
            try:
                result = evaluator.evaluate_fast(arch)
                elapsed = time.time() - t0
                times.append(elapsed)
                print(f"  Run {run+1}/{num_runs}, Arch {i+1}: {elapsed:.2f}s")
            except Exception as e:
                print(f"  Run {run+1}/{num_runs}, Arch {i+1}: FAILED ({e})")

    if not times:
        return float('inf'), float('inf'), float('inf')

    return sum(times) / len(times), min(times), max(times)


def calibrate_all_gpus(
    num_runs: int = 3,
    num_archs: int = 3
) -> List[CalibrationResult]:
    """
    Calibrate all available GPUs.

    Args:
        num_runs: Number of benchmark runs per architecture
        num_archs: Number of representative architectures to test

    Returns:
        List of CalibrationResult for each GPU
    """
    # Detect GPUs
    profiles = detect_gpus_with_profiles()

    if not profiles:
        print("No GPUs found!")
        return []

    # Generate representative architectures
    space = SearchSpace(mode="minimal")
    test_archs = []

    # Small architecture
    test_archs.append(ArchitectureConfig(
        arch_type="transformer",
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        ffn_multiplier=4.0,
        vocab_size=101
    ))

    # Medium architecture
    test_archs.append(ArchitectureConfig(
        arch_type="transformer",
        num_layers=6,
        hidden_dim=512,
        num_heads=8,
        ffn_multiplier=4.0,
        vocab_size=101
    ))

    # Large architecture (if requested)
    if num_archs >= 3:
        test_archs.append(ArchitectureConfig(
            arch_type="transformer",
            num_layers=8,
            hidden_dim=512,
            num_heads=8,
            ffn_multiplier=4.0,
            vocab_size=101
        ))

    print("\n" + "=" * 60)
    print("GPU CALIBRATION BENCHMARK")
    print("=" * 60)
    print(f"GPUs: {len(profiles)}")
    print(f"Architectures: {len(test_archs)}")
    print(f"Runs per arch: {num_runs}")

    # Benchmark each GPU
    results = []
    gpu_times = {}

    for profile in profiles:
        avg_time, min_time, max_time = benchmark_gpu(
            profile.device,
            test_archs[:num_archs],
            num_runs
        )
        gpu_times[profile.device] = avg_time

        results.append(CalibrationResult(
            device=profile.device,
            gpu_name=profile.name,
            avg_eval_time_s=avg_time,
            min_eval_time_s=min_time,
            max_eval_time_s=max_time,
            relative_weight=0.0,  # Will be calculated below
            num_runs=num_runs,
            timestamp=time.time()
        ))

    # Calculate relative weights (inverse of time, normalized)
    if results:
        # Find slowest GPU (highest time)
        max_time = max(r.avg_eval_time_s for r in results if r.avg_eval_time_s < float('inf'))

        for r in results:
            if r.avg_eval_time_s < float('inf'):
                # Weight = slowest_time / this_time (faster GPU gets higher weight)
                r.relative_weight = max_time / r.avg_eval_time_s
            else:
                r.relative_weight = 0.0

    return results


def save_calibration(results: List[CalibrationResult], output_path: Path):
    """Save calibration results to JSON"""
    data = {
        'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpus': [asdict(r) for r in results]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] Calibration results: {output_path}")


def load_calibration(path: Path) -> Dict[str, float]:
    """
    Load calibration results and return device -> weight mapping.

    This can be used to override default GPU weights.
    """
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    weights = {}
    for gpu in data.get('gpus', []):
        weights[gpu['device']] = gpu['relative_weight']

    return weights


def print_results(results: List[CalibrationResult]):
    """Print calibration results summary"""
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n  {r.device}: {r.gpu_name}")
        print(f"    Avg time: {r.avg_eval_time_s:.3f}s")
        print(f"    Min/Max: {r.min_eval_time_s:.3f}s / {r.max_eval_time_s:.3f}s")
        print(f"    Relative weight: {r.relative_weight:.3f}x")

    # Expected task distribution for 100 tasks
    if len(results) > 1:
        total_weight = sum(r.relative_weight for r in results)
        print("\n[Expected Task Distribution (100 tasks)]")
        for r in results:
            share = r.relative_weight / total_weight * 100
            tasks = int(100 * r.relative_weight / total_weight)
            print(f"  {r.device}: {tasks} tasks ({share:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate GPU performance for heterogeneous NAS"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of benchmark runs per architecture (default: 3)"
    )
    parser.add_argument(
        "--num_archs",
        type=int,
        default=3,
        help="Number of representative architectures (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/gpu_calibration.json",
        help="Output path for calibration results"
    )
    args = parser.parse_args()

    # Run calibration
    results = calibrate_all_gpus(
        num_runs=args.num_runs,
        num_archs=args.num_archs
    )

    if results:
        print_results(results)
        save_calibration(results, Path(args.output))

        print("\n[USAGE]")
        print("  Calibration will be auto-loaded by detect_gpus_with_profiles()")
        print("  Or load manually with:")
        print(f"    weights = load_calibration(Path('{args.output}'))")
    else:
        print("Calibration failed - no GPUs available or all benchmarks failed")


if __name__ == "__main__":
    main()
