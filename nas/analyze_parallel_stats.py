#!/usr/bin/env python3
"""
Parallel Evaluation Statistics Analyzer

Reads and summarizes worker and batch statistics from parallel NAS runs.

Usage:
    python analyze_parallel_stats.py --log_dir logs/experiment_name/parallel
    python analyze_parallel_stats.py --log_dir logs/medium_nas/parallel --compare logs/dual_gpu_exp/parallel
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if exists"""
    if path.exists():
        return json.loads(path.read_text())
    return None


def analyze_worker_stats(stats: List[Dict]) -> Dict:
    """Analyze worker statistics"""
    if not stats:
        return {}

    total_completed = sum(w['tasks_completed'] for w in stats)
    total_failed = sum(w['tasks_failed'] for w in stats)
    total_time = sum(w['total_eval_time_s'] for w in stats)

    # Per-worker analysis
    worker_summary = []
    for w in stats:
        share = w['tasks_completed'] / max(1, total_completed) * 100
        worker_summary.append({
            'device': w['device'],
            'completed': w['tasks_completed'],
            'failed': w['tasks_failed'],
            'avg_time': w['avg_eval_time_s'],
            'share_pct': share
        })

    return {
        'total_completed': total_completed,
        'total_failed': total_failed,
        'total_worker_time': total_time,
        'workers': worker_summary
    }


def analyze_batch_stats(stats: List[Dict]) -> Dict:
    """Analyze batch statistics"""
    if not stats:
        return {}

    total_tasks = sum(b['total_tasks'] for b in stats)
    total_valid = sum(b['valid'] for b in stats)
    total_errors = sum(b['errors'] for b in stats)
    total_elapsed = sum(b['elapsed_s'] for b in stats)

    # Device distribution aggregation
    device_totals = {}
    for b in stats:
        for device, count in b.get('device_distribution', {}).items():
            device_totals[device] = device_totals.get(device, 0) + count

    return {
        'num_batches': len(stats),
        'total_tasks': total_tasks,
        'total_valid': total_valid,
        'total_errors': total_errors,
        'total_elapsed': total_elapsed,
        'avg_time_per_task': total_elapsed / max(1, total_valid),
        'device_distribution': device_totals
    }


def print_analysis(log_dir: Path, label: str = ""):
    """Print full analysis for a log directory"""
    header = f"=== {label} ===" if label else f"=== {log_dir} ==="
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    # Worker stats
    worker_stats_file = log_dir / "parallel_worker_stats.json"
    worker_stats = load_json(worker_stats_file)

    if worker_stats:
        analysis = analyze_worker_stats(worker_stats)
        print("\n[Worker Statistics]")
        for w in analysis['workers']:
            print(f"  {w['device']:12s}: {w['completed']:3d} tasks ({w['share_pct']:5.1f}%), "
                  f"avg={w['avg_time']:.2f}s, failed={w['failed']}")
        print(f"\n  Total: {analysis['total_completed']} completed, "
              f"{analysis['total_failed']} failed")
        print(f"  Total worker time: {analysis['total_worker_time']:.1f}s")
    else:
        print("\n[Worker Statistics] Not found")

    # Batch stats
    batch_stats_file = log_dir / "parallel_batch_stats.json"
    batch_stats = load_json(batch_stats_file)

    if batch_stats:
        analysis = analyze_batch_stats(batch_stats)
        print("\n[Batch Statistics]")
        print(f"  Batches: {analysis['num_batches']}")
        print(f"  Tasks: {analysis['total_valid']}/{analysis['total_tasks']} valid "
              f"({analysis['total_errors']} errors)")
        print(f"  Total time: {analysis['total_elapsed']:.1f}s "
              f"({analysis['avg_time_per_task']:.2f}s/task)")
        print(f"  Device distribution: {analysis['device_distribution']}")
    else:
        print("\n[Batch Statistics] Not found")

    return worker_stats, batch_stats


def compare_runs(dir1: Path, dir2: Path):
    """Compare two parallel runs"""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Load stats
    ws1 = load_json(dir1 / "parallel_worker_stats.json")
    ws2 = load_json(dir2 / "parallel_worker_stats.json")
    bs1 = load_json(dir1 / "parallel_batch_stats.json")
    bs2 = load_json(dir2 / "parallel_batch_stats.json")

    if ws1 and ws2:
        a1 = analyze_worker_stats(ws1)
        a2 = analyze_worker_stats(ws2)

        print("\n[Worker Time Comparison]")
        print(f"  Run 1: {a1['total_worker_time']:.1f}s")
        print(f"  Run 2: {a2['total_worker_time']:.1f}s")

        if a1['total_worker_time'] > 0:
            speedup = a1['total_worker_time'] / max(1, a2['total_worker_time'])
            print(f"  Speedup: {speedup:.2f}x")

    if bs1 and bs2:
        a1 = analyze_batch_stats(bs1)
        a2 = analyze_batch_stats(bs2)

        print("\n[Throughput Comparison]")
        t1 = a1['total_valid'] / max(1, a1['total_elapsed'])
        t2 = a2['total_valid'] / max(1, a2['total_elapsed'])
        print(f"  Run 1: {t1:.2f} tasks/s")
        print(f"  Run 2: {t2:.2f} tasks/s")

        if t1 > 0:
            improvement = t2 / t1
            print(f"  Improvement: {improvement:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parallel NAS evaluation statistics"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to parallel logs directory (e.g., logs/exp/parallel)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Optional second log dir to compare against"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Directory not found: {log_dir}")
        return

    # Print analysis
    ws, bs = print_analysis(log_dir, args.log_dir)

    # Compare if second dir provided
    if args.compare:
        compare_dir = Path(args.compare)
        if compare_dir.exists():
            print_analysis(compare_dir, args.compare)
            compare_runs(log_dir, compare_dir)
        else:
            print(f"\nWarning: Compare directory not found: {compare_dir}")

    # JSON output
    if args.json:
        output = {
            'log_dir': str(log_dir),
            'worker_stats': analyze_worker_stats(ws) if ws else None,
            'batch_stats': analyze_batch_stats(bs) if bs else None
        }
        print("\n[JSON Output]")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
