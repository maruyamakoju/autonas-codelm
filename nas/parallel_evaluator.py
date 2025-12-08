"""
Multi-GPU Parallel Evaluator for NAS (Production Version)

Distributes architecture evaluation across multiple GPUs (e.g., 5090 + 4090)
using process-based parallelism to avoid GIL limitations.

Features:
- Process pool for true parallelism
- GPU assignment strategy (work-stealing queue)
- Robust error handling and timeout
- Worker statistics and logging
- Fault tolerance for failed evaluations
"""

import multiprocessing as mp
from multiprocessing import Queue, Process
import queue
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import traceback
import torch

from search_space import ArchitectureConfig
from evaluator import Evaluator, EvaluationResult
from fitness import FitnessConfig


@dataclass
class GPUProfile:
    """GPU profile with performance characteristics"""
    device: str
    name: str
    memory_gb: float
    compute_capability: str
    weight: float = 1.0  # Relative performance weight (higher = faster)

    def to_dict(self) -> Dict:
        return {
            "device": self.device,
            "name": self.name,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "weight": self.weight
        }


# Known GPU performance weights (relative to RTX 4090)
# These are estimated based on typical NAS workload performance
GPU_PERFORMANCE_WEIGHTS = {
    # RTX 50 series (Blackwell)
    "5090": 1.5,     # ~50% faster than 4090
    "5080": 1.2,
    "5070": 0.9,
    # RTX 40 series (Ada Lovelace)
    "4090": 1.0,     # Baseline
    "4080": 0.75,
    "4070": 0.55,
    "4060": 0.35,
    # RTX 30 series (Ampere)
    "3090": 0.7,
    "3080": 0.55,
    "3070": 0.4,
    # A100/H100 datacenter
    "A100": 1.3,
    "H100": 2.0,
}


def get_gpu_weight(gpu_name: str) -> float:
    """Get performance weight based on GPU name"""
    gpu_name_upper = gpu_name.upper()
    for key, weight in GPU_PERFORMANCE_WEIGHTS.items():
        if key in gpu_name_upper:
            return weight
    # Default weight for unknown GPUs
    return 1.0


@dataclass
class WorkerStats:
    """Statistics for a single worker"""
    worker_id: int
    device: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_eval_time_s: float = 0.0
    start_time: float = 0.0
    gpu_name: str = ""

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "device": self.device,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_eval_time_s": self.total_eval_time_s,
            "avg_eval_time_s": self.total_eval_time_s / max(1, self.tasks_completed),
            "uptime_s": time.time() - self.start_time if self.start_time else 0
        }


def worker_process(
    worker_id: int,
    device: str,
    task_queue: Queue,
    result_queue: Queue,
    stats_queue: Queue,
    use_real_training: bool,
    data_cfg_dict: Optional[Dict],
    max_train_steps: int,
    fitness_cfg_dict: Optional[Dict],
    log_dir: Optional[str]
):
    """
    Worker process that evaluates architectures on a specific GPU.

    Each worker:
    1. Creates its own Evaluator instance
    2. Pulls tasks from queue (work-stealing)
    3. Evaluates and pushes results
    4. Reports statistics periodically
    """
    # Set CUDA device
    if device.startswith("cuda:"):
        device_idx = int(device.split(":")[1])
        torch.cuda.set_device(device_idx)

    print(f"[Worker {worker_id}] Starting on {device}")

    # Initialize statistics
    stats = WorkerStats(
        worker_id=worker_id,
        device=device,
        start_time=time.time()
    )

    # Setup worker log directory
    worker_log_dir = None
    if log_dir:
        worker_log_dir = Path(log_dir) / f"worker_{worker_id}_{device.replace(':', '_')}"
        worker_log_dir.mkdir(parents=True, exist_ok=True)

    # Recreate configs from dicts (can't pickle dataclasses easily)
    data_cfg = None
    if data_cfg_dict:
        from datasets import CodeCharDatasetConfig
        data_cfg = CodeCharDatasetConfig(**data_cfg_dict)

    fitness_cfg = None
    if fitness_cfg_dict:
        fitness_cfg = FitnessConfig(**fitness_cfg_dict)

    # Create evaluator for this GPU
    evaluator = Evaluator(
        device=device,
        use_real_training=use_real_training,
        data_cfg=data_cfg,
        max_train_steps=max_train_steps,
        fitness_cfg=fitness_cfg,
        log_dir=str(worker_log_dir) if worker_log_dir else "logs/parallel"
    )

    eval_log = []  # Log of all evaluations

    while True:
        try:
            # Get task (blocks until available)
            task = task_queue.get(timeout=1.0)

            # Poison pill to stop worker
            if task is None:
                print(f"[Worker {worker_id}] Received shutdown signal")
                break

            task_id = task['task_id']
            arch_dict = task['architecture']

            # Reconstruct architecture config
            arch = ArchitectureConfig.from_dict(arch_dict)

            print(f"[Worker {worker_id}] Evaluating task {task_id} on {device}")

            t0 = time.time()
            result = None
            error = None

            try:
                result = evaluator.evaluate_fast(arch)
                stats.tasks_completed += 1
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                error_trace = traceback.format_exc()
                stats.tasks_failed += 1
                print(f"[Worker {worker_id}] ERROR on task {task_id}: {error}")
                print(f"[Worker {worker_id}] Traceback:\n{error_trace[:500]}")

            eval_time = time.time() - t0
            stats.total_eval_time_s += eval_time

            # Build result with full meta information
            result_dict = {
                'task_id': task_id,
                'result': result.to_dict() if result else None,
                'error': error,
                'meta': {
                    'worker_id': worker_id,
                    'device': device,
                    'arch_id': task_id,
                    'eval_time_s': eval_time,
                    'timestamp': time.time()
                }
            }
            result_queue.put(result_dict)

            # Log evaluation
            eval_log.append({
                'task_id': task_id,
                'arch_type': arch_dict.get('arch_type'),
                'num_layers': arch_dict.get('num_layers'),
                'hidden_dim': arch_dict.get('hidden_dim'),
                'eval_time_s': eval_time,
                'success': error is None,
                'fitness': result.fitness if result else None,
                'val_loss': result.val_loss if result else None
            })

        except queue.Empty:
            # No task available, continue waiting
            continue
        except Exception as e:
            print(f"[Worker {worker_id}] Fatal error: {e}")
            traceback.print_exc()
            break

    # Save final statistics
    if worker_log_dir:
        stats_file = worker_log_dir / "worker_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        eval_log_file = worker_log_dir / "eval_log.json"
        with open(eval_log_file, "w") as f:
            json.dump(eval_log, f, indent=2)

        print(f"[Worker {worker_id}] Saved stats to {stats_file}")

    # Send final stats to master
    stats_queue.put(stats.to_dict())

    print(f"[Worker {worker_id}] Shutting down (completed: {stats.tasks_completed}, "
          f"failed: {stats.tasks_failed}, total time: {stats.total_eval_time_s:.1f}s)")


class ParallelEvaluator:
    """
    Multi-GPU parallel evaluator (Production version)

    Usage:
        evaluator = ParallelEvaluator(
            devices=["cuda:0", "cuda:1"],
            use_real_training=True,
            data_cfg=data_cfg,
            max_train_steps=200,
            log_dir="logs/parallel_exp"
        )

        results = evaluator.evaluate_batch(architectures)

        evaluator.shutdown()
        evaluator.print_stats()
    """

    def __init__(
        self,
        devices: List[str] = None,
        use_real_training: bool = False,
        data_cfg = None,
        max_train_steps: int = 500,
        fitness_cfg: FitnessConfig = None,
        log_dir: str = None,
        max_eval_time_s: float = 1800.0,  # 30 minutes total timeout
        gpu_profiles: List[GPUProfile] = None  # Optional GPU profiles for heterogeneous scheduling
    ):
        """
        Initialize parallel evaluator with worker processes.

        Args:
            devices: List of GPU devices ["cuda:0", "cuda:1"]
            use_real_training: Whether to use real training
            data_cfg: Dataset configuration (CodeCharDatasetConfig)
            max_train_steps: Max training steps per architecture
            fitness_cfg: Fitness function configuration
            log_dir: Directory for worker logs
            max_eval_time_s: Maximum time for entire batch evaluation
            gpu_profiles: Optional GPU profiles for heterogeneous scheduling analysis
        """
        self.devices = devices or ["cuda:0"]
        self.gpu_profiles = gpu_profiles or []

        # Build device weight map from profiles
        self.device_weights = {}
        for p in self.gpu_profiles:
            self.device_weights[p.device] = p.weight
        # Default weight for devices without profiles
        for d in self.devices:
            if d not in self.device_weights:
                self.device_weights[d] = 1.0
        self.use_real_training = use_real_training
        self.max_train_steps = max_train_steps
        self.max_eval_time_s = max_eval_time_s
        self.log_dir = log_dir

        # Convert configs to dicts for pickling
        self.data_cfg_dict = None
        if data_cfg:
            self.data_cfg_dict = {
                'train_path': data_cfg.train_path,
                'val_path': data_cfg.val_path,
                'seq_len': data_cfg.seq_len,
                'batch_size': data_cfg.batch_size
            }

        self.fitness_cfg_dict = None
        if fitness_cfg:
            self.fitness_cfg_dict = fitness_cfg.to_dict()

        # Queues for communication
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.stats_queue = mp.Queue()

        # Worker statistics (collected after shutdown)
        self.worker_stats: List[Dict] = []

        # Batch statistics
        self.batch_stats: List[Dict] = []

        # Start worker processes
        self.workers = []
        for i, device in enumerate(self.devices):
            p = mp.Process(
                target=worker_process,
                args=(
                    i,
                    device,
                    self.task_queue,
                    self.result_queue,
                    self.stats_queue,
                    self.use_real_training,
                    self.data_cfg_dict,
                    self.max_train_steps,
                    self.fitness_cfg_dict,
                    self.log_dir
                )
            )
            p.start()
            self.workers.append(p)

        print(f"[ParallelEvaluator] Started {len(self.workers)} workers on {self.devices}")

    def evaluate_batch(
        self,
        architectures: List[ArchitectureConfig],
        timeout_per_task: float = None
    ) -> List[Optional[EvaluationResult]]:
        """
        Evaluate a batch of architectures in parallel.

        Args:
            architectures: List of architecture configs to evaluate
            timeout_per_task: Optional per-task timeout (uses global timeout if None)

        Returns:
            List of EvaluationResults (None for failed evaluations)
        """
        n = len(architectures)
        print(f"[ParallelEvaluator] Evaluating {n} architectures on {len(self.devices)} GPUs...")

        t0 = time.time()
        deadline = t0 + self.max_eval_time_s

        # Submit all tasks
        for i, arch in enumerate(architectures):
            task = {
                'task_id': i,
                'architecture': arch.to_dict()
            }
            self.task_queue.put(task)

        # Collect results
        results = [None] * n
        completed = 0
        errors = 0
        device_completed = {d: 0 for d in self.devices}

        while completed < n:
            # Calculate remaining time
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                print(f"[ParallelEvaluator] TIMEOUT: {completed}/{n} completed, "
                      f"{n - completed} tasks abandoned")
                break

            # Use smaller timeout for queue.get
            get_timeout = min(remaining_time, timeout_per_task or 300.0)

            try:
                result_dict = self.result_queue.get(timeout=get_timeout)
                task_id = result_dict['task_id']
                meta = result_dict.get('meta', {})
                device = meta.get('device', 'unknown')

                if result_dict.get('error'):
                    # Handle error
                    error_msg = result_dict.get('error', 'Unknown error')
                    print(f"[ParallelEvaluator] FAILED task {task_id} on {device}: {error_msg[:80]}")
                    errors += 1
                    results[task_id] = None
                elif result_dict['result']:
                    # Reconstruct EvaluationResult
                    r = result_dict['result']
                    arch = architectures[task_id]

                    result = EvaluationResult(
                        architecture=arch,
                        training_time_minutes=r['training_time_minutes'],
                        val_loss=r['raw_metrics']['val_loss'],
                        val_ppl=r['raw_metrics']['val_ppl'],
                        accuracy=r['raw_metrics']['accuracy'],
                        model_size_mb=r['raw_metrics']['model_size_mb'],
                        latency_ms=r['raw_metrics']['latency_ms'],
                        flops=r['raw_metrics']['flops'],
                        num_params=r['raw_metrics']['num_params'],
                        train_time_s=r['raw_metrics']['train_time_s'],
                        s_loss=r['normalized_scores']['s_loss'],
                        s_size=r['normalized_scores']['s_size'],
                        s_latency=r['normalized_scores']['s_latency'],
                        early_stopped=r['early_stopped'],
                        fitness=r['fitness']
                    )
                    results[task_id] = result
                    device_completed[device] = device_completed.get(device, 0) + 1

                    print(f"[ParallelEvaluator] Task {task_id} done on {device} "
                          f"({meta.get('eval_time_s', 0):.1f}s) -> Fitness: {result.fitness:.4f}")

                completed += 1

            except queue.Empty:
                print(f"[ParallelEvaluator] Waiting for results... ({completed}/{n} done)")
                continue
            except Exception as e:
                print(f"[ParallelEvaluator] Error collecting result: {e}")
                traceback.print_exc()
                break

        elapsed = time.time() - t0
        valid = sum(1 for r in results if r is not None)

        # Record batch stats
        batch_stat = {
            'timestamp': time.time(),
            'total_tasks': n,
            'completed': completed,
            'valid': valid,
            'errors': errors,
            'elapsed_s': elapsed,
            'avg_time_per_arch': elapsed / max(n, 1),
            'device_distribution': device_completed
        }
        self.batch_stats.append(batch_stat)

        print(f"\n[ParallelEvaluator] Batch Summary:")
        print(f"  Completed: {valid}/{n} ({errors} errors)")
        print(f"  Time: {elapsed:.1f}s ({elapsed/max(n,1):.1f}s/arch)")
        print(f"  Device distribution: {device_completed}")

        return results

    def shutdown(self):
        """Shutdown worker processes and collect statistics"""
        print("[ParallelEvaluator] Shutting down workers...")

        # Send poison pills
        for _ in self.workers:
            self.task_queue.put(None)

        # Collect worker statistics
        for _ in self.workers:
            try:
                stats = self.stats_queue.get(timeout=10)
                self.worker_stats.append(stats)
            except queue.Empty:
                pass

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                print(f"[ParallelEvaluator] Force terminating worker {p.pid}")
                p.terminate()

        print("[ParallelEvaluator] All workers shut down")

        # Save overall statistics
        if self.log_dir:
            self._save_stats()

    def _save_stats(self):
        """Save parallel evaluation statistics"""
        log_path = Path(self.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Save worker stats
        worker_stats_file = log_path / "parallel_worker_stats.json"
        with open(worker_stats_file, "w") as f:
            json.dump(self.worker_stats, f, indent=2)

        # Save batch stats
        batch_stats_file = log_path / "parallel_batch_stats.json"
        with open(batch_stats_file, "w") as f:
            json.dump(self.batch_stats, f, indent=2)

        print(f"[ParallelEvaluator] Stats saved to {log_path}")

    def print_stats(self):
        """Print summary statistics including heterogeneous scheduling analysis"""
        print("\n" + "="*60)
        print("PARALLEL EVALUATOR STATISTICS")
        print("="*60)

        if self.worker_stats:
            print("\n[Worker Statistics]")
            total_completed = 0
            total_failed = 0
            total_time = 0

            for ws in self.worker_stats:
                device = ws['device']
                weight = self.device_weights.get(device, 1.0)
                print(f"  Worker {ws['worker_id']} ({device}, weight={weight:.2f}x):")
                print(f"    Completed: {ws['tasks_completed']}, Failed: {ws['tasks_failed']}")
                print(f"    Total time: {ws['total_eval_time_s']:.1f}s, "
                      f"Avg: {ws['avg_eval_time_s']:.1f}s/task")
                total_completed += ws['tasks_completed']
                total_failed += ws['tasks_failed']
                total_time += ws['total_eval_time_s']

            print(f"\n[Total]")
            print(f"  Completed: {total_completed}, Failed: {total_failed}")
            print(f"  Worker time: {total_time:.1f}s")

            # Heterogeneous scheduling analysis
            if len(self.worker_stats) > 1 and self.gpu_profiles:
                print(f"\n[Heterogeneous Scheduling Analysis]")
                total_weight = sum(self.device_weights.get(ws['device'], 1.0) for ws in self.worker_stats)

                for ws in self.worker_stats:
                    device = ws['device']
                    weight = self.device_weights.get(device, 1.0)
                    expected_share = weight / total_weight
                    actual_share = ws['tasks_completed'] / max(1, total_completed)

                    # Calculate efficiency (actual vs expected)
                    efficiency = actual_share / max(0.01, expected_share) * 100

                    print(f"  {device}:")
                    print(f"    Expected share: {expected_share*100:.1f}%")
                    print(f"    Actual share: {actual_share*100:.1f}%")
                    print(f"    Scheduling efficiency: {efficiency:.1f}%")

        if self.batch_stats:
            print(f"\n[Batch Statistics]")
            for i, bs in enumerate(self.batch_stats):
                print(f"  Batch {i}: {bs['valid']}/{bs['total_tasks']} in {bs['elapsed_s']:.1f}s")


def detect_gpus() -> List[str]:
    """Detect available CUDA GPUs (simple version)"""
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available, using CPU")
        return ["cpu"]

    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")

    return devices


def detect_gpus_with_profiles() -> List[GPUProfile]:
    """
    Detect available CUDA GPUs with full profiles including performance weights.

    Returns list of GPUProfile objects with auto-detected weights based on GPU type.
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available")
        return []

    profiles = []
    print("\n" + "="*60)
    print("GPU DETECTION (Heterogeneous Scheduling)")
    print("="*60)

    total_weight = 0.0

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        mem_gb = props.total_memory / 1e9
        compute = f"{props.major}.{props.minor}"
        weight = get_gpu_weight(name)

        profile = GPUProfile(
            device=f"cuda:{i}",
            name=name,
            memory_gb=mem_gb,
            compute_capability=compute,
            weight=weight
        )
        profiles.append(profile)
        total_weight += weight

        print(f"\n  GPU {i}: {name}")
        print(f"    Device: cuda:{i}")
        print(f"    Memory: {mem_gb:.1f} GB")
        print(f"    Compute: {compute}")
        print(f"    Weight: {weight:.2f}x")

    if len(profiles) > 1:
        print(f"\n[Scheduling Info]")
        print(f"  Total weight: {total_weight:.2f}")
        for p in profiles:
            share = p.weight / total_weight * 100
            print(f"  {p.device} ({p.name}): {share:.1f}% of workload")

    print("="*60 + "\n")

    return profiles


def calculate_expected_distribution(profiles: List[GPUProfile], n_tasks: int) -> Dict[str, int]:
    """
    Calculate expected task distribution based on GPU weights.

    For heterogeneous GPUs (e.g., 5090 + 4090), faster GPUs should get more tasks.
    """
    total_weight = sum(p.weight for p in profiles)
    distribution = {}

    for p in profiles:
        # Calculate expected tasks (proportional to weight)
        expected = int(n_tasks * (p.weight / total_weight))
        distribution[p.device] = expected

    # Distribute remainder to highest weight GPU
    remainder = n_tasks - sum(distribution.values())
    if remainder > 0 and profiles:
        best_gpu = max(profiles, key=lambda p: p.weight)
        distribution[best_gpu.device] += remainder

    return distribution


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-GPU Parallel Evaluator Test")
    print("=" * 60)

    # Detect GPUs with full profiles
    print("\n[1] Detecting GPUs with profiles...")
    gpu_profiles = detect_gpus_with_profiles()

    if not gpu_profiles:
        print("No CUDA GPUs found. Using CPU.")
        devices = ["cpu"]
        gpu_profiles = []
    else:
        devices = [p.device for p in gpu_profiles]

    if len(devices) < 2:
        print("\nNOTE: Only one GPU found. For parallel speedup, need 2+ GPUs.")
        print("      System is ready for dual-GPU when second GPU is available.")
        print("\n[EXPECTED DUAL-GPU SETUP]")
        print("  cuda:0 (RTX 5090): weight=1.50x -> 60% of tasks")
        print("  cuda:1 (RTX 4090): weight=1.00x -> 40% of tasks")

    # Test with simple evaluation (no real training)
    print("\n[2] Testing parallel evaluation (simulated training)...")

    from search_space import SearchSpace

    space = SearchSpace(mode="minimal")

    # Generate test architectures
    test_archs = [space.sample_random() for _ in range(6)]

    # Create parallel evaluator with GPU profiles
    evaluator = ParallelEvaluator(
        devices=devices[:2] if len(devices) >= 2 else devices,
        use_real_training=False,
        max_train_steps=100,
        log_dir="logs/parallel_test",
        max_eval_time_s=300,
        gpu_profiles=gpu_profiles[:2] if len(gpu_profiles) >= 2 else gpu_profiles
    )

    # Evaluate
    results = evaluator.evaluate_batch(test_archs)

    # Print results
    print("\n[3] Results:")
    for i, r in enumerate(results):
        if r:
            print(f"  Arch {i}: L{r.architecture.num_layers} H{r.architecture.hidden_dim} "
                  f"-> Fitness: {r.fitness:.4f}")
        else:
            print(f"  Arch {i}: FAILED")

    # Shutdown and print stats
    evaluator.shutdown()
    evaluator.print_stats()

    # Show expected distribution for dual-GPU scenario
    if len(gpu_profiles) >= 2:
        print("\n[4] Heterogeneous Scheduling Preview:")
        expected = calculate_expected_distribution(gpu_profiles[:2], 100)
        for device, count in expected.items():
            profile = next((p for p in gpu_profiles if p.device == device), None)
            if profile:
                print(f"  {device} ({profile.name}): {count} tasks (weight={profile.weight:.2f})")

    print("\n[DONE] Parallel evaluator test complete!")
