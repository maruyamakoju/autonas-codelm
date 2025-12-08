"""
Evolutionary Neural Architecture Search

遺伝的アルゴリズムによるアーキテクチャ探索:
- Population-based search
- Multi-objective optimization (accuracy, size, latency)
- Elite selection + crossover + mutation
- Parallel evaluation on multiple GPUs
"""

from typing import List, Tuple, Optional, Dict
import random
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

from search_space import ArchitectureConfig, SearchSpace
from evaluator import Evaluator, EvaluationResult
from parallel_evaluator import ParallelEvaluator, detect_gpus


@dataclass
class EvolutionConfig:
    """進化的探索の設定"""

    # Population
    population_size: int = 50
    num_generations: int = 100

    # Selection
    elite_ratio: float = 0.2  # Top 20% survive
    tournament_size: int = 3

    # Genetic operators
    mutation_rate: float = 0.3  # 30% of genes mutate
    crossover_rate: float = 0.7  # 70% chance of crossover

    # Evaluation
    evaluation_mode: str = "fast"  # "fast", "medium", "full"
    parallel_gpus: List[str] = None  # ["cuda:0", "cuda:1"]

    # Logging
    log_dir: str = "logs/evolution"
    save_frequency: int = 5  # Save every 5 generations

    def __post_init__(self):
        if self.parallel_gpus is None:
            self.parallel_gpus = ["cuda:0"]


class EvolutionaryNAS:
    """
    遺伝的アルゴリズムによるNAS

    アルゴリズム:
    1. 初期集団をランダムサンプリング
    2. 各個体を評価（accuracy, size, latency）
    3. エリート選択（上位20%）
    4. トーナメント選択で親を選ぶ
    5. 交叉と突然変異で子世代を生成
    6. 次世代へ
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Evaluator,
        config: EvolutionConfig = None,
        parallel_evaluator: ParallelEvaluator = None
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.parallel_evaluator = parallel_evaluator
        self.config = config or EvolutionConfig()

        # Logging
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Evolution state
        self.generation = 0
        self.population: List[ArchitectureConfig] = []
        self.fitness_history: List[Dict] = []
        self.best_architecture: Optional[EvaluationResult] = None

        if self.parallel_evaluator:
            print(f"[NAS] Using parallel evaluation on {len(self.parallel_evaluator.devices)} GPUs")

    def initialize_population(self) -> List[ArchitectureConfig]:
        """
        初期集団の生成

        戦略:
        - 50% ランダムサンプリング
        - 50% スマートサンプリング（既知の良い設定の近傍）
        """
        population = []

        # Random sampling
        num_random = self.config.population_size // 2
        for _ in range(num_random):
            arch = self.search_space.sample_random()
            population.append(arch)

        # Smart sampling (near good baselines)
        num_smart = self.config.population_size - num_random
        for _ in range(num_smart):
            arch = self.search_space.sample_smart()
            population.append(arch)

        print(f"Initialized population: {len(population)} architectures")
        return population

    def evaluate_population(
        self,
        population: List[ArchitectureConfig]
    ) -> List[EvaluationResult]:
        """
        集団の評価

        Uses parallel evaluation if ParallelEvaluator is available,
        otherwise falls back to sequential evaluation.

        Returns:
            List of EvaluationResult (never None - failed evals get fitness=0)
        """
        # Use parallel evaluation if available
        if self.parallel_evaluator:
            print(f"\n[Gen {self.generation}] Parallel evaluation of {len(population)} architectures...")
            raw_results = self.parallel_evaluator.evaluate_batch(population)
        else:
            # Sequential evaluation (single GPU)
            raw_results = []
            for i, arch in enumerate(population, 1):
                print(f"\n[Gen {self.generation}] Evaluating {i}/{len(population)}...")

                # Evaluate based on mode
                if self.config.evaluation_mode == "fast":
                    result = self.evaluator.evaluate_fast(arch)
                elif self.config.evaluation_mode == "medium":
                    result = self.evaluator.evaluate_medium(arch)
                else:
                    result = self.evaluator.evaluate_full(arch)

                raw_results.append(result)

        # Process results: convert None to dummy with fitness=0
        results = []
        failed_indices = []

        for idx, (arch, raw_result) in enumerate(zip(population, raw_results)):
            if raw_result is None:
                # Failed evaluation - create dummy result with fitness=0
                failed_indices.append(idx)
                dummy = EvaluationResult(
                    architecture=arch,
                    training_time_minutes=0.0,
                    val_loss=float('inf'),
                    val_ppl=float('inf'),
                    fitness=0.0,
                    early_stopped=True
                )
                results.append(dummy)
            else:
                results.append(raw_result)
                # Track best
                if (self.best_architecture is None or
                    raw_result.fitness > self.best_architecture.fitness):
                    self.best_architecture = raw_result
                    print(f"  *** NEW BEST: Fitness {raw_result.fitness:.3f}")

        # Report failures
        if failed_indices:
            print(f"\n[Gen {self.generation}] WARNING: {len(failed_indices)} evaluations failed: {failed_indices}")

        return results

    def select_elite(
        self,
        population: List[ArchitectureConfig],
        results: List[Optional[EvaluationResult]]
    ) -> Tuple[List[ArchitectureConfig], List[EvaluationResult]]:
        """
        エリート選択（上位N%を保存）

        Returns:
            (elite_architectures, elite_results)
        """
        # Filter out None results (early stopped)
        valid_pairs = [
            (arch, result)
            for arch, result in zip(population, results)
            if result is not None
        ]

        if not valid_pairs:
            print("⚠️  No valid architectures in population!")
            return [], []

        # Sort by fitness (descending)
        valid_pairs.sort(key=lambda x: x[1].fitness, reverse=True)

        # Select top N%
        num_elite = max(1, int(len(valid_pairs) * self.config.elite_ratio))
        elite_pairs = valid_pairs[:num_elite]

        elite_archs = [arch for arch, _ in elite_pairs]
        elite_results = [result for _, result in elite_pairs]

        print(f"\n[ELITE] Selected {len(elite_archs)} architectures")
        print(f"   Best fitness: {elite_results[0].fitness:.3f}")
        print(f"   Worst elite fitness: {elite_results[-1].fitness:.3f}")

        return elite_archs, elite_results

    def tournament_selection(
        self,
        population: List[ArchitectureConfig],
        results: List[EvaluationResult],
        k: int = None
    ) -> ArchitectureConfig:
        """
        トーナメント選択

        Args:
            population: 候補アーキテクチャ
            results: 評価結果
            k: トーナメントサイズ

        Returns:
            選ばれたアーキテクチャ
        """
        if k is None:
            k = self.config.tournament_size

        # Ensure k doesn't exceed population size
        k = min(k, len(population))

        # Randomly select k individuals
        indices = random.sample(range(len(population)), k)
        tournament = [(population[i], results[i]) for i in indices]

        # Select best from tournament
        winner = max(tournament, key=lambda x: x[1].fitness)
        return winner[0]

    def crossover(
        self,
        parent1: ArchitectureConfig,
        parent2: ArchitectureConfig
    ) -> ArchitectureConfig:
        """
        交叉（2点交叉）

        Args:
            parent1, parent2: 親アーキテクチャ

        Returns:
            子アーキテクチャ
        """
        # Convert to dicts
        p1_dict = parent1.to_dict()
        p2_dict = parent2.to_dict()

        # Create child by randomly selecting from each parent
        child_dict = {}
        for key in p1_dict.keys():
            if random.random() < 0.5:
                child_dict[key] = p1_dict[key]
            else:
                child_dict[key] = p2_dict[key]

        child = ArchitectureConfig.from_dict(child_dict)

        # Validity check and fix
        is_valid, error = child.is_valid()
        if not is_valid:
            # Fix common issues
            if "hidden_dim" in error and "num_heads" in error:
                # Make hidden_dim divisible by num_heads
                while child.hidden_dim % child.num_heads != 0:
                    child.num_heads = random.choice(self.search_space.space["num_heads"])

        return child

    def mutate(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """
        突然変異

        Args:
            architecture: アーキテクチャ

        Returns:
            変異後のアーキテクチャ
        """
        arch_dict = architecture.to_dict()

        # Mutate each gene with mutation_rate probability
        for key in arch_dict.keys():
            if random.random() < self.config.mutation_rate:
                # Replace with random value from search space
                if key in self.search_space.space:
                    arch_dict[key] = random.choice(self.search_space.space[key])

        mutated = ArchitectureConfig.from_dict(arch_dict)

        # Validity check and fix
        is_valid, error = mutated.is_valid()
        if not is_valid:
            if "hidden_dim" in error and "num_heads" in error:
                while mutated.hidden_dim % mutated.num_heads != 0:
                    mutated.num_heads = random.choice(self.search_space.space["num_heads"])

        return mutated

    def create_offspring(
        self,
        parents: List[ArchitectureConfig],
        parent_results: List[EvaluationResult],
        num_offspring: int
    ) -> List[ArchitectureConfig]:
        """
        子世代の生成

        Args:
            parents: 親世代
            parent_results: 親の評価結果
            num_offspring: 生成する子の数

        Returns:
            子世代
        """
        offspring = []

        for _ in range(num_offspring):
            # Crossover
            if random.random() < self.config.crossover_rate and len(parents) >= 2:
                # Tournament selection for parents
                parent1 = self.tournament_selection(parents, parent_results)
                parent2 = self.tournament_selection(parents, parent_results)
                child = self.crossover(parent1, parent2)
            else:
                # Just copy a parent
                parent = self.tournament_selection(parents, parent_results)
                child = ArchitectureConfig.from_dict(parent.to_dict())

            # Mutation
            child = self.mutate(child)

            offspring.append(child)

        return offspring

    def evolve_generation(self) -> Dict:
        """
        1世代の進化

        Returns:
            世代の統計情報
        """
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")

        start_time = time.time()

        # Evaluate population
        results = self.evaluate_population(self.population)

        # Select elite
        elite_archs, elite_results = self.select_elite(self.population, results)

        if not elite_archs:
            print("WARNING: No elite architectures! Re-initializing population...")
            self.population = self.initialize_population()
            return {}

        # Create offspring
        num_offspring = self.config.population_size - len(elite_archs)
        offspring = self.create_offspring(elite_archs, elite_results, num_offspring)

        # Next generation = elite + offspring
        self.population = elite_archs + offspring

        # Statistics
        valid_results = [r for r in results if r is not None]
        stats = {
            "generation": self.generation,
            "num_evaluated": len(results),
            "num_valid": len(valid_results),
            "num_elite": len(elite_archs),
            "best_fitness": max(r.fitness for r in valid_results) if valid_results else 0.0,
            "mean_fitness": np.mean([r.fitness for r in valid_results]) if valid_results else 0.0,
            "std_fitness": np.std([r.fitness for r in valid_results]) if valid_results else 0.0,
            "time_minutes": (time.time() - start_time) / 60.0
        }

        self.fitness_history.append(stats)

        # Log
        print(f"\n[SUMMARY] Generation {self.generation}:")
        print(f"   Valid: {stats['num_valid']}/{stats['num_evaluated']}")
        print(f"   Best fitness: {stats['best_fitness']:.3f}")
        print(f"   Mean fitness: {stats['mean_fitness']:.3f} +/- {stats['std_fitness']:.3f}")
        print(f"   Time: {stats['time_minutes']:.1f} min")

        # Save
        if self.generation % self.config.save_frequency == 0:
            self.save_checkpoint()

        self.generation += 1

        return stats

    def run(self, num_generations: int = None) -> EvaluationResult:
        """
        進化的探索の実行

        Args:
            num_generations: 世代数（Noneならconfig使用）

        Returns:
            最良アーキテクチャの評価結果
        """
        if num_generations is None:
            num_generations = self.config.num_generations

        print(f"\n{'='*60}")
        print(f"Starting Evolutionary NAS")
        print(f"{'='*60}")
        print(f"Population size: {self.config.population_size}")
        print(f"Generations: {num_generations}")
        print(f"Evaluation mode: {self.config.evaluation_mode}")
        print(f"Search space size: {self.search_space.get_search_space_size():,}")

        # Initialize
        self.population = self.initialize_population()

        # Evolve
        for gen in range(num_generations):
            stats = self.evolve_generation()

            if not stats:
                print("WARNING: Evolution failed, stopping...")
                break

        # Final results
        print(f"\n{'='*60}")
        print(f"Evolution Complete!")
        print(f"{'='*60}")

        if self.best_architecture:
            print(f"\n[BEST ARCHITECTURE]")
            print(f"   Type: {self.best_architecture.architecture.arch_type}")
            print(f"   Layers: {self.best_architecture.architecture.num_layers}")
            print(f"   Hidden: {self.best_architecture.architecture.hidden_dim}")
            print(f"   Heads: {self.best_architecture.architecture.num_heads}")
            print(f"   Accuracy: {self.best_architecture.accuracy:.3f}")
            print(f"   Size: {self.best_architecture.model_size_mb:.1f} MB")
            print(f"   Latency: {self.best_architecture.latency_ms:.2f} ms")
            print(f"   Fitness: {self.best_architecture.fitness:.3f}")

            # Save best
            self.save_best_architecture()

        return self.best_architecture

    def save_checkpoint(self):
        """チェックポイントの保存"""
        checkpoint = {
            "generation": self.generation,
            "config": asdict(self.config),
            "fitness_history": self.fitness_history,
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None
        }

        filepath = self.log_dir / f"checkpoint_gen{self.generation}.json"
        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"   [SAVE] Checkpoint: {filepath}")

    def save_best_architecture(self):
        """最良アーキテクチャの保存"""
        if not self.best_architecture:
            return

        filepath = self.log_dir / "best_architecture.json"
        with open(filepath, "w") as f:
            json.dump(self.best_architecture.to_dict(), f, indent=2)

        print(f"\n[SAVE] Best architecture: {filepath}")


if __name__ == "__main__":
    import torch
    import argparse

    parser = argparse.ArgumentParser(description="Evolutionary NAS")
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--use_real_training", action="store_true")
    parser.add_argument("--train_path", type=str, default="../data/code_char/train.txt")
    parser.add_argument("--val_path", type=str, default="../data/code_char/val.txt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--search_mode", type=str, default="minimal", choices=["minimal", "medium", "full"])
    parser.add_argument("--parallel", action="store_true", help="Use multi-GPU parallel evaluation")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU list (e.g., 'cuda:0,cuda:1')")
    args = parser.parse_args()

    print("="*60)
    print("Evolutionary NAS")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Real training: {args.use_real_training}")
    print(f"Device: {args.device}")
    print(f"Parallel: {args.parallel}")

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"

    search_space = SearchSpace(mode=args.search_mode)

    # Evaluator configuration
    if args.use_real_training:
        from datasets import CodeCharDatasetConfig

        data_cfg = CodeCharDatasetConfig(
            train_path=args.train_path,
            val_path=args.val_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )

        evaluator = Evaluator(
            device=device,
            log_dir=f"logs/{args.experiment_name}",
            use_real_training=True,
            data_cfg=data_cfg,
            max_train_steps=args.max_train_steps
        )
    else:
        evaluator = Evaluator(device=device, log_dir=f"logs/{args.experiment_name}")

    config = EvolutionConfig(
        population_size=args.population,
        num_generations=args.generations,
        elite_ratio=0.2,
        mutation_rate=0.3,
        evaluation_mode="fast",
        log_dir=f"logs/{args.experiment_name}/evolution"
    )

    # Setup parallel evaluator if requested
    parallel_eval = None
    if args.parallel:
        from parallel_evaluator import detect_gpus_with_profiles, GPUProfile

        # Detect GPUs with full profiles for heterogeneous scheduling
        gpu_profiles = detect_gpus_with_profiles()

        # Override with user-specified GPUs if provided
        if args.gpus:
            devices = [g.strip() for g in args.gpus.split(",")]
            # Filter profiles to match specified devices
            gpu_profiles = [p for p in gpu_profiles if p.device in devices]
        else:
            devices = [p.device for p in gpu_profiles] if gpu_profiles else detect_gpus()

        print(f"\n[PARALLEL] Setting up multi-GPU evaluation on {devices}")

        parallel_log_dir = f"logs/{args.experiment_name}/parallel"

        if args.use_real_training:
            parallel_eval = ParallelEvaluator(
                devices=devices,
                use_real_training=True,
                data_cfg=data_cfg,
                max_train_steps=args.max_train_steps,
                log_dir=parallel_log_dir,
                max_eval_time_s=3600.0,  # 1 hour timeout per batch
                gpu_profiles=gpu_profiles  # Enable heterogeneous scheduling
            )
        else:
            parallel_eval = ParallelEvaluator(
                devices=devices,
                use_real_training=False,
                max_train_steps=args.max_train_steps,
                log_dir=parallel_log_dir,
                max_eval_time_s=1800.0,  # 30 min timeout
                gpu_profiles=gpu_profiles  # Enable heterogeneous scheduling
            )

    # Run evolution
    nas = EvolutionaryNAS(
        search_space=search_space,
        evaluator=evaluator,
        config=config,
        parallel_evaluator=parallel_eval
    )

    try:
        best = nas.run()
    finally:
        # Clean up parallel evaluator and print stats
        if parallel_eval:
            parallel_eval.shutdown()
            parallel_eval.print_stats()

    # Plot fitness history (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        generations = [h["generation"] for h in nas.fitness_history]
        best_fitness = [h["best_fitness"] for h in nas.fitness_history]
        mean_fitness = [h["mean_fitness"] for h in nas.fitness_history]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
        plt.plot(generations, mean_fitness, 'r--', label='Mean', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = config.log_dir + "/fitness_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n[PLOT] Saved fitness history: {plot_path}")

    except ImportError:
        print("\nMatplotlib not installed, skipping plot")
