"""
NAS Evaluator - アーキテクチャの評価システム

評価指標:
- Val Loss / Perplexity
- Model Size (MB)
- Inference Latency (ms)
- FLOPs

Multi-objective fitness with normalized scores
"""

from typing import Dict, List, Tuple, Optional
import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import json
from pathlib import Path

from search_space import ArchitectureConfig
from fitness import FitnessFunction, FitnessConfig, FitnessResult
from typing import TYPE_CHECKING


@dataclass
class EvaluationResult:
    """評価結果 with research-grade fitness scoring"""

    # Metadata (required fields first, with defaults for dataclass compatibility)
    architecture: Optional[ArchitectureConfig] = None
    training_time_minutes: float = 0.0

    # Raw metrics
    val_loss: float = 0.0
    val_ppl: float = 0.0
    accuracy: float = 0.0  # Legacy / converted from val_loss
    model_size_mb: float = 0.0
    latency_ms: float = 0.0
    flops: int = 0
    num_params: int = 0
    train_time_s: float = 0.0

    # Normalized scores (0-1, higher is better)
    s_loss: float = 0.0
    s_size: float = 0.0
    s_latency: float = 0.0

    # Other metadata
    early_stopped: bool = False

    # Fitness score (weighted combination)
    fitness: float = 0.0

    def compute_fitness(self, fitness_fn: FitnessFunction = None) -> float:
        """
        適応度スコアの計算 using research-grade fitness function

        Uses FitnessFunction from fitness.py for normalized scoring
        """
        if fitness_fn is None:
            fitness_fn = FitnessFunction()

        metrics = {
            "val_loss": self.val_loss,
            "val_ppl": self.val_ppl,
            "num_params": self.num_params,
            "model_size_mb": self.model_size_mb,
            "latency_ms": self.latency_ms,
            "train_time_s": self.train_time_s
        }

        result = fitness_fn.compute(metrics)

        # Update scores
        self.s_loss = result.s_loss
        self.s_size = result.s_size
        self.s_latency = result.s_latency
        self.fitness = result.fitness

        return self.fitness

    def to_dict(self) -> Dict:
        """Convert to dictionary with full metrics"""
        return {
            "raw_metrics": {
                "val_loss": self.val_loss,
                "val_ppl": self.val_ppl,
                "accuracy": self.accuracy,
                "model_size_mb": self.model_size_mb,
                "latency_ms": self.latency_ms,
                "flops": self.flops,
                "num_params": self.num_params,
                "train_time_s": self.train_time_s
            },
            "normalized_scores": {
                "s_loss": self.s_loss,
                "s_size": self.s_size,
                "s_latency": self.s_latency
            },
            "fitness": self.fitness,
            "training_time_minutes": self.training_time_minutes,
            "early_stopped": self.early_stopped,
            "architecture": self.architecture.to_dict() if self.architecture else None
        }


class Evaluator:
    """
    アーキテクチャ評価器

    評価戦略:
    1. 小規模評価（1000サンプル、5分）- スクリーニング
    2. 中規模評価（10000サンプル、20分）- 有望候補の選定
    3. フル評価（100000サンプル、60分）- 最終評価
    """

    def __init__(
        self,
        dataset_name: str = "code_dataset",
        device: str = "cuda:0",
        log_dir: str = "logs/nas",
        use_real_training: bool = False,
        data_cfg: Optional['CodeCharDatasetConfig'] = None,
        max_train_steps: int = 500,
        fitness_cfg: Optional[FitnessConfig] = None
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Real training configuration
        self.use_real_training = use_real_training
        self.data_cfg = data_cfg
        self.max_train_steps = max_train_steps

        # Fitness configuration
        self.fitness_cfg = fitness_cfg or FitnessConfig()

        # Legacy (for simulated training)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def evaluate_fast(
        self,
        config: ArchitectureConfig,
        num_samples: int = 1000,
        num_epochs: int = 5
    ) -> Optional[EvaluationResult]:
        """
        高速評価（スクリーニング用）

        Args:
            config: Architecture configuration
            num_samples: Number of training samples
            num_epochs: Number of training epochs

        Returns:
            EvaluationResult or None (if early stopped)
        """
        print(f"\n[FAST EVAL] {config.arch_type}, L{config.num_layers}, H{config.hidden_dim}")

        start_time = time.time()

        # Estimate size (quick check)
        model_size_mb = config.estimate_size_mb()

        # Quick check: if model is too large, skip
        if model_size_mb > 500:
            print(f"  → Skipped (too large: {model_size_mb:.1f} MB)")
            return None

        # Train: real or simulated
        if self.use_real_training and self.data_cfg is not None:
            # Real training
            from train_loop import train_one_architecture

            metrics = train_one_architecture(
                config,
                self.data_cfg,
                device=self.device,
                max_steps=self.max_train_steps,
                lr=3e-4,
                log_interval=max(self.max_train_steps // 5, 50)
            )

            val_loss = metrics['val_loss']
            val_ppl = metrics['val_ppl']
            num_params = int(metrics['num_params'])
            latency_ms = metrics['latency_ms']
            model_size_mb = metrics['model_size_mb']
            train_time_s = metrics['train_time_s']
            flops = 0
            # accuracy for backward compatibility
            accuracy = min(1.0, max(0.0, torch.exp(-torch.tensor(val_loss)).item()))

        else:
            # Simulated training (legacy)
            model = self._build_model(config)

            accuracy = self._train_simulated(
                model, config, num_samples, num_epochs
            )

            latency_ms = self._measure_latency(model, config)
            flops = self._estimate_flops(config)
            # Simulate val_loss from accuracy
            val_loss = -torch.log(torch.tensor(max(0.01, accuracy))).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            num_params = sum(p.numel() for p in model.parameters())
            train_time_s = 0.0

        training_time = (time.time() - start_time) / 60.0  # minutes

        result = EvaluationResult(
            val_loss=val_loss,
            val_ppl=val_ppl,
            accuracy=accuracy,
            model_size_mb=model_size_mb,
            latency_ms=latency_ms,
            flops=flops,
            num_params=num_params,
            train_time_s=train_time_s,
            architecture=config,
            training_time_minutes=training_time,
            early_stopped=False
        )

        # Compute fitness with research-grade fitness function
        fitness_fn = FitnessFunction(self.fitness_cfg)
        result.compute_fitness(fitness_fn)

        print(f"  -> Loss: {val_loss:.3f} (s={result.s_loss:.3f}), "
              f"Size: {model_size_mb:.1f}MB (s={result.s_size:.3f}), "
              f"Lat: {latency_ms:.2f}ms (s={result.s_latency:.3f})")
        print(f"  -> Fitness: {result.fitness:.4f} "
              f"[w_loss={self.fitness_cfg.w_loss}, w_size={self.fitness_cfg.w_size}, w_lat={self.fitness_cfg.w_latency}]")

        return result

    def evaluate_medium(
        self,
        config: ArchitectureConfig,
        num_samples: int = 10000,
        num_epochs: int = 10
    ) -> EvaluationResult:
        """中規模評価"""
        # Similar to fast, but more thorough
        pass

    def evaluate_full(
        self,
        config: ArchitectureConfig,
        num_samples: int = 100000,
        num_epochs: int = 20
    ) -> EvaluationResult:
        """完全評価"""
        # Full training and evaluation
        pass

    def _build_model(self, config: ArchitectureConfig) -> nn.Module:
        """
        モデル構築

        Uses models.py implementation
        """
        from models import build_model

        try:
            model = build_model(config)
            return model.to(self.device)
        except NotImplementedError:
            # Fall back to simple model for unsupported architectures
            print(f"  [WARNING] {config.arch_type} not implemented, using simple model")
            return self._build_simple_model(config)

    def _build_simple_model(self, config: ArchitectureConfig) -> nn.Module:
        """Simple fallback model for unsupported architectures"""
        class SimpleModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
                self.layers = nn.ModuleList([
                    nn.Linear(config.hidden_dim, config.hidden_dim)
                    for _ in range(config.num_layers)
                ])
                self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

            def forward(self, x):
                x = self.embedding(x)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        return SimpleModel(config).to(self.device)

    def _train_simulated(
        self,
        model: nn.Module,
        config: ArchitectureConfig,
        num_samples: int,
        num_epochs: int
    ) -> float:
        """
        訓練のシミュレーション（実装は後で）

        今はランダムな精度を返す（テスト用）

        TODO: 実際の訓練実装
        """
        import random

        # Simulate training time
        time.sleep(0.1)

        # Random accuracy (for testing)
        # 実際は訓練して本物の精度を測定
        base_accuracy = 0.3
        layer_bonus = config.num_layers * 0.02
        dim_bonus = (config.hidden_dim / 1024) * 0.1

        accuracy = base_accuracy + layer_bonus + dim_bonus
        accuracy += random.uniform(-0.05, 0.05)  # noise
        accuracy = max(0.0, min(1.0, accuracy))

        return accuracy

    def _measure_latency(
        self,
        model: nn.Module,
        config: ArchitectureConfig,
        num_runs: int = 100
    ) -> float:
        """
        推論レイテンシの測定

        Args:
            model: Model to measure
            config: Architecture config
            num_runs: Number of runs for averaging

        Returns:
            Average latency in milliseconds
        """
        model.eval()

        # Dummy input
        batch_size = 1
        seq_length = 128
        dummy_input = torch.randint(
            0, config.vocab_size, (batch_size, seq_length)
        ).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_latency_ms = (end_time - start_time) / num_runs * 1000

        return avg_latency_ms

    def _estimate_flops(self, config: ArchitectureConfig) -> int:
        """
        FLOPs の推定

        Transformer の場合（1トークンあたり）:
        - Attention: 2 * seq_len * hidden_dim^2
        - FFN: 2 * 2 * hidden_dim * ffn_dim

        Returns:
            Estimated FLOPs for one forward pass
        """
        seq_len = 128  # assume

        flops = 0

        for _ in range(config.num_layers):
            # Attention
            flops += 2 * seq_len * config.hidden_dim ** 2

            # FFN
            ffn_dim = int(config.hidden_dim * config.ffn_multiplier)
            flops += 2 * 2 * config.hidden_dim * ffn_dim

        return flops

    def save_result(self, result: EvaluationResult, filename: str = None):
        """評価結果の保存"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"eval_{timestamp}.json"

        filepath = self.log_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved evaluation result to {filepath}")


def compare_architectures(results: List[EvaluationResult]):
    """
    アーキテクチャの比較（Pandas DataFrame）

    Args:
        results: List of evaluation results

    Returns:
        DataFrame with comparison (or None if pandas not installed)
    """
    try:
        import pandas as pd

        data = []
        for r in results:
            data.append({
                "arch_type": r.architecture.arch_type,
                "layers": r.architecture.num_layers,
                "hidden": r.architecture.hidden_dim,
                "accuracy": r.accuracy,
                "size_mb": r.model_size_mb,
                "latency_ms": r.latency_ms,
                "fitness": r.fitness
            })

        df = pd.DataFrame(data)
        df = df.sort_values("fitness", ascending=False)
        return df

    except ImportError:
        print("Pandas not installed. Install with: pip install pandas")
        return None


if __name__ == "__main__":
    from search_space import get_baseline_architectures, SearchSpace

    print("=" * 60)
    print("NAS Evaluator Test")
    print("=" * 60)

    evaluator = Evaluator(device="cuda:0" if torch.cuda.is_available() else "cpu")

    # Test with baselines
    baselines = get_baseline_architectures()

    results = []
    for i, config in enumerate(baselines, 1):
        print(f"\n[{i}/{len(baselines)}] Evaluating baseline...")
        result = evaluator.evaluate_fast(config)

        if result:
            results.append(result)
            evaluator.save_result(result, f"baseline_{i}.json")

    # Test with random architectures
    space = SearchSpace(mode="minimal")

    print("\n" + "=" * 60)
    print("Random Architecture Evaluation")
    print("=" * 60)

    for i in range(5):
        config = space.sample_random()
        print(f"\n[{i+1}/5] Evaluating random architecture...")
        result = evaluator.evaluate_fast(config)

        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    results.sort(key=lambda r: r.fitness, reverse=True)

    for i, r in enumerate(results[:5], 1):
        print(f"\nRank {i}:")
        print(f"  Type: {r.architecture.arch_type}")
        print(f"  Size: L{r.architecture.num_layers} H{r.architecture.hidden_dim}")
        print(f"  Accuracy: {r.accuracy:.3f}")
        print(f"  Size: {r.model_size_mb:.1f} MB")
        print(f"  Latency: {r.latency_ms:.2f} ms")
        print(f"  Fitness: {r.fitness:.3f}")
