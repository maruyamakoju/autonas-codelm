"""
Neural Architecture Search - Search Space Definition

MIT CS PhD レベルの探索空間設計
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import random
import json


@dataclass
class ArchitectureConfig:
    """アーキテクチャ設定"""

    # Architecture type
    arch_type: str  # "transformer", "linear_transformer", "mamba", etc.

    # Model size
    num_layers: int
    hidden_dim: int
    num_heads: int
    ffn_multiplier: float

    # Components
    normalization: str  # "layernorm", "rmsnorm", "groupnorm"
    activation: str  # "gelu", "silu", "geglu", "swiglu"
    position_encoding: str  # "absolute", "rope", "alibi"

    # Attention details
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Vocabulary
    vocab_size: int = 50000
    max_seq_length: int = 2048

    # Quantization (for final model)
    quantization: str = "fp16"  # "fp16", "int8", "int4"
    pruning_ratio: float = 0.0  # 0.0 to 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "arch_type": self.arch_type,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_multiplier": self.ffn_multiplier,
            "normalization": self.normalization,
            "activation": self.activation,
            "position_encoding": self.position_encoding,
            "attention_dropout": self.attention_dropout,
            "residual_dropout": self.residual_dropout,
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "quantization": self.quantization,
            "pruning_ratio": self.pruning_ratio
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ArchitectureConfig':
        """Create from dictionary"""
        return cls(**d)

    def estimate_parameters(self) -> int:
        """
        パラメータ数の推定

        Transformer の場合:
        - Embedding: vocab_size * hidden_dim
        - Each layer:
            - Attention: 4 * hidden_dim^2
            - FFN: 2 * hidden_dim * (ffn_multiplier * hidden_dim)
        """
        # Embedding
        params = self.vocab_size * self.hidden_dim

        # Layers
        for _ in range(self.num_layers):
            # Attention (Q, K, V, O)
            params += 4 * self.hidden_dim * self.hidden_dim

            # FFN
            ffn_dim = int(self.hidden_dim * self.ffn_multiplier)
            params += 2 * self.hidden_dim * ffn_dim

            # Layer norm (x2)
            params += 2 * self.hidden_dim

        # Final layer norm
        params += self.hidden_dim

        # LM head
        params += self.hidden_dim * self.vocab_size

        return params

    def estimate_size_mb(self) -> float:
        """
        モデルサイズの推定（MB）
        """
        params = self.estimate_parameters()

        # Quantization
        if self.quantization == "fp16":
            bytes_per_param = 2
        elif self.quantization == "int8":
            bytes_per_param = 1
        elif self.quantization == "int4":
            bytes_per_param = 0.5
        else:
            bytes_per_param = 4  # fp32

        # Pruning
        effective_params = params * (1.0 - self.pruning_ratio)

        size_bytes = effective_params * bytes_per_param
        size_mb = size_bytes / (1024 * 1024)

        return size_mb

    def is_valid(self) -> Tuple[bool, str]:
        """
        設定の妥当性チェック

        Returns:
            (is_valid, error_message)
        """
        # Hidden dim must be divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            return False, f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"

        # Model size check (target: 50-100MB)
        size_mb = self.estimate_size_mb()
        if size_mb > 500:  # 500MB を超えたら警告
            return False, f"Model size ({size_mb:.1f} MB) too large"

        return True, ""


class SearchSpace:
    """
    NAS 探索空間の定義

    MIT CS PhD レベルの設計:
    - 最新論文の知見を統合
    - 効率的な探索戦略
    - 理論的裏付け
    """

    def __init__(self, mode: str = "full"):
        """
        Args:
            mode: "minimal", "medium", "full"
                - minimal: 高速テスト用（10^3 通り）
                - medium: 中規模探索（10^6 通り）
                - full: 完全探索（10^9 通り）
        """
        self.mode = mode
        self.space = self._define_space(mode)

    def _define_space(self, mode: str) -> Dict[str, List[Any]]:
        """探索空間の定義"""

        if mode == "minimal":
            return {
                "arch_type": ["transformer"],
                "num_layers": [4, 6],
                "hidden_dim": [256, 512],
                "num_heads": [4, 8],
                "ffn_multiplier": [4.0],
                "normalization": ["layernorm"],
                "activation": ["gelu"],
                "position_encoding": ["absolute"],
                "attention_dropout": [0.1],
                "residual_dropout": [0.1],
                "quantization": ["fp16"],
                "pruning_ratio": [0.0]
            }

        elif mode == "medium":
            return {
                "arch_type": ["transformer", "linear_transformer"],
                "num_layers": [4, 6, 8, 12],
                "hidden_dim": [256, 384, 512, 768],
                "num_heads": [4, 6, 8, 12],
                "ffn_multiplier": [2.0, 3.0, 4.0],
                "normalization": ["layernorm", "rmsnorm"],
                "activation": ["gelu", "silu"],
                "position_encoding": ["absolute", "rope"],
                "attention_dropout": [0.0, 0.1],
                "residual_dropout": [0.0, 0.1],
                "quantization": ["fp16", "int8"],
                "pruning_ratio": [0.0, 0.2, 0.4]
            }

        else:  # full
            return {
                "arch_type": [
                    "transformer",
                    "linear_transformer",
                    "flash_attention",
                    "grouped_query_attention",
                    "mamba",
                    "rwkv"
                ],
                "num_layers": [2, 4, 6, 8, 12, 16],
                "hidden_dim": [128, 256, 384, 512, 768, 1024],
                "num_heads": [2, 4, 6, 8, 12, 16],
                "ffn_multiplier": [2.0, 2.5, 3.0, 4.0],
                "normalization": ["layernorm", "rmsnorm", "groupnorm"],
                "activation": ["gelu", "silu", "geglu", "swiglu"],
                "position_encoding": ["absolute", "rope", "alibi"],
                "attention_dropout": [0.0, 0.05, 0.1, 0.2],
                "residual_dropout": [0.0, 0.05, 0.1, 0.2],
                "quantization": ["fp16", "int8", "int4"],
                "pruning_ratio": [0.0, 0.2, 0.4, 0.6]
            }

    def sample_random(self) -> ArchitectureConfig:
        """ランダムにアーキテクチャをサンプリング"""
        config = {}
        for key, values in self.space.items():
            config[key] = random.choice(values)

        arch = ArchitectureConfig(**config)

        # Validity check
        is_valid, error = arch.is_valid()
        if not is_valid:
            # Retry with adjusted parameters
            if "hidden_dim" in error and "num_heads" in error:
                # Adjust num_heads to be divisible
                while config["hidden_dim"] % config["num_heads"] != 0:
                    config["num_heads"] = random.choice(self.space["num_heads"])
                arch = ArchitectureConfig(**config)

        return arch

    def sample_smart(self, base_config: ArchitectureConfig = None) -> ArchitectureConfig:
        """
        スマートサンプリング（既知の良い設定の近傍を探索）

        Args:
            base_config: 基準となる設定（Noneの場合はベストプラクティス）
        """
        if base_config is None:
            # Best practice baseline (LLaMA-like)
            base_config = ArchitectureConfig(
                arch_type="transformer",
                num_layers=6,
                hidden_dim=512,
                num_heads=8,
                ffn_multiplier=2.5,
                normalization="rmsnorm",
                activation="swiglu",
                position_encoding="rope"
            )

        # Mutate slightly
        config = base_config.to_dict()

        # Choose 1-3 parameters to mutate
        num_mutations = random.randint(1, 3)
        keys_to_mutate = random.sample(list(self.space.keys()), num_mutations)

        for key in keys_to_mutate:
            config[key] = random.choice(self.space[key])

        arch = ArchitectureConfig(**config)

        # Validity check
        is_valid, error = arch.is_valid()
        if not is_valid:
            # Fall back to random sampling
            return self.sample_random()

        return arch

    def get_search_space_size(self) -> int:
        """探索空間のサイズを計算"""
        size = 1
        for values in self.space.values():
            size *= len(values)
        return size


def get_baseline_architectures() -> List[ArchitectureConfig]:
    """
    ベースラインアーキテクチャ（既知の良い設定）

    Returns:
        List of baseline configurations
    """
    baselines = []

    # 1. GPT-2 Small-like (but smaller)
    baselines.append(ArchitectureConfig(
        arch_type="transformer",
        num_layers=6,
        hidden_dim=384,
        num_heads=6,
        ffn_multiplier=4.0,
        normalization="layernorm",
        activation="gelu",
        position_encoding="absolute"
    ))

    # 2. LLaMA-style (efficient)
    baselines.append(ArchitectureConfig(
        arch_type="transformer",
        num_layers=6,
        hidden_dim=512,
        num_heads=8,
        ffn_multiplier=2.5,
        normalization="rmsnorm",
        activation="swiglu",
        position_encoding="rope"
    ))

    # 3. Ultra-small (50MB target)
    baselines.append(ArchitectureConfig(
        arch_type="transformer",
        num_layers=4,
        hidden_dim=256,
        num_heads=4,
        ffn_multiplier=2.0,
        normalization="rmsnorm",
        activation="silu",
        position_encoding="rope",
        quantization="int8",
        pruning_ratio=0.4
    ))

    # 4. Mamba (State Space Model)
    baselines.append(ArchitectureConfig(
        arch_type="mamba",
        num_layers=6,
        hidden_dim=512,
        num_heads=8,  # Mamba doesn't use heads, but keep for compatibility
        ffn_multiplier=4.0,
        normalization="rmsnorm",
        activation="silu",
        position_encoding="absolute"  # SSM doesn't need positional encoding
    ))

    return baselines


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("NAS Search Space Test")
    print("=" * 60)

    # Test different modes
    for mode in ["minimal", "medium", "full"]:
        space = SearchSpace(mode=mode)
        size = space.get_search_space_size()
        print(f"\n{mode.upper()} mode: {size:,} possible configurations")

        # Sample random
        arch = space.sample_random()
        print(f"Random sample:")
        print(f"  Type: {arch.arch_type}")
        print(f"  Layers: {arch.num_layers}")
        print(f"  Hidden: {arch.hidden_dim}")
        print(f"  Parameters: {arch.estimate_parameters():,}")
        print(f"  Size: {arch.estimate_size_mb():.1f} MB")

    # Test baselines
    print("\n" + "=" * 60)
    print("Baseline Architectures")
    print("=" * 60)

    baselines = get_baseline_architectures()
    for i, arch in enumerate(baselines, 1):
        print(f"\nBaseline {i}: {arch.arch_type}")
        print(f"  Layers: {arch.num_layers}, Hidden: {arch.hidden_dim}")
        print(f"  Norm: {arch.normalization}, Act: {arch.activation}")
        print(f"  Parameters: {arch.estimate_parameters():,}")
        print(f"  Size: {arch.estimate_size_mb():.1f} MB")

        is_valid, error = arch.is_valid()
        print(f"  Valid: {is_valid}")
        if not is_valid:
            print(f"  Error: {error}")
