# Neural Architecture Search - Design Document

## 概要

超軽量コード理解モデルのための自律NASシステム設計。

**目標**: GPT-4 レベルの性能を 1/100 のサイズで実現するアーキテクチャを自動発見。

---

## 探索空間の定義

### 1. アーキテクチャタイプ

```python
ARCHITECTURE_TYPES = {
    "transformer": {
        "description": "標準的な Transformer",
        "pros": "汎用性高い、実装豊富",
        "cons": "計算量 O(n²)",
        "papers": ["Attention is All You Need (2017)"]
    },

    "linear_transformer": {
        "description": "Linear Attention（O(n)）",
        "pros": "長文対応、計算効率",
        "cons": "性能劣化の可能性",
        "papers": [
            "Transformers are RNNs (2020)",
            "Linear Transformers Are Secretly Fast Weight Programmers (2021)"
        ]
    },

    "flash_attention": {
        "description": "FlashAttention（メモリ最適化）",
        "pros": "メモリ効率、速度",
        "cons": "カスタムCUDA必要",
        "papers": [
            "FlashAttention (2022)",
            "FlashAttention-2 (2023)"
        ]
    },

    "grouped_query_attention": {
        "description": "GQA（Google PaLM 2）",
        "pros": "推論速度、メモリ削減",
        "cons": "やや複雑",
        "papers": ["GQA: Training Generalized Multi-Query Transformer Models (2023)"]
    },

    "mamba": {
        "description": "State Space Model",
        "pros": "O(n) 計算、長文対応",
        "cons": "新しい、実装少ない",
        "papers": ["Mamba: Linear-Time Sequence Modeling (2023)"]
    },

    "rwkv": {
        "description": "RNN-like Transformer",
        "pros": "推論速度、メモリ効率",
        "cons": "訓練難しい",
        "papers": ["RWKV: Reinventing RNNs for the Transformer Era (2023)"]
    },

    "hybrid": {
        "description": "上記の組み合わせ",
        "pros": "最適化の余地大",
        "cons": "探索空間爆発",
        "papers": ["カスタム"]
    }
}
```

### 2. モデルサイズパラメータ

```python
MODEL_SIZE = {
    # レイヤー数
    "num_layers": {
        "range": [2, 4, 6, 8, 12, 16],
        "default": 6,
        "note": "小さいほど軽量、大きいほど高性能"
    },

    # 隠れ層次元
    "hidden_dim": {
        "range": [128, 256, 384, 512, 768, 1024],
        "default": 512,
        "note": "50-100MB目標なら 256-512 が現実的"
    },

    # Attention ヘッド数
    "num_heads": {
        "range": [2, 4, 6, 8, 12, 16],
        "default": 8,
        "constraint": "hidden_dim % num_heads == 0"
    },

    # FFN 中間次元
    "ffn_dim": {
        "formula": "hidden_dim * ffn_multiplier",
        "ffn_multiplier": [2, 3, 4],
        "default": 4,
        "note": "SwiGLU の場合は 2.5-3 が一般的"
    },

    # 語彙サイズ
    "vocab_size": {
        "options": [32000, 50000, 100000],
        "default": 50000,
        "note": "コード特化なら小さめでOK"
    }
}
```

### 3. 正規化手法

```python
NORMALIZATION = {
    "layernorm": {
        "formula": "(x - mean) / std",
        "pros": "標準的、安定",
        "cons": "やや遅い",
        "papers": ["Layer Normalization (2016)"]
    },

    "rmsnorm": {
        "formula": "x / rms(x)",
        "pros": "高速（mean計算不要）",
        "cons": "やや不安定",
        "papers": ["Root Mean Square Layer Normalization (2019)", "LLaMA (2023)"]
    },

    "groupnorm": {
        "description": "チャネルをグループ化",
        "pros": "バッチサイズ非依存",
        "cons": "実装複雑",
        "papers": ["Group Normalization (2018)"]
    }
}
```

### 4. 活性化関数

```python
ACTIVATION = {
    "gelu": {
        "formula": "x * Φ(x)",
        "pros": "BERT/GPT標準",
        "cons": "やや遅い",
        "papers": ["GELU (2016)"]
    },

    "silu": {
        "formula": "x * sigmoid(x)",
        "pros": "高速、性能良い",
        "cons": "なし",
        "papers": ["Sigmoid Linear Unit (2017)", "LLaMA uses SiLU"]
    },

    "geglu": {
        "formula": "GELU(x * W) ⊗ (x * V)",
        "pros": "高性能（GPT-3, PaLM）",
        "cons": "パラメータ増",
        "papers": ["GLU Variants Improve Transformer (2020)"]
    },

    "swiglu": {
        "formula": "Swish(x * W) ⊗ (x * V)",
        "pros": "最新のベストプラクティス",
        "cons": "パラメータ増",
        "papers": ["LLaMA (2023)", "PaLM (2022)"]
    }
}
```

### 5. 位置エンコーディング

```python
POSITION_ENCODING = {
    "absolute": {
        "description": "固定sin/cos（BERT/GPT）",
        "pros": "シンプル",
        "cons": "長文で性能劣化"
    },

    "rope": {
        "description": "Rotary Position Embedding",
        "pros": "長文対応、相対位置",
        "cons": "やや複雑",
        "papers": ["RoFormer (2021)", "LLaMA uses RoPE"]
    },

    "alibi": {
        "description": "Attention with Linear Biases",
        "pros": "超長文対応、パラメータ不要",
        "cons": "性能やや劣る",
        "papers": ["ALiBi (2021)", "BLOOM uses ALiBi"]
    }
}
```

### 6. 量子化・圧縮

```python
QUANTIZATION = {
    "fp16": {
        "size_reduction": "1x",
        "accuracy_loss": "0%",
        "note": "ベースライン"
    },

    "int8": {
        "size_reduction": "2x",
        "accuracy_loss": "< 1%",
        "method": "Post-Training Quantization",
        "papers": ["LLM.int8() (2022)"]
    },

    "int4": {
        "size_reduction": "4x",
        "accuracy_loss": "1-3%",
        "method": "GPTQ / AWQ",
        "papers": ["GPTQ (2023)", "AWQ (2023)"]
    }
}

PRUNING = {
    "magnitude": {
        "description": "重み絶対値が小さいニューロンを削除",
        "ratio": [0.2, 0.4, 0.6],
        "pros": "シンプル"
    },

    "structured": {
        "description": "レイヤー単位で削除",
        "ratio": [0.2, 0.4],
        "pros": "実装効率"
    },

    "distillation_aware": {
        "description": "蒸留時に同時プルーニング",
        "pros": "性能保持",
        "papers": ["Distillation-Aware Pruning (2021)"]
    }
}
```

---

## 探索戦略

### 遺伝的アルゴリズム

```python
class EvolutionaryNAS:
    """
    遺伝的アルゴリズムでアーキテクチャ探索
    """

    def __init__(self):
        self.population_size = 50
        self.num_generations = 100
        self.elite_ratio = 0.2
        self.mutation_rate = 0.3

    def fitness_function(self, architecture):
        """
        適応度関数（多目的最適化）

        目標:
        - 高精度（accuracy）
        - 小サイズ（model_size）
        - 低レイテンシ（latency）
        """
        # 評価指標
        accuracy = evaluate_accuracy(architecture)
        size_mb = get_model_size_mb(architecture)
        latency_ms = measure_latency(architecture)

        # 重み付きスコア
        # accuracy: 50%, size: 30%, latency: 20%
        fitness = (
            0.5 * accuracy / 100.0 +
            0.3 * (1.0 - size_mb / 500.0) +  # 500MB以下を期待
            0.2 * (1.0 - latency_ms / 100.0) # 100ms以下を期待
        )

        return fitness

    def crossover(self, parent1, parent2):
        """
        交叉（2つの親から子を生成）
        """
        child = {}
        for key in parent1.keys():
            # 50% の確率でどちらかの親から継承
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child

    def mutate(self, architecture):
        """
        突然変異
        """
        mutated = architecture.copy()
        for key in mutated.keys():
            if random.random() < self.mutation_rate:
                # ランダムに別の値に変更
                mutated[key] = random.choice(SEARCH_SPACE[key])
        return mutated
```

### 効率的評価（Early Stopping）

```python
def efficient_evaluate(architecture):
    """
    計算コスト削減のための効率的評価

    戦略:
    1. 小規模データで事前評価（1000サンプル）
    2. 明らかに性能が悪いものは早期打ち切り
    3. 有望なもののみフル評価
    """
    # Phase 1: 小規模評価（5分）
    small_accuracy = train_small(architecture, samples=1000, epochs=5)

    if small_accuracy < 0.3:  # 30%未満なら打ち切り
        return {
            "accuracy": small_accuracy,
            "size": get_size(architecture),
            "latency": estimate_latency(architecture),
            "early_stopped": True
        }

    # Phase 2: 中規模評価（20分）
    if small_accuracy > 0.5:
        medium_accuracy = train_medium(architecture, samples=10000, epochs=10)

        if medium_accuracy < 0.6:
            return {"accuracy": medium_accuracy, "early_stopped": True}

    # Phase 3: フル評価（60分）
    full_accuracy = train_full(architecture, samples=100000, epochs=20)

    return {
        "accuracy": full_accuracy,
        "size": get_size(architecture),
        "latency": measure_latency(architecture),
        "early_stopped": False
    }
```

---

## 実験設計

### ベースライン実験（Week 2）

**Dataset**: MNIST（簡単なタスクで動作確認）

```python
baseline_experiment = {
    "dataset": "MNIST",
    "num_architectures": 50,
    "num_generations": 10,
    "evaluation_time": "5分/arch",
    "total_time": "4時間",
    "goal": "システム動作確認"
}
```

### 小規模実験（Week 3-4）

**Dataset**: CodeSearchNet（コード検索）

```python
small_experiment = {
    "dataset": "CodeSearchNet (Python)",
    "num_architectures": 100,
    "num_generations": 20,
    "evaluation_time": "20分/arch",
    "total_time": "33時間（2日間）",
    "goal": "有望なアーキテクチャ候補を発見"
}
```

### 本格実験（Month 2-3）

**Dataset**: HumanEval + MBPP

```python
full_experiment = {
    "dataset": "HumanEval + MBPP",
    "num_architectures": 1000,
    "num_generations": 50,
    "evaluation_time": "60分/arch",
    "total_time": "1000時間 ≈ 42日",
    "parallel": "RTX 5090 + 4090 → 半分に短縮",
    "actual_time": "21日",
    "goal": "最良アーキテクチャ発見"
}
```

---

## 評価指標

### Primary Metrics

1. **Pass@1 (HumanEval)**
   - GPT-4: ~67%
   - 目標: >60% (90% of GPT-4)

2. **Model Size**
   - GPT-4: ~1.7TB
   - 目標: 50-100MB (1/17,000)

3. **Inference Latency**
   - GPT-4 API: ~500ms
   - 目標: <10ms (50x faster)

### Secondary Metrics

4. **FLOPs**
   - 少ないほど効率的

5. **Memory Footprint**
   - 推論時のメモリ使用量

6. **Training Time**
   - 短いほど実用的

---

## リスク管理

### 技術的リスク

| リスク | 軽減策 |
|--------|--------|
| 探索空間が広すぎて収束しない | 段階的に探索空間を拡大 |
| 評価に時間がかかりすぎる | Early stopping + 並列評価 |
| 良いアーキテクチャが見つからない | ベースラインを複数用意 |

### 計算リソースリスク

| リスク | 軽減策 |
|--------|--------|
| GPU故障 | GCPバックアップ |
| 電力コスト超過 | 夜間実行（電気代安い） |
| 時間超過 | 優先度付け |

---

## 次のステップ

### 今週
- [x] この設計文書作成
- [ ] `nas/search_space.py` 実装開始
- [ ] 論文サーベイ（20本）

### 来週
- [ ] ベースライン実験（MNIST）
- [ ] 評価システム構築
- [ ] 遺伝的アルゴリズム実装

---

**最終更新**: 2025-12-07
**作成者**: Koju（MIT CS PhD）
