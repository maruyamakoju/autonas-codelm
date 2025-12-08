# NAS Experiments Command Reference

よく使うコマンド集。未来の自分用。

## Quick Reference

| 目的 | コマンド |
|------|---------|
| スモークテスト | `python evolution.py --experiment_name smoke_test --population 4 --generations 2 ...` |
| ミディアム実験 | `python evolution.py --experiment_name medium_nas --population 10 --generations 5 ...` |
| 本番実験 | `.\scripts\run_dual_gpu_production.ps1` |
| Sanity Check | `bash scripts/sanity_check.sh` |
| 並列ログ解析 | `python analyze_parallel_stats.py --log_dir logs/exp/parallel` |
| GPU校正 | `python calibrate_gpus.py` |

---

## 1. Smoke Test (動作確認用)

```bash
cd nas

# 最小構成でサクッと動作確認 (~2-3分)
python evolution.py \
  --experiment_name "smoke_test" \
  --population 4 \
  --generations 2 \
  --use_real_training \
  --train_path "../data/code_char/train.txt" \
  --val_path "../data/code_char/val.txt" \
  --seq_len 64 \
  --batch_size 8 \
  --max_train_steps 50 \
  --device cuda:0 \
  --search_mode "minimal"
```

---

## 2. Medium Run (開発・デバッグ用)

```bash
# 中規模実験 (~30-40分)
python evolution.py \
  --experiment_name "medium_nas" \
  --population 10 \
  --generations 5 \
  --use_real_training \
  --train_path "../data/code_char/train.txt" \
  --val_path "../data/code_char/val.txt" \
  --seq_len 128 \
  --batch_size 16 \
  --max_train_steps 150 \
  --device cuda:0 \
  --search_mode "minimal"
```

---

## 3. Parallel Evaluation (Single GPU)

```bash
# 並列評価テスト（1GPU、マルチプロセス）
python evolution.py \
  --experiment_name "parallel_test" \
  --population 8 \
  --generations 3 \
  --use_real_training \
  --train_path "../data/code_char/train.txt" \
  --val_path "../data/code_char/val.txt" \
  --seq_len 128 \
  --batch_size 16 \
  --max_train_steps 100 \
  --search_mode "minimal" \
  --parallel \
  --gpus "cuda:0"
```

---

## 4. Dual-GPU Experiments (5090 + 4090)

### Quick Test (~10-15分)
```powershell
cd nas/scripts
.\run_dual_gpu_quick_test.ps1
```

### Standard Run
```powershell
.\run_dual_gpu.ps1
```

### Production Run (パラメータ指定可)
```powershell
.\run_dual_gpu_production.ps1 -Population 50 -Generations 30 -MaxTrainSteps 500
```

### コマンドライン直接実行
```bash
python evolution.py \
  --experiment_name "code_nas_dual_gpu_v1" \
  --population 40 \
  --generations 20 \
  --use_real_training \
  --train_path "../data/code_char/train.txt" \
  --val_path "../data/code_char/val.txt" \
  --seq_len 256 \
  --batch_size 32 \
  --max_train_steps 500 \
  --search_mode "medium" \
  --parallel \
  --gpus "cuda:0,cuda:1"
```

---

## 5. CodeNAS v1 (Production Single GPU)

**本番実験用設定。1GPUで本格的なアーキテクチャ探索を実行。**

### 設定値
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| population | 24 | 世代あたりの個体数 |
| generations | 8 | 進化世代数 |
| search_mode | medium | 探索空間（より多様なアーキテクチャ） |
| max_train_steps | 300 | 各アーキテクチャの訓練ステップ数 |
| seq_len | 256 | シーケンス長 |
| batch_size | 32 | バッチサイズ |
| device | cuda:0 | 使用GPU |

### PowerShellスクリプト実行
```powershell
cd nas\scripts
.\run_codenas_v1_single.ps1
```

### パラメータ指定実行
```powershell
.\run_codenas_v1_single.ps1 -Population 32 -Generations 10
```

### コマンドライン直接実行
```bash
python evolution.py \
  --experiment_name "code_nas_v1_single" \
  --population 24 \
  --generations 8 \
  --use_real_training \
  --train_path "../data/code_char/train.txt" \
  --val_path "../data/code_char/val.txt" \
  --seq_len 256 \
  --batch_size 32 \
  --max_train_steps 300 \
  --device "cuda:0" \
  --search_mode "medium"
```

### 期待される結果
- 実行時間: 1.5〜3時間
- 期待Fitness: 1.0
- 出力: `logs/code_nas_v1_single/evolution/best_architecture.json`

### 実測結果 (2024-12)

| 指標 | 値 |
|------|-----|
| Best Fitness | 1.0000 |
| Val Loss | 0.0188 |
| Val PPL | 1.02 |
| Accuracy | 98.14% |
| Params | 2.68M |
| Model Size | 3.06 MB |
| Latency | 3.0 ms |
| Evaluated | 144 archs (6 gen) |
| Runtime | ~15-20 min |

**Best Architecture:**
- `models/codenas_v1_best_transformer.json`

```
arch_type: transformer
num_layers: 4
hidden_dim: 256
num_heads: 8
ffn_multiplier: 3.0
normalization: rmsnorm
activation: gelu
position_encoding: rope
```

> 短いコードコーパス（1.3MB accelerate）+ 300 steps + 小モデル優位のため予想より短時間で完了。
> Gen 0で既にfitness=1.0達成、以降も同等の精度を維持。

---

## 5.5 CodeNAS v2 (Two-stage NAS)

**Multi-fidelity NAS: Stage 1でスクリーニング → Stage 2でTop-k精密評価**

### 設定値
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| population | 24 | 世代あたりの個体数 |
| generations | 8 | 進化世代数 |
| search_mode | medium | 探索空間 |
| two_stage | true | 2段階評価有効 |
| stage1_steps | 50 | スクリーニング（少ステップ） |
| stage2_steps | 300 | 精密評価（多ステップ） |
| top_k | 6 | Stage 2に進む候補数 |
| seq_len | 256 | シーケンス長 |
| batch_size | 32 | バッチサイズ |
| device | cuda:0 | 使用GPU |

### PowerShell実行（推奨）
```powershell
cd nas\scripts
.\run_codenas_v2_two_stage.ps1

# カスタムパラメータ指定
.\run_codenas_v2_two_stage.ps1 -Population 32 -Generations 10 -TopK 8
```

### コマンドライン実行（直接）
```bash
python evolution.py \
  --experiment_name "code_nas_v2_two_stage" \
  --population 24 \
  --generations 8 \
  --use_real_training \
  --train_path "../data/code_char/train.txt" \
  --val_path "../data/code_char/val.txt" \
  --seq_len 256 \
  --batch_size 32 \
  --device "cuda:0" \
  --search_mode "medium" \
  --two_stage \
  --stage1_steps 50 \
  --stage2_steps 300 \
  --top_k 6
```

### 期待される効果
- Stage 1 (50 steps): 24候補を高速スクリーニング → 約1分/アーキ
- Stage 2 (300 steps): Top 6のみ精密評価 → 約3-5分/アーキ
- v1比: 評価時間削減 (24×300 → 24×50 + 6×300 = 3000 steps vs 7200 steps = 58%削減)

### 比較ツール
```bash
# v1 vs v2を比較（引数なしでデフォルト比較）
python compare_experiments.py

# または明示的に指定
python compare_experiments.py logs/code_nas_v1_single logs/code_nas_v2_two_stage
```

### 実測結果 (2024-12)

| メトリック | 値 | 備考 |
|-----------|-----|------|
| **Best Fitness** | 1.0000 | ✅ 完璧なスコア |
| **Architecture** | Transformer L4 H384 | Heads=4, FFN×2.0, SiLU, RoPE |
| **Parameters** | 4.80M | v1 (2.68M) より大きい |
| **Model Size** | 7.32 MB | v1 (3.06 MB) より大きい |
| **Val Loss** | 0.0142 | v1 (0.0188) より**良い** |
| **Val PPL** | 1.01 | v1 (1.02) より良い |
| **Accuracy** | 98.59% | v1 (98.14%) より0.45%高い |
| **Latency** | 3.53 ms | v1 (3.01 ms) より遅い |
| **Train Time** | 3.52 s | v1 (5.03 s) より速い |
| **Generations** | 8 | Population=24, Stage1=50, Stage2=300 |

#### v1 vs v2 比較

```bash
python compare_experiments.py
```

**結論**:
- v2は**精度でv1を上回る**（Val Loss 0.0142 vs 0.0188）
- しかし**サイズとレイテンシではv1が優れる**（3.06MB vs 7.32MB）
- Two-stage NASは精度重視の探索に有効
- **実用的には軽量性を重視してv1を推奨**

---

## 6. Sanity Check (並列 vs 非並列)

> **結果**: PASSED (2024-12) - Sequential: 1.0, Parallel: 1.0

```bash
cd nas
bash scripts/sanity_check.sh
```

結果確認:
```bash
# Sequential結果
cat logs/sanity_seq/evolution/best_architecture.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Fitness: {d[\"fitness\"]:.4f}')"

# Parallel結果
cat logs/sanity_par/evolution/best_architecture.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Fitness: {d[\"fitness\"]:.4f}')"
```

---

## 7. Analysis Tools

### 並列ログ解析
```bash
# 基本解析
python analyze_parallel_stats.py --log_dir logs/medium_nas/parallel

# 2つの実験を比較
python analyze_parallel_stats.py \
  --log_dir logs/single_gpu_exp/parallel \
  --compare logs/dual_gpu_exp/parallel
```

### GPU校正（実測ベンチマーク）
```bash
python calibrate_gpus.py --num_runs 3 --output logs/gpu_calibration.json
```

### 結果確認
```bash
# Best architecture
cat logs/medium_nas/evolution/best_architecture.json | python -m json.tool

# Fitness history
cat logs/medium_nas/evolution/fitness_history.json | python -c "
import sys, json
data = json.load(sys.stdin)
for h in data:
    print(f\"Gen {h['generation']}: best={h['best_fitness']:.4f}, mean={h['mean_fitness']:.4f}\")
"
```

---

## 8. GPU Status Check

```bash
# GPU状態確認
nvidia-smi

# CUDA利用可能確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 詳細GPU情報
python -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)')
"
```

---

## 9. Directory Structure

```
nas/
├── evolution.py          # メインNASエンジン
├── parallel_evaluator.py # マルチGPU評価器
├── evaluator.py          # アーキテクチャ評価
├── fitness.py            # 適応度関数
├── search_space.py       # 探索空間
├── train_loop.py         # 訓練ループ
├── analyze_parallel_stats.py  # ログ解析
├── calibrate_gpus.py     # GPU校正
├── scripts/
│   ├── sanity_check.sh
│   ├── run_dual_gpu.ps1
│   ├── run_dual_gpu_quick_test.ps1
│   └── run_dual_gpu_production.ps1
└── logs/
    └── <experiment_name>/
        ├── evolution/
        │   ├── best_architecture.json
        │   ├── fitness_history.json
        │   ├── fitness_history.png
        │   └── checkpoint_gen*.json
        └── parallel/
            ├── parallel_worker_stats.json
            ├── parallel_batch_stats.json
            └── worker_*_cuda_*/
```

---

## 10. Troubleshooting

### OOM (Out of Memory)
```bash
# バッチサイズを下げる
--batch_size 8

# モデルサイズを制限（search_mode=minimalで小さいモデルのみ）
--search_mode "minimal"
```

### CUDA Error
```bash
# GPUリセット
nvidia-smi --gpu-reset

# プロセス確認・終了
nvidia-smi
kill -9 <PID>
```

### Parallel Evaluation Timeout
```python
# evolution.pyで調整
max_eval_time_s=7200.0  # 2 hours
```

---

## 11. Expected Results

| 設定 | 期待Fitness | 期待時間 |
|------|-------------|----------|
| Smoke test | 0.5-0.9 | 2-3分 |
| Medium run | 0.9-1.0 | 30-40分 |
| Production (1GPU) | 1.0 | 2-4時間 |
| Production (2GPU) | 1.0 | 1-2時間 |

---

*Last updated: 2024-12*
