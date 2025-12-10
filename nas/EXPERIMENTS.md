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

### アーキテクチャ可視化

```bash
# Current best model (v1)
python visualize_architecture.py models/codenas_best_current.json

# v1 original
python visualize_architecture.py logs/code_nas_v1_single/evolution/best_architecture.json

# v2 two-stage
python visualize_architecture.py logs/code_nas_v2_two_stage/evolution/best_architecture.json
```

---

## 5.6 v1 Production Training (Full Training)

**本命アーキテクチャ（v1）の本格訓練**

### 設定値
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| architecture | v1 single-stage | L4 H256 Heads=8 FFN×3.0 |
| max_steps | 10,000 | 本番用長時間訓練 |
| learning_rate | 3e-4 → 1e-5 | Cosine decay with warmup |
| warmup_steps | 500 | LR warmup |
| batch_size | 32 | |
| seq_len | 256 | |
| device | cuda:0 | RTX 5090 |

### コマンド実行
```bash
python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_production \
  --max_steps 10000 \
  --log_dir logs/train_v1_production
```

### 実測結果 (2024-12)

| メトリック | 値 | 備考 |
|-----------|-----|------|
| **Final Val Loss** | 0.0065 | Best: 0.0065 |
| **Final Val PPL** | 1.01 | 非常に低い |
| **Parameters** | 2.68M | |
| **Model Size** | 5.10 MB | 推定値 |
| **Latency** | 2.97 ms | RTX 5090 |
| **Training Time** | 9.6 min | 10,000 steps |
| **Steps/sec** | 18.5 (avg) | 初期56→後半18 |
| **Checkpoint** | v1_production_best.pt | |

### 学習曲線
- Step 100: Loss 3.56 → PPL 35.11
- Step 500: Loss 0.0175 → PPL 1.02 (warmup完了)
- Step 1000: Loss 0.0098 → PPL 1.01
- Step 5000: Loss 0.0069 → PPL 1.01
- Step 10000: Loss 0.0053 → PPL 1.01 ✅

### Playgroundテスト
```bash
python eval_playground.py
# デフォルトで訓練済みモデル (v1_production_best.pt) をロード
```

**生成品質**: 限定的（単純なパターン生成）
- 理由: 訓練データが小さい（1.3MB Python code）
- 改善策: より大きなデータセット、より長い訓練

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

## 12. BigData Training (データ拡張実験)

**目的**: データサイズを 1.3MB → 100MB+ に増やして、モード崩壊問題を解決する。

### Step 1: 大規模Pythonコーパスを準備

#### オプションA: The Stack dataset (推奨)

```bash
# HuggingFace The Stackから Pythonサブセットをダウンロード
# https://huggingface.co/datasets/bigcode/the-stack

# 例: 100MBサンプルをダウンロード (手動またはHF datasetsライブラリ)
# ダウンロード先: data/raw_python/
```

#### オプションB: CodeSearchNet dataset

```bash
# https://github.com/github/CodeSearchNet
# Python部分のみ抽出: data/raw_python/
```

#### オプションC: GitHub人気リポジトリ

```bash
# 例: requests, flask, django, numpy, pandas, scikit-learn
cd data/raw_python
git clone https://github.com/psf/requests.git
git clone https://github.com/pallets/flask.git
git clone https://github.com/django/django.git
# など...
```

### Step 2: コーパスを char-level テキストに変換

```bash
cd nas

# 生データから train/val を生成
python scripts/prepare_python_corpus.py \
  --src_dir ../data/raw_python \
  --train_out ../data/code_char_big/train.txt \
  --val_out ../data/code_char_big/val.txt \
  --val_ratio 0.01 \
  --min_file_size 100 \
  --max_file_size 262144

# 出力例:
# [COLLECT] OK Collected 12,345 valid Python files
# [WRITE] OK Train corpus: 12,222 files, 1,234,567 lines, 123.45 MB
# [WRITE] OK Val corpus: 123 files, 12,345 lines, 1.23 MB
```

**期待されるデータサイズ**:
- 最低: 50MB (小規模改善期待)
- 推奨: 100-500MB (大幅改善期待)
- 理想: 1GB+ (GPT-2レベルの品質期待)

### Step 3: v1アーキテクチャで再学習

```bash
cd nas

# 100K steps で学習 (現状の10倍)
python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_bigdata_char \
  --train_path ../data/code_char_big/train.txt \
  --val_path ../data/code_char_big/val.txt \
  --max_steps 100000 \
  --log_dir logs/train_v1_bigdata_char \
  --device cuda:0

# ログ:
# - Checkpoint: logs/train_v1_bigdata_char/v1_bigdata_char_best.pt
# - TensorBoard: logs/train_v1_bigdata_char/events.out.tfevents.*
```

**学習時間 (概算)**:
- 50MB データ: ~30-60分 (100K steps)
- 100MB データ: ~60-120分
- 500MB データ: ~3-6時間
- 1GB データ: ~6-12時間

### Step 4: 評価パイプラインで再検証

```bash
cd nas

# バッチ評価
python eval_playground.py \
  --checkpoint logs/train_v1_bigdata_char/v1_bigdata_char_best.pt \
  --eval_file eval/prompts/simple_python.txt \
  --output eval/results_bigdata.jsonl

# 解析
python eval/inspect_results.py eval/results_bigdata.jsonl --show_quality_examples

# 期待される改善:
# - Mode collapse率: 85.2% → <30%
# - Python keywords出現率: 0% → >50%
# - 平均repetition比率: 92.22% → <30%
```

### Step 5: 結果をドキュメントに反映

```bash
# EVALUATION_SUMMARY.md に BigData 版の結果を追記
# README.md の Phase 3 に進捗を更新
```

---

## 13. 次のステップ (BigData 実験後)

### BigData 実験が成功した場合
→ さらにデータを増やして品質を向上
→ Token-level modeling に移行して効率化

### BigData 実験でも改善が不十分な場合
→ Token-level modeling (BPE/SentencePiece) に切り替え
→ Knowledge distillation (GPT-4 → student model)

---

## 14. BigData Training (データ拡張実験) 🔥 RECOMMENDED

**Status**: Token-level infrastructure complete, ready for large-scale data
**Goal**: Mode collapse解決 (85% → <10%) via 100MB-1GB Python corpus
**Hardware**: RTX 5090 (heavy training OK)

### Step 1: コーパス準備 (Char + Token 両対応)

```bash
cd nas

# Option A: 既存データを再利用 (テスト用、7.92MB)
python scripts/prepare_python_corpus.py \
  --src_dir ../data/raw_python \
  --char_train ../data/code_char_big/train.txt \
  --char_val   ../data/code_char_big/val.txt \
  --token_train ../data/code_token_big/train.txt \
  --token_val   ../data/code_token_big/val.txt \
  --mode both \
  --val_ratio 0.01

# Option B: 大規模データ収集 (本番用、100MB-1GB)
# 1. データソースをダウンロード
cd ../data/raw_python

# The Stack dataset (推奨、高品質Python corpus)
# https://huggingface.co/datasets/bigcode/the-stack
# または GitHub repos を直接clone:
git clone --depth 1 https://github.com/psf/requests.git
git clone --depth 1 https://github.com/pallets/flask.git
git clone --depth 1 https://github.com/django/django.git
git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git
git clone --depth 1 https://github.com/pandas-dev/pandas.git
git clone --depth 1 https://github.com/numpy/numpy.git
# ... (15-20 repos で 100MB 達成可能)

# 2. コーパス生成 (target_size_mb で容量制限)
cd ../../nas
python scripts/prepare_python_corpus.py \
  --src_dir ../data/raw_python \
  --char_train ../data/code_char_bigdata/train.txt \
  --char_val   ../data/code_char_bigdata/val.txt \
  --token_train ../data/code_token_bigdata/train.txt \
  --token_val   ../data/code_token_bigdata/val.txt \
  --mode both \
  --val_ratio 0.01 \
  --target_size_mb 500 \
  --max_file_size 524288
```

**期待される出力**:
```
CHAR-LEVEL:
  Total:  500 MB, 15M lines
  Train:  495 MB (10,000 files)
  Val:    5 MB (100 files)

TOKEN-LEVEL:
  Total:       500 MB, 250M tokens
  Compression: 2.0x (chars/tokens)
  Vocab size:  50,257 (gpt2)
```

### Step 2: Token-level BigData 実験 (推奨) ⭐⭐⭐⭐⭐

```bash
cd nas

# スモークテスト (10K steps, ~1時間)
python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_token_bigdata_smoke \
  --train_path ../data/code_token_bigdata/train.txt \
  --val_path   ../data/code_token_bigdata/val.txt \
  --max_steps 10000 \
  --warmup_steps 500 \
  --use_tokens \
  --log_dir logs/train_v1_token_bigdata_smoke \
  --device cuda:0

# 本番訓練 (100K steps, ~10-20時間、RTX 5090で高速)
python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_token_bigdata_production \
  --train_path ../data/code_token_bigdata/train.txt \
  --val_path   ../data/code_token_bigdata/val.txt \
  --max_steps 100000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --use_tokens \
  --log_dir logs/train_v1_token_bigdata_production \
  --device cuda:0

# 評価
python eval_playground.py \
  --checkpoint logs/train_v1_token_bigdata_production/v1_token_bigdata_production_best.pt \
  --eval_file eval/prompts/simple_python.txt \
  --output eval/results_token_bigdata.jsonl \
  --mode token

python eval/inspect_results.py eval/results_token_bigdata.jsonl --show_quality_examples
```

**期待される改善**:
- Mode collapse: 100% → <10%
- Python keywords: 0% → >90%
- Val PPL: 1.036 → <1.01
- 生成品質: Repetitive → Coherent code

### Step 3: Char-level BigData 実験 (参考、非推奨)

```bash
# Char-level は 100MB でも不十分な可能性が高い
# Token-level 推奨

cd nas

python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_char_bigdata \
  --train_path ../data/code_char_bigdata/train.txt \
  --val_path   ../data/code_char_bigdata/val.txt \
  --max_steps 100000 \
  --log_dir logs/train_v1_char_bigdata \
  --device cuda:0
```

### Step 4: 比較分析

```bash
cd nas

# Token-level vs Char-level 比較
python compare_training_runs.py \
  logs/train_v1_token_bigdata_production \
  logs/train_v1_char_bigdata
```

### 実験パラメータ推奨値

| Dataset Size | Char-level Steps | Token-level Steps | Training Time (RTX 5090) |
|--------------|------------------|-------------------|--------------------------|
| 7.92 MB | 5,000 | 5,000 | ~10 min |
| 50 MB | 50,000 | 20,000 | ~2-4 hours |
| 100 MB | 100,000 | 30,000 | ~4-8 hours |
| 500 MB | 500,000 | 100,000 | ~20-40 hours |
| 1 GB | 1,000,000 | 200,000 | ~40-80 hours |

**Note**: Token-level は Char-level の 1/3-1/5 の steps で同等品質に到達

### トラブルシューティング

**Q: データダウンロードが遅い**
A: GitHub repos の shallow clone を使う (`--depth 1`)

**Q: メモリ不足**
A: `--max_file_size` を下げる、または `--target_size_mb` を小さくする

**Q: Mode collapse が解決しない**
A: さらにデータ量を増やす (500MB → 1GB)、または Knowledge Distillation (Option C) を試す

---

*Last updated: 2025-12-09*
