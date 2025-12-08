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

## 5. Sanity Check (並列 vs 非並列)

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

## 6. Analysis Tools

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

## 7. GPU Status Check

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

## 8. Directory Structure

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

## 9. Troubleshooting

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

## 10. Expected Results

| 設定 | 期待Fitness | 期待時間 |
|------|-------------|----------|
| Smoke test | 0.5-0.9 | 2-3分 |
| Medium run | 0.9-1.0 | 30-40分 |
| Production (1GPU) | 1.0 | 2-4時間 |
| Production (2GPU) | 1.0 | 1-2時間 |

---

*Last updated: 2024-12*
