#!/bin/bash
# ============================================================
# Dual-GPU NAS Experiment Template
#
# For: RTX 5090 (cuda:0) + RTX 4090 (cuda:1)
# ============================================================

echo "========================================"
echo "NAS Dual-GPU Experiment"
echo "========================================"
echo ""

# Check available GPUs
echo "[INFO] Available GPUs:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device count: {torch.cuda.device_count()}'); [print(f'  [{i}] {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# ============================================================
# Configuration - Adjust as needed
# ============================================================
EXPERIMENT_NAME="code_nas_dual_gpu_v1"
POPULATION=40
GENERATIONS=20
MAX_TRAIN_STEPS=500
SEQ_LEN=256
BATCH_SIZE=32
SEARCH_MODE="medium"

# Data paths
TRAIN_PATH="../data/code_char/train.txt"
VAL_PATH="../data/code_char/val.txt"

# GPU configuration (5090 + 4090)
GPUS="cuda:0,cuda:1"

echo "[CONFIG]"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Population: $POPULATION"
echo "  Generations: $GENERATIONS"
echo "  Max steps: $MAX_TRAIN_STEPS"
echo "  Seq len: $SEQ_LEN"
echo "  Batch size: $BATCH_SIZE"
echo "  Search mode: $SEARCH_MODE"
echo "  GPUs: $GPUS"
echo ""

# ============================================================
# Run the experiment
# ============================================================
echo "[START] Running NAS with dual GPUs..."
echo ""

python evolution.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --population $POPULATION \
  --generations $GENERATIONS \
  --use_real_training \
  --train_path "$TRAIN_PATH" \
  --val_path "$VAL_PATH" \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --max_train_steps $MAX_TRAIN_STEPS \
  --search_mode "$SEARCH_MODE" \
  --parallel \
  --gpus "$GPUS"

echo ""
echo "========================================"
echo "Experiment Complete!"
echo "========================================"
echo "Results: logs/$EXPERIMENT_NAME/"
echo ""
