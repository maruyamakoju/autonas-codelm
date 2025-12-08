#!/bin/bash
# Sanity Check: Compare parallel vs sequential evaluation
# This script runs both modes with the same settings and compares results

echo "========================================"
echo "NAS Parallel vs Sequential Sanity Check"
echo "========================================"
echo ""

# Settings
POPULATION=6
GENERATIONS=3
MAX_STEPS=200
TRAIN_PATH="../data/code_char/train.txt"
VAL_PATH="../data/code_char/val.txt"

echo "[1/2] Running SEQUENTIAL evaluation..."
python evolution.py \
  --experiment_name "sanity_seq" \
  --population $POPULATION \
  --generations $GENERATIONS \
  --use_real_training \
  --train_path "$TRAIN_PATH" \
  --val_path "$VAL_PATH" \
  --max_train_steps $MAX_STEPS \
  --search_mode "minimal"

echo ""
echo "[2/2] Running PARALLEL evaluation (single GPU)..."
python evolution.py \
  --experiment_name "sanity_par" \
  --population $POPULATION \
  --generations $GENERATIONS \
  --use_real_training \
  --train_path "$TRAIN_PATH" \
  --val_path "$VAL_PATH" \
  --max_train_steps $MAX_STEPS \
  --search_mode "minimal" \
  --parallel \
  --gpus "cuda:0"

echo ""
echo "========================================"
echo "COMPARISON"
echo "========================================"
echo ""
echo "[Sequential Results]"
cat logs/sanity_seq/evolution/best_architecture.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Fitness: {d[\"fitness\"]:.4f}')"

echo ""
echo "[Parallel Results]"
cat logs/sanity_par/evolution/best_architecture.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Fitness: {d[\"fitness\"]:.4f}')"

echo ""
echo "Done! Check logs/sanity_seq and logs/sanity_par for full results."
