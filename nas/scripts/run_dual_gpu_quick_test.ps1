# ============================================================
# Dual-GPU NAS Quick Test (PowerShell)
#
# Short run to verify dual-GPU setup works correctly
# Expected runtime: ~10-15 minutes
# ============================================================

Write-Host "========================================"
Write-Host "NAS Dual-GPU Quick Test"
Write-Host "========================================"
Write-Host ""

# Check available GPUs
Write-Host "[INFO] Checking GPU setup..."
python -c @"
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {props.name} - {props.total_memory / 1024**3:.1f} GB')
"@
Write-Host ""

# Quick test configuration
$EXPERIMENT_NAME = "dual_gpu_quick_test"
$POPULATION = 8
$GENERATIONS = 3
$MAX_TRAIN_STEPS = 100
$SEQ_LEN = 128
$BATCH_SIZE = 16
$SEARCH_MODE = "minimal"
$GPUS = "cuda:0,cuda:1"

Write-Host "[CONFIG] Quick Test Settings"
Write-Host "  Population: $POPULATION"
Write-Host "  Generations: $GENERATIONS"
Write-Host "  Max steps: $MAX_TRAIN_STEPS"
Write-Host ""

Write-Host "[START] Running quick test..."
Write-Host ""

python evolution.py `
  --experiment_name $EXPERIMENT_NAME `
  --population $POPULATION `
  --generations $GENERATIONS `
  --use_real_training `
  --train_path "../data/code_char/train.txt" `
  --val_path "../data/code_char/val.txt" `
  --seq_len $SEQ_LEN `
  --batch_size $BATCH_SIZE `
  --max_train_steps $MAX_TRAIN_STEPS `
  --search_mode $SEARCH_MODE `
  --parallel `
  --gpus $GPUS

Write-Host ""
Write-Host "[DONE] Check logs/$EXPERIMENT_NAME/ for results"
Write-Host ""

# Show worker stats if available
$statsFile = "logs/$EXPERIMENT_NAME/parallel/worker_stats.json"
if (Test-Path $statsFile) {
    Write-Host "[STATS] Worker Statistics:"
    Get-Content $statsFile | python -c "import sys,json; d=json.load(sys.stdin); [print(f'  Worker {w[\"worker_id\"]} ({w[\"device\"]}): {w[\"tasks_completed\"]} tasks, {w[\"total_eval_time_s\"]:.1f}s') for w in d['workers']]"
}
