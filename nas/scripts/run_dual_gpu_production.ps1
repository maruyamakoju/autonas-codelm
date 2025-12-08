# ============================================================
# Dual-GPU NAS Production Run (PowerShell)
#
# Full NAS search with comprehensive settings
# Expected runtime: 4-8 hours depending on architecture
# ============================================================

param(
    [string]$ExperimentName = "code_nas_prod_$(Get-Date -Format 'yyyyMMdd_HHmm')",
    [int]$Population = 50,
    [int]$Generations = 30,
    [int]$MaxTrainSteps = 1000,
    [int]$SeqLen = 256,
    [int]$BatchSize = 32,
    [string]$SearchMode = "full",
    [string]$GPUs = "cuda:0,cuda:1"
)

Write-Host "========================================"
Write-Host "NAS Production Run - Dual GPU"
Write-Host "========================================"
Write-Host ""

# GPU info
Write-Host "[INFO] GPU Configuration:"
python -c @"
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  cuda:{i} -> {props.name} ({props.total_memory / 1024**3:.1f} GB)')
"@
Write-Host ""

Write-Host "[CONFIG] Production Settings"
Write-Host "  Experiment: $ExperimentName"
Write-Host "  Population: $Population"
Write-Host "  Generations: $Generations"
Write-Host "  Max train steps: $MaxTrainSteps"
Write-Host "  Seq length: $SeqLen"
Write-Host "  Batch size: $BatchSize"
Write-Host "  Search mode: $SearchMode"
Write-Host "  GPUs: $GPUs"
Write-Host ""

# Estimate time
$archsTotal = $Population * $Generations
$avgTimePerArch = 60  # seconds estimate
$totalMinutes = [math]::Round($archsTotal * $avgTimePerArch / 60 / 2)  # divide by 2 for dual GPU
Write-Host "[ESTIMATE] ~$totalMinutes minutes with dual GPU"
Write-Host ""

$confirm = Read-Host "Start production run? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled."
    exit
}

Write-Host ""
Write-Host "[START] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

$startTime = Get-Date

python evolution.py `
  --experiment_name $ExperimentName `
  --population $Population `
  --generations $Generations `
  --use_real_training `
  --train_path "../data/code_char/train.txt" `
  --val_path "../data/code_char/val.txt" `
  --seq_len $SeqLen `
  --batch_size $BatchSize `
  --max_train_steps $MaxTrainSteps `
  --search_mode $SearchMode `
  --parallel `
  --gpus $GPUs

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "========================================"
Write-Host "Production Run Complete!"
Write-Host "========================================"
Write-Host "  Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
Write-Host "  Results: logs/$ExperimentName/"
Write-Host ""

# Show best architecture
$bestFile = "logs/$ExperimentName/evolution/best_architecture.json"
if (Test-Path $bestFile) {
    Write-Host "[BEST] Architecture:"
    Get-Content $bestFile | python -c @"
import sys, json
d = json.load(sys.stdin)
print(f"  Fitness: {d['fitness']:.4f}")
print(f"  Val Loss: {d['result']['raw_metrics']['val_loss']:.4f}")
print(f"  Size: {d['result']['raw_metrics']['model_size_mb']:.1f} MB")
print(f"  Latency: {d['result']['raw_metrics']['latency_ms']:.2f} ms")
"@
}

# Show worker stats
$statsFile = "logs/$ExperimentName/parallel/worker_stats.json"
if (Test-Path $statsFile) {
    Write-Host ""
    Write-Host "[STATS] GPU Utilization:"
    Get-Content $statsFile | python -c @"
import sys, json
d = json.load(sys.stdin)
total_tasks = sum(w['tasks_completed'] for w in d['workers'])
for w in d['workers']:
    pct = w['tasks_completed'] / total_tasks * 100 if total_tasks > 0 else 0
    print(f"  {w['device']}: {w['tasks_completed']} tasks ({pct:.1f}%), {w['total_eval_time_s']:.1f}s")
"@
}
