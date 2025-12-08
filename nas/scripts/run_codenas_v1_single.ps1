# CodeNAS v1 - Single GPU Production Run
# 本番用単一GPU実験スクリプト
#
# Usage:
#   .\run_codenas_v1_single.ps1
#   .\run_codenas_v1_single.ps1 -Population 32 -Generations 10
#   .\run_codenas_v1_single.ps1 -ExperimentName "my_experiment"

param(
    [string]$ExperimentName = "code_nas_v1_single",
    [int]$Population = 24,
    [int]$Generations = 8,
    [int]$MaxTrainSteps = 300,
    [int]$SeqLen = 256,
    [int]$BatchSize = 32,
    [string]$SearchMode = "medium",
    [string]$Device = "cuda:0"
)

# Move to nas directory
Set-Location "$PSScriptRoot/.."

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " CodeNAS v1 - Single GPU Production" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Experiment  : $ExperimentName" -ForegroundColor Yellow
Write-Host " Population  : $Population"
Write-Host " Generations : $Generations"
Write-Host " MaxSteps    : $MaxTrainSteps"
Write-Host " SeqLen      : $SeqLen"
Write-Host " BatchSize   : $BatchSize"
Write-Host " SearchMode  : $SearchMode"
Write-Host " Device      : $Device"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Estimate time
$tasksTotal = $Population * $Generations
$estTimeMin = [math]::Round($tasksTotal * 0.8, 0)  # ~0.8 min per task estimate
$estTimeMax = [math]::Round($tasksTotal * 1.5, 0)
Write-Host " Estimated tasks: $tasksTotal" -ForegroundColor Gray
Write-Host " Estimated time : $estTimeMin - $estTimeMax minutes" -ForegroundColor Gray
Write-Host ""

# Confirm start
$confirm = Read-Host "Start experiment? (y/n)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Aborted." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting CodeNAS v1..." -ForegroundColor Green
Write-Host ""

# Run evolution
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
    --device $Device `
    --search_mode $SearchMode

# Check result
$resultPath = "logs/$ExperimentName/evolution/best_architecture.json"
if (Test-Path $resultPath) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Best architecture saved to:" -ForegroundColor Yellow
    Write-Host "  $resultPath"
    Write-Host ""

    # Show fitness
    $result = Get-Content $resultPath | ConvertFrom-Json
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  Fitness   : $($result.fitness)"
    Write-Host "  Val Loss  : $([math]::Round($result.raw_metrics.val_loss, 4))"
    Write-Host "  Val PPL   : $([math]::Round($result.raw_metrics.val_ppl, 2))"
    Write-Host "  Params    : $([math]::Round($result.raw_metrics.num_params / 1e6, 2))M"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "WARNING: Result file not found!" -ForegroundColor Red
    Write-Host "Check logs for errors."
}
