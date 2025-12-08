# CodeNAS v2 - Two-stage NAS Production Run
# Usage:
#   .\run_codenas_v2_two_stage.ps1
#   .\run_codenas_v2_two_stage.ps1 -Population 32 -Generations 10 -TopK 8

param(
    [string]$ExperimentName = "code_nas_v2_two_stage",
    [int]$Population = 24,
    [int]$Generations = 8,
    [int]$Stage1Steps = 50,
    [int]$Stage2Steps = 300,
    [int]$TopK = 6
)

$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host " CodeNAS v2 - Two-stage NAS"
Write-Host "========================================"
Write-Host " Experiment : $ExperimentName"
Write-Host " Population : $Population"
Write-Host " Generations: $Generations"
Write-Host " Stage1     : $Stage1Steps steps"
Write-Host " Stage2     : $Stage2Steps steps (Top $TopK)"
Write-Host "========================================"

cd ..  # nas/scripts → nas

python evolution.py `
  --experiment_name $ExperimentName `
  --population $Population `
  --generations $Generations `
  --use_real_training `
  --train_path "../data/code_char/train.txt" `
  --val_path "../data/code_char/val.txt" `
  --seq_len 256 `
  --batch_size 32 `
  --device "cuda:0" `
  --search_mode "medium" `
  --two_stage `
  --stage1_steps $Stage1Steps `
  --stage2_steps $Stage2Steps `
  --top_k $TopK
