param(
    [int]$TotalSteps = 2000000,
    [int]$Seed = 20260730,
    [string]$InitialCheckpoint = "artifacts/strong_policy_upgrade/dagger/iteration_2/training/best_behavior_cloning.pt"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $Python (Join-Path $PSScriptRoot "train_riskshield_hrmppo_v2.py") `
    --initial-checkpoint $InitialCheckpoint `
    --total-steps $TotalSteps `
    --seed $Seed `
    --device cuda
exit $LASTEXITCODE
