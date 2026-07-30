param(
    [string]$Checkpoint = "artifacts/strong_policy_upgrade/hrmppo_v2/final_checkpoint.pt",
    [int]$Seeds = 30
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
& $Python (Join-Path $PSScriptRoot "evaluate_riskshield_hrmppo_v2.py") `
    --checkpoint $Checkpoint `
    --seeds $Seeds `
    --device cuda
exit $LASTEXITCODE
