param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Webots = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
$PolicyConfig = Join-Path $RepoRoot "configs\webots\policy_control.json"
$PerceptionConfig = Join-Path $RepoRoot "configs\perception\stage5b_live_perception.json"
$Stage5CRunner = Join-Path $RepoRoot "scripts\run_stage5c_rl_motor_runtime.ps1"
$FinalSummary = Join-Path $RepoRoot "reports\final_project_completion\final_runtime_summary.json"

foreach ($Required in @(
    $Python,
    $Webots,
    $PolicyConfig,
    $PerceptionConfig,
    $Stage5CRunner
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Production prerequisite is missing: $Required"
    }
}

Set-Location -LiteralPath $RepoRoot
$Policy = Get-Content -LiteralPath $PolicyConfig -Raw | ConvertFrom-Json
$Perception = Get-Content -LiteralPath $PerceptionConfig -Raw | ConvertFrom-Json
$Checkpoint = Join-Path `
    $RepoRoot `
    "artifacts\runs\maskable_ppo_deadlock_safe_shield_seed42_u100\evaluations\best_model\best_model.zip"
$CvCheckpoint = Join-Path $RepoRoot $Perception.model.relative_path

foreach ($Model in @(
    @{
        Path = $Checkpoint
        Hash = [string]$Policy.checkpoint_sha256
        Name = "RL"
    },
    @{
        Path = $CvCheckpoint
        Hash = [string]$Perception.model.sha256
        Name = "CV"
    }
)) {
    if (-not (Test-Path -LiteralPath $Model.Path -PathType Leaf)) {
        throw "$($Model.Name) checkpoint is missing: $($Model.Path)"
    }
    $ActualHash = (Get-FileHash -LiteralPath $Model.Path -Algorithm SHA256).Hash
    if ($ActualHash -ne $Model.Hash.ToUpperInvariant()) {
        throw "$($Model.Name) checkpoint SHA-256 mismatch."
    }
}

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable.'; print('CUDA_DEVICE=' + torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA validation failed."
}

& $Webots --version
if ($LASTEXITCODE -ne 0) {
    throw "Webots validation failed."
}

& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $Stage5CRunner `
    -RepoRoot $RepoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Complete autonomous inspection failed."
}

$RuntimeSummary = Join-Path `
    $RepoRoot `
    "webots\logs\stage5c_rl_motor_runtime\stage5c_runtime_summary.json"
if (-not (Test-Path -LiteralPath $RuntimeSummary -PathType Leaf)) {
    throw "Production runtime summary is missing."
}
Copy-Item -LiteralPath $RuntimeSummary -Destination $FinalSummary -Force
Write-Host "COMPLETE_AUTONOMOUS_INSPECTION=PASSED"
Write-Host "Final summary: $FinalSummary"
