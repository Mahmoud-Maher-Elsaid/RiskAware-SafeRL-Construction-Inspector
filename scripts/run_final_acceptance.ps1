param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python is missing: $Python"
}
Set-Location -LiteralPath $RepoRoot

Write-Host "ACCEPTANCE_REPOSITORY=RUNNING"
$TrackedNoise = git ls-files | Select-String -Pattern '(^|/)(__pycache__|\.pytest_cache|\.ruff_cache)(/|$)|\.bak$|\.pyc$'
if ($TrackedNoise) { throw "Generated cache or backup content is tracked." }
$BackupFiles = Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -ErrorAction Stop |
    Where-Object {
        $_.FullName -notlike "$RepoRoot\.git\*" -and
        $_.FullName -notlike "$RepoRoot\.venv\*" -and
        $_.FullName -notlike "$RepoRoot\.python311\*" -and
        $_.Name -match '\.(bak|orig)$'
    }
if ($BackupFiles) { throw "Backup files remain in the repository." }
Write-Host "ACCEPTANCE_REPOSITORY=PASSED"

Write-Host "ACCEPTANCE_SOFTWARE=RUNNING"
& $Python -m compileall -q src scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\validate_grid_environment.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\run_full_research_benchmark.py --validate-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "ACCEPTANCE_SOFTWARE=PASSED"

Write-Host "ACCEPTANCE_EVIDENCE=RUNNING"
& $Python -c @'
import json
from pathlib import Path
import pandas as pd
root = Path("reports/final_submission")
assert json.load((root / "stage1_grid_environment/validation.json").open())["status"] == "PASSED"
assert len(pd.read_csv(root / "stage2_planner_baselines/raw_results.csv")) == 90
assert json.load((root / "stage3_rl_baselines/summary.json").open())["status"] == "PASSED"
assert json.load((root / "stage4_riskshield_ppo/comparison_summary.json").open())["status"] == "PASSED"
assert json.load((root / "stage5_safety_shield/summary.json").open())["status"] == "PASSED"
assert json.load((root / "stage6_webots/world_smoke_summary.json").open())["status"] == "PASSED"
assert json.load((root / "stage7_perception/validation.json").open())["status"] == "PASSED"
assert json.load((root / "stage8_uncertainty/summary.json").open())["run_count"] == 320
benchmark = json.load((root / "stage9_benchmark/benchmark_summary.json").open())
assert benchmark["run_count"] == benchmark["unique_run_ids"] == 1350
assert benchmark["missing_run_count"] == benchmark["failed_run_count"] == 0
ablation = json.load((root / "stage9_benchmark/ablation_summary.json").open())
assert ablation["run_count"] == ablation["unique_run_count"] == 180
assert json.load((root / "paper_result.json").open())["status"] == "PASSED"
print("FINAL_EVIDENCE_SCHEMA=PASSED")
'@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "ACCEPTANCE_EVIDENCE=PASSED"

Write-Host "ACCEPTANCE_BENCHMARK=RUNNING"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\run_full_research_benchmark.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "ACCEPTANCE_BENCHMARK=PASSED"

Write-Host "ACCEPTANCE_PRODUCTION=RUNNING"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\run_complete_autonomous_inspection.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$RuntimePath = Join-Path $RepoRoot "reports\final_project_completion\final_runtime_summary.json"
$Runtime = Get-Content -LiteralPath $RuntimePath -Raw | ConvertFrom-Json
foreach ($Flag in @(
    "policy_loaded",
    "policy_controls_motors",
    "safety_shield_active",
    "cv_model_connected",
    "cuda_inference_verified",
    "perception_live_during_mission",
    "perception_affects_runtime_state"
)) {
    if (-not [bool]$Runtime.$Flag) { throw "Required runtime flag is false: $Flag" }
}
if ([bool]$Runtime.manual_control_used) { throw "Manual control was used." }
if ([bool]$Runtime.fallback_controller_used) { throw "Fallback controller was used." }
Write-Host "ACCEPTANCE_PRODUCTION=PASSED"
Write-Host "FINAL_ACCEPTANCE=PASSED"
