param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "scripts\run_full_research_benchmark.py"
$Ablations = Join-Path $RepoRoot "scripts\run_stage9_ablations.py"
$Figures = Join-Path $RepoRoot "scripts\generate_final_figures.py"
$Config = Join-Path $RepoRoot "configs\benchmarks\full_matrix.yaml"
foreach ($Required in @($Python, $Script, $Ablations, $Figures, $Config)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required benchmark file is missing: $Required"
    }
}
Set-Location -LiteralPath $RepoRoot
& $Python -c "import pyarrow, scipy, torch; assert torch.cuda.is_available(); print('BENCHMARK_DEPENDENCIES=PASSED')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Script --config $Config --validate-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Script --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Ablations
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Figures
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Summary = Get-Content -LiteralPath (Join-Path $RepoRoot "reports\final_submission\stage9_benchmark\benchmark_summary.json") -Raw | ConvertFrom-Json
if ($Summary.status -ne "PASSED" -or [int]$Summary.run_count -ne 1350 -or [int]$Summary.unique_run_ids -ne 1350 -or [int]$Summary.missing_run_count -ne 0 -or [int]$Summary.failed_run_count -ne 0) {
    throw "Full benchmark summary did not pass the 1350-run gate."
}
$AblationSummary = Get-Content -LiteralPath (Join-Path $RepoRoot "reports\final_submission\stage9_benchmark\ablation_summary.json") -Raw | ConvertFrom-Json
if ($AblationSummary.status -ne "PASSED" -or [int]$AblationSummary.run_count -ne 180 -or [int]$AblationSummary.unique_run_count -ne 180) {
    throw "Ablation summary did not pass the 180-run gate."
}
Write-Host "COMPLETE_RESEARCH_BENCHMARK=PASSED"
