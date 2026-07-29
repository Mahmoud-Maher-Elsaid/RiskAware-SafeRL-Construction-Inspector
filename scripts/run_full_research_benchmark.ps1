param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script = Join-Path $RepoRoot "scripts\run_full_research_benchmark.py"
$Config = Join-Path $RepoRoot "configs\benchmarks\full_matrix.yaml"
foreach ($Required in @($Python, $Script, $Config)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required benchmark file is missing: $Required"
    }
}
Set-Location -LiteralPath $RepoRoot
& $Python $Script --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Summary = Get-Content -LiteralPath (Join-Path $RepoRoot "reports\final_submission\stage9_benchmark\benchmark_summary.json") -Raw | ConvertFrom-Json
if ($Summary.status -ne "PASSED" -or [int]$Summary.run_count -ne 1350 -or [int]$Summary.unique_run_ids -ne 1350) {
    throw "Full benchmark summary did not pass the 1350-run gate."
}
Write-Host "COMPLETE_RESEARCH_BENCHMARK=PASSED"
