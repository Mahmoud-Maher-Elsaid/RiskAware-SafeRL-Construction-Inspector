param([string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$StateRoot = Join-Path $RepoRoot ".runtime\final_completion"
New-Item -ItemType Directory -Path $StateRoot -Force | Out-Null
$State = Join-Path $StateRoot "state.json"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Output = Join-Path $RepoRoot "reports\strong_policy_upgrade\benchmark_v2"
if (-not (Test-Path $Python -PathType Leaf)) { throw "Python missing: $Python" }
$existing = Join-Path $Output "hierarchical_v4_rows.csv"
if (-not (Test-Path $existing)) {
    @{ phase = "hierarchical_benchmark_v2"; status = "running"; started_at = [DateTimeOffset]::Now.ToString("o"); command = "$Python scripts/run_hierarchical_benchmark_v2.py --output $Output --device cuda" } | ConvertTo-Json | Set-Content $State
    & $Python (Join-Path $RepoRoot "scripts\run_hierarchical_benchmark_v2.py") --output $Output --device cuda
    if ($LASTEXITCODE -ne 0) { @{ phase = "hierarchical_benchmark_v2"; status = "failed"; exit_code = $LASTEXITCODE } | ConvertTo-Json | Set-Content $State; exit $LASTEXITCODE }
}
$rows = @(Import-Csv $existing)
if ($rows.Count -ne 270 -or @($rows.run_id | Sort-Object -Unique).Count -ne 270) { throw "Hierarchical benchmark integrity failed: $($rows.Count) rows" }
@{ phase = "hierarchical_benchmark_v2"; status = "passed"; completed = $rows.Count; completed_at = [DateTimeOffset]::Now.ToString("o") } | ConvertTo-Json | Set-Content $State
Write-Host "FINAL_COMPLETION_ORCHESTRATOR=PASSED"
