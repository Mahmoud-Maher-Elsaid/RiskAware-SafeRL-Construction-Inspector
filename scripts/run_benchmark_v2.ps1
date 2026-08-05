$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$summary = Join-Path $repo 'reports\strong_policy_upgrade\benchmark_v2\integrity_report.json'
if (-not (Test-Path -LiteralPath $python)) { throw "Repository Python not found: $python" }
if (-not (Test-Path -LiteralPath $summary)) {
    & $python (Join-Path $repo 'scripts\build_benchmark_v2_outputs.py')
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark v2 output construction failed.' }
}
$report = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
if ($report.total_rows -ne 1620 -or $report.unique_run_ids -ne 1620 -or $report.duplicate_rows -ne 0 -or $report.missing_rows -ne 0 -or $report.unresolved_execution_failures -ne 0) {
    throw 'Benchmark v2 integrity gate failed.'
}
Write-Output 'BENCHMARK_V2=PASSED'
