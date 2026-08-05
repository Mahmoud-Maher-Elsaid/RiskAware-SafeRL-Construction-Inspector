$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$output = Join-Path $repo 'reports\strong_policy_upgrade\benchmark_v2'
if (-not (Test-Path -LiteralPath $python)) { throw "Repository Python not found: $python" }
& $python (Join-Path $repo 'scripts\run_hierarchical_benchmark_v2.py') --output $output --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Hierarchical evaluation failed.' }
Write-Output 'HIERARCHICAL_EVALUATION=PASSED'
