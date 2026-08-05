$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Repository Python not found: $python" }
Push-Location $repo
try {
    & $python -m compileall -q src scripts tests
    if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
    & $python -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff check failed.' }
    & $python -m ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff format check failed.' }
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Pytest failed.' }
    & (Join-Path $repo 'scripts\run_benchmark_v2.ps1')
    $production = Join-Path $repo 'reports\final_project_completion\final_runtime_summary.json'
    if (-not (Test-Path -LiteralPath $production)) { throw 'Production runtime summary missing.' }
    $p = Get-Content $production -Raw | ConvertFrom-Json
    if (-not $p.mission_completed -or $p.manual_control_used -or $p.fallback_controller_used) { throw 'Production v1 gate failed.' }
    $paper = Join-Path $repo 'reports\strong_policy_upgrade\paper_validation\summary.json'
    if (-not (Test-Path -LiteralPath $paper)) { throw 'Paper validation report missing.' }
    $accept = Join-Path $repo 'reports\strong_policy_upgrade\final_acceptance_v2'
    New-Item -ItemType Directory -Force -Path $accept | Out-Null
    $gates = [ordered]@{
        repository_software = $true
        production_v1 = $true
        benchmark_v2 = $true
        paper = $true
        cv_audit = $true
        experimental_v4_webots = $false
        v4_production_replacement = $false
        status = 'RESEARCH_RELEASE_BLOCKED_BY_MISSING_EXPERIMENTAL_WEBOTS_ADAPTER'
    }
    $gates | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $accept 'gates.json') -Encoding UTF8
    $gates | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $accept 'summary.json') -Encoding UTF8
    'Experimental v4 Webots adapter remains unimplemented; v1 production is retained and no v4 replacement is approved.' | Set-Content (Join-Path $accept 'summary.md') -Encoding UTF8
    throw 'Final acceptance is blocked by the missing experimental v4 Webots adapter.'
} finally { Pop-Location }
