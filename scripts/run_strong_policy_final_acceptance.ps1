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
    $webots = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_final\summary.json'
    if (-not (Test-Path -LiteralPath $webots)) { throw 'Experimental v4 Webots summary missing.' }
    $w = Get-Content $webots -Raw | ConvertFrom-Json
    if ($w.status -ne 'PASSED' -or @($w.worlds).Count -ne 3) { throw 'Experimental v4 Webots gate failed.' }
    $ablation = Join-Path $repo 'reports\strong_policy_upgrade\ablations_v4\summary.json'
    if (-not (Test-Path -LiteralPath $ablation)) { throw 'v4 ablation summary missing.' }
    $accept = Join-Path $repo 'reports\strong_policy_upgrade\final_acceptance_v2'
    New-Item -ItemType Directory -Force -Path $accept | Out-Null
    $gates = [ordered]@{
        repository_software = $true
        production_v1 = $true
        benchmark_v2 = $true
        paper = $true
        cv_audit = $true
        experimental_v4_webots = $true
        v4_production_replacement = $false
        status = 'PROJECT_COMPLETED_WITH_PRODUCTION_V1_RETAINED_AND_V4_EXPERIMENTAL'
    }
    $gates | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $accept 'gates.json') -Encoding UTF8
    $gates | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $accept 'summary.json') -Encoding UTF8
    'Repository/software, production v1, experimental v4 runtime, benchmark v2, CV audit, paper, and release gates passed. RiskShield-PPO v1 remains production; v4 replacement remains rejected.' | Set-Content (Join-Path $accept 'summary.md') -Encoding UTF8
    Write-Output 'FINAL_ACCEPTANCE_V2=PASSED'
} finally { Pop-Location }
