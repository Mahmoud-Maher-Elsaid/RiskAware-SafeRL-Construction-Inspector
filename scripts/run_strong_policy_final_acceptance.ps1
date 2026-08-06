$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Repository Python not found: $python" }
Push-Location $repo
try {
    & $python -m compileall -q src scripts tests webots/controllers
    if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
    & $python -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff check failed.' }
    & $python -m ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw 'Ruff format check failed.' }
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Pytest failed.' }
    & (Join-Path $repo 'scripts\run_benchmark_v2.ps1')

    $productionPath = Join-Path $repo 'reports\final_project_completion\final_runtime_summary.json'
    $production = Get-Content -LiteralPath $productionPath -Raw | ConvertFrom-Json
    $productionGate = [bool]$production.mission_completed -and -not [bool]$production.manual_control_used -and -not [bool]$production.fallback_controller_used
    $webotsRoot = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_real'
    $worlds = @('site_small', 'site_medium', 'site_dynamic')
    $worldSummaries = @($worlds | ForEach-Object { Get-Content (Join-Path (Join-Path $webotsRoot $_) 'summary.json') -Raw | ConvertFrom-Json })
    $experimentalGate = ($worldSummaries.Count -eq 3) -and (($worldSummaries | Where-Object { [int]$_.policy_decisions -ge 100 -and $_.synthetic_observation_fallback -eq $false -and $_.observation_schema_verified -and $_.structured_state_validated -and $_.causal_option_mask_active -and $_.recurrent_state_updated -and $_.learned_option_selection_active -and $_.causal_planner_active -and $_.planner_output_affects_motor_commands -and $_.invalid_option_count -eq 0 -and $_.invalid_primitive_count -eq 0 -and $_.invalid_motor_command_count -eq 0 }).Count -eq 3)
    $benchmark = Get-Content (Join-Path $repo 'reports\strong_policy_upgrade\benchmark_v2\integrity_report.json') -Raw | ConvertFrom-Json
    $benchmarkGate = $benchmark.total_rows -eq 1620 -and $benchmark.unique_run_ids -eq 1620 -and $benchmark.duplicate_rows -eq 0 -and $benchmark.missing_rows -eq 0 -and $benchmark.unresolved_execution_failures -eq 0 -and -not $benchmark.historical_rows_changed
    $visiblePath = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_visible_demo\summary.json'
    $visible = Get-Content -LiteralPath $visiblePath -Raw | ConvertFrom-Json
    $visibleGate = $visible.status -eq 'PASSED' -and [bool]$visible.visible_window_verified -and [bool]$visible.controller_started -and [bool]$visible.supervisor_started -and [bool]$visible.summary_generated -and [int]$visible.policy_decisions -ge 100
    $paperValidationPath = Join-Path $repo 'reports\strong_policy_upgrade\paper_validation\summary.json'
    $paperValidation = Get-Content -LiteralPath $paperValidationPath -Raw | ConvertFrom-Json
    $paperGate = (Test-Path -LiteralPath (Join-Path $repo 'paper\main.pdf')) -and $paperValidation.rebuilt_from_latest_source -eq $true -and $paperValidation.visual_inspection -eq 'PASSED' -and [int]$paperValidation.fatal_errors -eq 0 -and [int]$paperValidation.undefined_citations -eq 0 -and [int]$paperValidation.undefined_references -eq 0
    $cvGate = Test-Path -LiteralPath (Join-Path $repo 'reports\strong_policy_upgrade\cv_final_audit\summary.json')
    $ablationGate = Test-Path -LiteralPath (Join-Path $repo 'reports\strong_policy_upgrade\ablations_v4\summary.json')
    $releaseGate = Test-Path -LiteralPath (Join-Path $repo 'release\manifest.json')
    $gates = [ordered]@{
        repository_software = $true
        github_ci = (Test-Path -LiteralPath (Join-Path $repo '.github\workflows\ci.yml'))
        production_v1 = $productionGate
        experimental_v4_webots = $experimentalGate
        visible_v4_demo = $visibleGate
        benchmark_v2 = $benchmarkGate
        cv_audit = $cvGate
        ablations_statistics = $ablationGate
        paper = $paperGate
        documentation_release = $releaseGate
        production_replacement_approved = $false
        production_policy = 'RiskShield-PPO v1'
        status = 'PROJECT_COMPLETED_WITH_PRODUCTION_V1_RETAINED_AND_V4_EXPERIMENTAL'
    }
    $accept = Join-Path $repo 'reports\strong_policy_upgrade\final_acceptance_v2'
    New-Item -ItemType Directory -Force -Path $accept | Out-Null
    $gates | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $accept 'gates.json') -Encoding UTF8
    [ordered]@{ status=$gates.status; gates=$gates; worlds=$worldSummaries; visible_demo=$visible; paper_validation=$paperValidation; benchmark=$benchmark } | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $accept 'summary.json') -Encoding UTF8
    [ordered]@{ webots_root=$webotsRoot; world_count=$worldSummaries.Count; benchmark_rows=$benchmark.total_rows; paper='paper/main.pdf'; release='release/manifest.json' } | ConvertTo-Json | Set-Content (Join-Path $accept 'artifact_manifest.json') -Encoding UTF8
    $failed = @($gates.GetEnumerator() | Where-Object { $_.Value -is [bool] -and $_.Value -eq $false -and $_.Key -ne 'production_replacement_approved' })
    if ($failed.Count -gt 0) { throw 'One or more final acceptance gates failed.' }
    'Repository/software, production v1, corrected experimental v4 runtime, benchmark v2, CV audit, ablations, paper, and release gates passed. RiskShield-PPO v1 remains production; v4 replacement remains rejected.' | Set-Content (Join-Path $accept 'summary.md') -Encoding UTF8
    Write-Output 'FINAL_ACCEPTANCE_V2=PASSED'
} finally { Pop-Location }
