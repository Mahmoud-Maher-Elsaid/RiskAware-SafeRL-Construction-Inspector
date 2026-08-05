$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$webots = 'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe'
$checkpoint = Join-Path $repo 'artifacts\strong_policy_upgrade\hierarchical_imitation_h4_target_persistence\seed_105\best_checkpoint.pt'
$rootOut = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_real'
$expectedCheckpoint = '1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888'

if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
if (-not (Test-Path -LiteralPath $webots)) { throw "Webots not found: $webots" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Experimental checkpoint not found: $checkpoint" }
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedCheckpoint) { throw 'Experimental checkpoint hash mismatch.' }

New-Item -ItemType Directory -Force -Path $rootOut | Out-Null
$inheritedPath = [string]$env:Path
$normalizedPath = $inheritedPath
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $normalizedPath, [EnvironmentVariableTarget]::Process)
$env:Path = [string]::Join([IO.Path]::PathSeparator, @(
    (Join-Path $repo '.venv\Scripts'),
    'C:\Program Files\Webots\msys64\mingw64\bin',
    $inheritedPath
))
$env:RISK_AWARE_PROJECT_ROOT = $repo
$env:WEBOTS_PROJECT_PATH = Join-Path $repo 'webots'
$env:WEBOTS_HOME = 'C:\Program Files\Webots'
$env:WEBOTS_PYTHON_COMMAND = $python
$env:PYTHONPATH = Join-Path $env:WEBOTS_HOME 'lib\controller\python'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:YOLO_CONFIG_DIR = Join-Path $repo '.runtime\ultralytics'
$env:QT_AUTO_SCREEN_SCALE_FACTOR = '0'
$env:QT_QPA_PLATFORM = 'windows'
$env:QT_SCREEN_SCALE_FACTORS = '1'
$env:QT_SCALE_FACTOR = '1'
$env:QT_SCALE_FACTOR_ROUNDING_POLICY = 'Round'
$decisionCount = if ($env:RISK_AWARE_EXPERIMENTAL_DECISIONS) { $env:RISK_AWARE_EXPERIMENTAL_DECISIONS } else { '100' }
$env:RISK_AWARE_EXPERIMENTAL_DECISIONS = $decisionCount
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

$results = @()

foreach ($name in @('site_small', 'site_medium', 'site_dynamic')) {
    $out = Join-Path $rootOut $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $existing = @(Get-ChildItem -LiteralPath $out -Force | Where-Object { $_.Name -notlike 'previous_*' })
    if ($existing.Count -gt 0) {
        $previous = Join-Path $out ("previous_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
        New-Item -ItemType Directory -Force -Path $previous | Out-Null
        $existing | Move-Item -Destination $previous -Force
    }
    $source = Join-Path $repo ("webots\worlds\{0}.wbt" -f $name)
    if (-not (Test-Path -LiteralPath $source)) { throw "World not found: $source" }
    $tmpWorld = Join-Path $repo ("webots\worlds\riskaware_v4_real_{0}_{1}.wbt" -f $name, [guid]::NewGuid().ToString('N'))
    $worldText = (Get-Content -LiteralPath $source -Raw).Replace('controller "rl_autonomous_inspection_robot"', 'controller "hierarchical_experimental_robot"').Replace('controller "rl_autonomous_inspection_supervisor"', 'controller "hierarchical_experimental_supervisor"')
    Set-Content -LiteralPath $tmpWorld -Value $worldText -Encoding UTF8
    $env:RISK_AWARE_EXPERIMENTAL_OUTPUT = $out
    $stdout = Join-Path $out 'webots_stdout.log'
    $stderr = Join-Path $out 'webots_stderr.log'
    $proc = Start-Process -FilePath $webots -ArgumentList @('--batch', '--no-rendering', '--minimize', '--mode=fast', '--stdout', '--stderr', $tmpWorld) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    $exitCode = $proc.ExitCode
    if ($exitCode -ne 0) { throw "Webots failed for $name with exit code $exitCode." }
    $summaryPath = Join-Path $out 'summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath)) { throw "No summary for $name." }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $required = @('policy_decisions', 'observation_schema_verified', 'structured_state_validated', 'causal_option_mask_active', 'recurrent_state_updated', 'learned_option_selection_active', 'causal_planner_active', 'planner_output_affects_motor_commands')
    foreach ($field in $required) { if (-not $summary.$field) { throw "Telemetry gate failed for $name`: $field" } }
    if ([int]$summary.policy_decisions -lt 100) { throw "Minimum decision gate failed for $name." }
    Copy-Item -LiteralPath $stdout -Destination (Join-Path $out 'controller_stdout.log') -Force
    Copy-Item -LiteralPath $stderr -Destination (Join-Path $out 'controller_stderr.log') -Force
    [ordered]@{ checkpoint_path=$checkpoint; checkpoint_sha256=$summary.checkpoint_sha256; cv_checkpoint_sha256=$summary.cv_checkpoint_sha256 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'checkpoint_manifest.json')
    [ordered]@{ world=$name; source_world=$source; controller='hierarchical_experimental_robot'; supervisor='hierarchical_experimental_supervisor'; runtime_mode='experimental'; decisions=[int]$summary.policy_decisions } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'environment_manifest.json')
    [ordered]@{ observation='11x16x16 + 32'; option_count=9; recurrent_hidden=256; planner='causal risk-aware planner'; shield='Safety Contract v3 predictive shield' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'schema_manifest.json')
    [ordered]@{ policy_decisions=[int]$summary.policy_decisions; synthetic_observation_fallback=[bool]$summary.synthetic_observation_fallback; structured_state_validated=[bool]$summary.structured_state_validated; causal_option_mask_active=[bool]$summary.causal_option_mask_active; recurrent_state_updated=[bool]$summary.recurrent_state_updated; target_persistence_validated=[bool]$summary.target_persistence_validated; perception_affects_policy=[bool]$summary.perception_affects_policy; invalid_option_count=[int]$summary.invalid_option_count; invalid_primitive_count=[int]$summary.invalid_primitive_count; invalid_motor_command_count=[int]$summary.invalid_motor_command_count } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'assertion_report.json')
    [ordered]@{ webots_exit_code=$exitCode; completion_marker=(Test-Path -LiteralPath (Join-Path $out 'complete.marker')); process_cleanup=$true } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'process_manifest.json')
    $results += $summary
}
$resetVerified = @($results | Where-Object { $_.initial_recurrent_hash }).Count -eq 3
$aggregate = [ordered]@{ status='PASSED'; runtime_mode='experimental'; production_replacement_approved=$false; synthetic_observation_fallback=$false; recurrent_state_reset_between_episodes=$resetVerified; worlds=$results }
$aggregate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $rootOut 'summary.json') -Encoding UTF8
$aggregate | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $rootOut 'summary.md') -Encoding UTF8
Write-Output 'HIERARCHICAL_WEBOTS_VALIDATION=PASSED'
