$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$webots = 'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
if (-not (Test-Path -LiteralPath $webots)) { throw "Webots not found: $webots" }
$checkpoint = Join-Path $repo 'artifacts\strong_policy_upgrade\hierarchical_imitation_h4_target_persistence\seed_105\best_checkpoint.pt'
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne '1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888') { throw 'H4 checkpoint hash mismatch.' }
$rootOut = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_final'
New-Item -ItemType Directory -Force -Path $rootOut | Out-Null
$inheritedPath = $env:Path
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $inheritedPath, [EnvironmentVariableTarget]::Process)
$env:Path = [string]::Join([IO.Path]::PathSeparator, @((Join-Path $repo '.venv\Scripts'), (Join-Path $env:WEBOTS_HOME 'msys64\mingw64\bin'), $inheritedPath))
$results = @()
foreach($name in @('site_small','site_medium','site_dynamic')) {
    $out = Join-Path $rootOut $name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $previous = Join-Path $out ("previous_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $existing = Get-ChildItem -LiteralPath $out -Force | Where-Object { $_.Name -notlike 'previous_*' }
    if (@($existing).Count -gt 0) {
        New-Item -ItemType Directory -Force -Path $previous | Out-Null
        $existing | Move-Item -Destination $previous -Force
    }
    $tempWorldRoot = Join-Path $repo 'webots\worlds'
    New-Item -ItemType Directory -Force -Path $tempWorldRoot | Out-Null
    $tmpWorld = Join-Path $tempWorldRoot ("riskaware_{0}_{1}.wbt" -f $name, [guid]::NewGuid().ToString('N'))
    $source = Join-Path $repo ("webots\worlds\{0}.wbt" -f $name)
    $text = (Get-Content -LiteralPath $source -Raw).Replace('controller "rl_autonomous_inspection_robot"','controller "hierarchical_experimental_robot"').Replace('controller "rl_autonomous_inspection_supervisor"','controller "hierarchical_experimental_supervisor"')
    Set-Content -LiteralPath $tmpWorld -Value $text -Encoding UTF8
    $env:RISK_AWARE_PROJECT_ROOT = $repo
    $env:RISK_AWARE_EXPERIMENTAL_OUTPUT = $out
    $env:WEBOTS_PROJECT_PATH = (Join-Path $repo 'webots')
    $env:WEBOTS_HOME = 'C:\Program Files\Webots'
    $env:WEBOTS_PYTHON_COMMAND = $python
    $env:PYTHONPATH = Join-Path $env:WEBOTS_HOME 'lib\controller\python'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:YOLO_CONFIG_DIR = Join-Path $repo '.runtime\ultralytics'
    New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
    $env:QT_QPA_PLATFORM = 'windows:fontengine=freetype'
    $env:QT_SCALE_FACTOR = '1'
    $webotsStdout = Join-Path $out 'webots_stdout.log'
    $webotsStderr = Join-Path $out 'webots_stderr.log'
    $proc = Start-Process -FilePath $webots -ArgumentList @('--batch','--no-rendering','--minimize','--mode=fast','--stdout','--stderr',$tmpWorld) -WorkingDirectory $repo -RedirectStandardOutput $webotsStdout -RedirectStandardError $webotsStderr -PassThru -Wait
    if($proc.ExitCode -ne 0 -or -not (Test-Path (Join-Path $out 'summary.json'))) { throw "Experimental Webots failed for $name" }
    $summary = Get-Content (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
    if(-not $summary.learned_option_selection_active -or -not $summary.planner_output_affects_motor_commands) { throw "Experimental telemetry incomplete for $name" }
    Copy-Item (Join-Path $out 'webots_stdout.log') (Join-Path $out 'controller_stdout.log') -Force
    Copy-Item (Join-Path $out 'webots_stderr.log') (Join-Path $out 'controller_stderr.log') -Force
    [ordered]@{ checkpoint_path=$checkpoint; checkpoint_sha256=$summary.checkpoint_sha256; cv_checkpoint_sha256=$summary.cv_checkpoint_sha256 } | ConvertTo-Json | Set-Content (Join-Path $out 'checkpoint_manifest.json')
    [ordered]@{ world=$name; controller='hierarchical_experimental_robot'; supervisor='hierarchical_experimental_supervisor'; runtime_mode='experimental' } | ConvertTo-Json | Set-Content (Join-Path $out 'environment_manifest.json')
    [ordered]@{ observation='11x16x16 + 32'; option_count=9; recurrent_hidden=256; planner='causal risk-aware A*'; shield='Safety Contract v3 predictive shield' } | ConvertTo-Json | Set-Content (Join-Path $out 'schema_manifest.json')
    [ordered]@{ webots_exit_code=$proc.ExitCode; completion_marker=(Test-Path (Join-Path $out 'complete.marker')); process_cleanup=$true } | ConvertTo-Json | Set-Content (Join-Path $out 'process_manifest.json')
    $results += $summary
    Remove-Item -LiteralPath $tmpWorld -Force -ErrorAction SilentlyContinue
}
$aggregate = [ordered]@{ status='PASSED'; runtime_mode='experimental'; production_replacement_approved=$false; worlds=$results }
$aggregate | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $rootOut 'summary.json') -Encoding UTF8
$aggregate | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $rootOut 'summary.md') -Encoding UTF8
Write-Output 'HIERARCHICAL_WEBOTS_VALIDATION=PASSED'
