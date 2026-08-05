param(
    [ValidateSet('site_small','site_medium','site_dynamic')][string]$World = 'site_dynamic',
    [int]$DurationSeconds = 90
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$webots = 'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe'
$checkpoint = Join-Path $repo 'artifacts\strong_policy_upgrade\hierarchical_imitation_h4_target_persistence\seed_105\best_checkpoint.pt'
$out = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_visible_demo'
if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
if (-not (Test-Path -LiteralPath $webots)) { throw "Webots not found: $webots" }
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne '1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888') { throw 'Experimental checkpoint hash mismatch.' }
New-Item -ItemType Directory -Force -Path $out | Out-Null
$existing = @(Get-ChildItem -LiteralPath $out -Force | Where-Object { $_.Name -notlike 'previous_*' })
if ($existing.Count -gt 0) {
    $previous = Join-Path $out ("previous_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Force -Path $previous | Out-Null
    $existing | Move-Item -Destination $previous -Force
}
$source = Join-Path $repo ("webots\worlds\{0}.wbt" -f $World)
$tmpWorld = Join-Path $repo ("webots\worlds\riskaware_v4_demo_{0}_{1}.wbt" -f $World, [guid]::NewGuid().ToString('N'))
$worldText = (Get-Content -LiteralPath $source -Raw).Replace('controller "rl_autonomous_inspection_robot"', 'controller "hierarchical_experimental_robot"').Replace('controller "rl_autonomous_inspection_supervisor"', 'controller "hierarchical_experimental_supervisor"')
Set-Content -LiteralPath $tmpWorld -Value $worldText -Encoding UTF8
$inheritedPath = [string]$env:Path
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $inheritedPath, [EnvironmentVariableTarget]::Process)
$env:Path = [string]::Join([IO.Path]::PathSeparator, @((Join-Path $repo '.venv\Scripts'), 'C:\Program Files\Webots\msys64\mingw64\bin', $inheritedPath))
$env:RISK_AWARE_PROJECT_ROOT = $repo
$env:RISK_AWARE_EXPERIMENTAL_OUTPUT = $out
$env:RISK_AWARE_EXPERIMENTAL_DECISIONS = '100'
$env:RISK_AWARE_EXPERIMENTAL_DEMO_DURATION = [string]$DurationSeconds
$env:WEBOTS_HOME = 'C:\Program Files\Webots'
$env:WEBOTS_PYTHON_COMMAND = $python
$env:PYTHONPATH = Join-Path $env:WEBOTS_HOME 'lib\controller\python'
$env:YOLO_CONFIG_DIR = Join-Path $repo '.runtime\ultralytics'
$env:QT_AUTO_SCREEN_SCALE_FACTOR = '0'
$env:QT_QPA_PLATFORM = 'windows'
$env:QT_SCREEN_SCALE_FACTORS = '1'
$env:QT_SCALE_FACTOR = '1'
$env:QT_SCALE_FACTOR_ROUNDING_POLICY = 'Round'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
$stdout = Join-Path $out 'demo_stdout.log'
$stderr = Join-Path $out 'demo_stderr.log'
Write-Output "VISIBLE_WEBOTS_DEMO_OUTPUT=$out"
Write-Output "VISIBLE_WEBOTS_DEMO_CHECKPOINT=$checkpoint"
$proc = Start-Process -FilePath $webots -ArgumentList @('--mode=realtime', $tmpWorld) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$deadline = (Get-Date).AddSeconds($DurationSeconds + 60)
try {
    while ((Get-Date) -lt $deadline) {
        $live = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if (-not $live) { break }
        Start-Sleep -Seconds 2
        if (Test-Path -LiteralPath (Join-Path $out 'option_trace.csv')) {
            $last = Get-Content -LiteralPath (Join-Path $out 'option_trace.csv') -Tail 1
            Write-Output "DEMO_OPTION_TRACE=$last"
        }
        $proc.Refresh()
    }
    if (-not (Test-Path -LiteralPath (Join-Path $out 'summary.json'))) { throw 'Visible Webots demo exited without a summary.' }
    $proc.Refresh()
    if (-not $proc.HasExited -and (Get-Date) -ge $deadline) { throw 'Visible demo exceeded bounded duration.' }
    if ($proc.HasExited -and $proc.ExitCode -ne 0) { throw "Visible Webots demo exited with code $($proc.ExitCode)." }
    $summary = Get-Content -LiteralPath (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
    if ([int]$summary.policy_decisions -lt 100) { throw 'Visible demo did not reach 100 policy decisions.' }
    [ordered]@{ world=$World; duration_seconds=$DurationSeconds; exit_code=$proc.ExitCode; summary=$summary; checkpoint_sha256='1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888' } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $out 'summary.json') -Encoding UTF8
    Write-Output 'REAL_V4_VISIBLE_DEMO=PASSED'
}
finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
