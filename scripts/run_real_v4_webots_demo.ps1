param(
    [ValidateSet('site_small', 'site_medium', 'site_dynamic')]
    [string]$World = 'site_dynamic',
    [ValidateRange(30, 600)]
    [int]$DurationSeconds = 90
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$webots = 'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe'
$checkpoint = Join-Path $repo 'artifacts\strong_policy_upgrade\hierarchical_imitation_h4_target_persistence\seed_105\best_checkpoint.pt'
$rootOut = Join-Path $repo 'reports\strong_policy_upgrade\webots_v4_visible_demo'
$source = Join-Path $repo ("webots\worlds\{0}.wbt" -f $World)
$stableWorld = Join-Path $repo ("webots\worlds\{0}_v4_visible_demo.wbt" -f $World)
$checkpointSha = '1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888'

foreach ($path in @($python, $webots, $checkpoint, $source)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path not found: $path" }
}
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checkpointSha) { throw 'Experimental checkpoint hash mismatch.' }

New-Item -ItemType Directory -Force -Path $rootOut | Out-Null
$attemptName = 'attempt_{0}' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$attempt = Join-Path $rootOut $attemptName
New-Item -ItemType Directory -Force -Path $attempt | Out-Null
$worldText = (Get-Content -LiteralPath $source -Raw).Replace('controller "rl_autonomous_inspection_robot"', 'controller "hierarchical_experimental_robot"').Replace('controller "rl_autonomous_inspection_supervisor"', 'controller "hierarchical_experimental_supervisor"')
Set-Content -LiteralPath $stableWorld -Value $worldText -Encoding UTF8
if (-not ((Get-Content -LiteralPath $stableWorld -Raw) -match 'controller "hierarchical_experimental_robot"')) { throw 'Stable world robot controller was not replaced.' }
if (-not ((Get-Content -LiteralPath $stableWorld -Raw) -match 'controller "hierarchical_experimental_supervisor"')) { throw 'Stable world supervisor was not replaced.' }

$inheritedPath = [string]$env:Path
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $inheritedPath, [EnvironmentVariableTarget]::Process)
$env:Path = [string]::Join([IO.Path]::PathSeparator, @((Join-Path $repo '.venv\Scripts'), 'C:\Program Files\Webots\msys64\mingw64\bin', $inheritedPath))
$env:RISK_AWARE_PROJECT_ROOT = $repo
$env:RISK_AWARE_EXPERIMENTAL_OUTPUT = $attempt
$env:RISK_AWARE_EXPERIMENTAL_DECISIONS = '100'
$env:RISK_AWARE_EXPERIMENTAL_DEMO_DURATION = [string]$DurationSeconds
$env:WEBOTS_HOME = 'C:\Program Files\Webots'
$env:WEBOTS_PYTHON_COMMAND = $python
$env:PYTHONPATH = Join-Path $env:WEBOTS_HOME 'lib\controller\python'
$env:YOLO_CONFIG_DIR = Join-Path $repo '.runtime\ultralytics'
$env:QT_AUTO_SCREEN_SCALE_FACTOR = '0'
$env:QT_ENABLE_HIGHDPI_SCALING = '0'
$env:QT_DEVICE_PIXEL_RATIO = '1'
$env:QT_QPA_PLATFORM = 'windows'
$env:QT_SCREEN_SCALE_FACTORS = '1'
$env:QT_SCALE_FACTOR = '0.01'
$env:QT_SCALE_FACTOR_ROUNDING_POLICY = 'Round'
$env:QT_OPENGL = 'software'
$env:LIBGL_ALWAYS_SOFTWARE = '1'
$env:WEBOTS_USE_GLES = '1'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

$launcherState = [ordered]@{ attempt=$attemptName; world=$World; stable_world=$stableWorld; duration_seconds=$DurationSeconds; launcher_started=(Get-Date).ToUniversalTime().ToString('o'); webots_process_started=$false; visible_window_verified=$false; controller_started=$false; supervisor_started=$false; summary_generated=$false; exit_code=$null }
$launcherState | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attempt 'launcher_state.json') -Encoding UTF8
$stdout = Join-Path $attempt 'webots_stdout.log'
$stderr = Join-Path $attempt 'webots_stderr.log'
$args = @('--batch', '--mode=realtime', '--no-rendering', '--minimize', '--stdout', '--stderr', $stableWorld)
$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $webots
$psi.WorkingDirectory = $repo
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false
$quotedArgs = ($args | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\\"') + '"' } else { $_ } }) -join ' '
$psi.Arguments = $quotedArgs
foreach ($key in @($psi.Environment.Keys)) {
    if ($key -ieq 'Path' -or $key -ieq 'QT_SCALE_FACTOR' -or $key -ieq 'QT_SCREEN_SCALE_FACTORS' -or $key -ieq 'QT_AUTO_SCREEN_SCALE_FACTOR') {
        [void]$psi.Environment.Remove($key)
    }
}
$psi.Environment['Path'] = $env:Path
$psi.Environment['RISK_AWARE_PROJECT_ROOT'] = $repo
$psi.Environment['RISK_AWARE_EXPERIMENTAL_OUTPUT'] = $attempt
$psi.Environment['RISK_AWARE_EXPERIMENTAL_DECISIONS'] = '100'
$psi.Environment['RISK_AWARE_EXPERIMENTAL_DEMO_DURATION'] = [string]$DurationSeconds
$psi.Environment['WEBOTS_HOME'] = $env:WEBOTS_HOME
$psi.Environment['WEBOTS_PYTHON_COMMAND'] = $python
$psi.Environment['PYTHONPATH'] = $env:PYTHONPATH
$psi.Environment['YOLO_CONFIG_DIR'] = $env:YOLO_CONFIG_DIR
$psi.Environment['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
$psi.Environment['QT_ENABLE_HIGHDPI_SCALING'] = '0'
$psi.Environment['QT_DEVICE_PIXEL_RATIO'] = '1'
$psi.Environment['QT_QPA_PLATFORM'] = 'windows'
$psi.Environment['QT_SCREEN_SCALE_FACTORS'] = '1'
$psi.Environment['QT_SCALE_FACTOR'] = '0.01'
$psi.Environment['QT_SCALE_FACTOR_ROUNDING_POLICY'] = 'Round'
$psi.Environment['QT_OPENGL'] = 'software'
$psi.Environment['LIBGL_ALWAYS_SOFTWARE'] = '1'
$psi.Environment['WEBOTS_USE_GLES'] = '1'
$proc = [System.Diagnostics.Process]::new()
$proc.StartInfo = $psi
[void]$proc.Start()
$restoreType = Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class WebotsWindowRestore {
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
'@ -PassThru
$launcherState.webots_process_started = $true
$launcherState.webots_pid = $proc.Id
$launcherState.world_load_requested = (Get-Date).ToUniversalTime().ToString('o')
$launcherState | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attempt 'launcher_state.json') -Encoding UTF8
$deadline = (Get-Date).AddSeconds($DurationSeconds + 90)
$startupDeadline = (Get-Date).AddSeconds(45)
$restored = $false
Write-Output "VISIBLE_WEBOTS_ATTEMPT=$attempt"
Write-Output "VISIBLE_WEBOTS_WORLD=$stableWorld"
try {
    while ((Get-Date) -lt $deadline) {
        $live = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if (-not $live) { break }
        if ((Get-Date) -lt $startupDeadline -and (Test-Path -LiteralPath (Join-Path $attempt 'controller_start.json'))) {
            $launcherState.controller_started = $true
            $launcherState.supervisor_started = Test-Path -LiteralPath (Join-Path $attempt 'marker_supervisor_main_started.json')
            $launcherState | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attempt 'launcher_state.json') -Encoding UTF8
        }
        if (-not $restored -and (Test-Path -LiteralPath (Join-Path $attempt 'summary.json'))) {
            $window = Get-Process -Name 'webots','webots-bin' -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
            if ($window) { [WebotsWindowRestore]::ShowWindow($window.MainWindowHandle, 9) | Out-Null; $launcherState.visible_window_verified = $true; $restored = $true }
        }
        if (Test-Path -LiteralPath (Join-Path $attempt 'failure.json')) { throw 'Experimental controller reported failure.' }
        if (Test-Path -LiteralPath (Join-Path $attempt 'option_trace.csv')) {
            $last = Get-Content -LiteralPath (Join-Path $attempt 'option_trace.csv') -Tail 1
            if ($last) { Write-Output "DEMO_OPTION_TRACE=$last" }
        }
        Start-Sleep -Seconds 2
        $proc.Refresh()
        if ((Test-Path -LiteralPath (Join-Path $attempt 'summary.json')) -and (Test-Path -LiteralPath (Join-Path $attempt 'complete.marker')) -and ((Get-Date) -gt (Get-Item (Join-Path $attempt 'complete.marker')).LastWriteTime.AddSeconds(15))) { break }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $attempt 'controller_start.json'))) { throw 'Visible controller did not start within 45 seconds.' }
    if (-not (Test-Path -LiteralPath (Join-Path $attempt 'summary.json'))) { throw 'Visible demo summary was not generated.' }
    $summary = Get-Content -LiteralPath (Join-Path $attempt 'summary.json') -Raw | ConvertFrom-Json
    if ([int]$summary.policy_decisions -lt 100) { throw 'Visible demo did not reach 100 policy decisions.' }
    $launcherState.summary_generated = $true
    $launcherState.controller_started = $true
    $launcherState.supervisor_started = Test-Path -LiteralPath (Join-Path $attempt 'marker_supervisor_main_started.json')
    $launcherState | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attempt 'launcher_state.json') -Encoding UTF8
    [ordered]@{ status='PASSED'; attempt=$attempt; world=$World; duration_seconds=$DurationSeconds; visible_window_verified=[bool]$launcherState.visible_window_verified; controller_started=$true; supervisor_started=[bool]$launcherState.supervisor_started; summary_generated=$true; policy_decisions=[int]$summary.policy_decisions; summary=$summary; checkpoint_sha256=$checkpointSha } | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $rootOut 'summary.json') -Encoding UTF8
    Write-Output 'REAL_V4_VISIBLE_DEMO=PASSED'
}
finally {
    if ($proc) {
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" -ErrorAction SilentlyContinue)
        foreach ($child in $children) { Stop-Process -Id ([int]$child.ProcessId) -Force -ErrorAction SilentlyContinue }
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
