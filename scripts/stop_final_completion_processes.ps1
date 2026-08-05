param([string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector")
$root = (Resolve-Path $RepoRoot).Path
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($root) } | ForEach-Object {
    if ($_.ProcessId -ne $PID) { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Write-Host "FINAL_COMPLETION_PROJECT_PROCESSES_STOPPED"
