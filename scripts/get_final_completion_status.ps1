param([string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector")
$state = Join-Path $RepoRoot ".runtime\final_completion\state.json"
if (Test-Path $state) { Get-Content $state -Raw } else { '{"status":"not_started"}' }
