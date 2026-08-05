param(
    [string]$RepoRoot = "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot
$Output = Join-Path $RepoRoot "reports\final_submission\release_manifest.json"
$ReleaseDir = Join-Path $RepoRoot "release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
$Files = git ls-files
$Records = foreach ($Relative in $Files) {
    $Path = Join-Path $RepoRoot $Relative
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $Item = Get-Item -LiteralPath $Path
        [ordered]@{
            path = $Relative.Replace("\", "/")
            bytes = $Item.Length
            sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}
$Manifest = [ordered]@{
    schema_version = 1
    generated_at = [DateTimeOffset]::Now.ToString("o")
    git_commit = (git rev-parse HEAD)
    tracked_file_count = @($Records).Count
    files = @($Records)
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding utf8
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ReleaseDir "manifest.json") -Encoding utf8
$Sums = foreach ($Record in @($Records)) { "$($Record.sha256)  $($Record.path)" }
$Sums | Set-Content -LiteralPath (Join-Path $ReleaseDir "SHA256SUMS.txt") -Encoding utf8
@"
# RiskAware SafeRL Construction Inspector Release

This package contains source, configuration, compact reports, benchmark summaries, paper source/PDF, and reproducibility metadata. Raw datasets and local-only checkpoints are intentionally excluded; their expected paths and SHA-256 values are recorded in the research reports and model manifests.

Production remains RiskShield-PPO v1. The hierarchical HRMPPO-MPC v4 system is experimental and its production-replacement gate was rejected by paired evidence.
"@ | Set-Content -LiteralPath (Join-Path $ReleaseDir "README.md") -Encoding utf8
Write-Host "FINAL_RELEASE_MANIFEST=PASSED ($(@($Records).Count) tracked files)"
