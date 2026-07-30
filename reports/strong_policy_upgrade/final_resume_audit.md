# Final strong-policy resume audit

Date: 2026-07-30

## Result

`FINAL_STRONG_POLICY_RECOVERY=PASSED`

The repository was recovered on `final/strong-policy-upgrade` at
`c0e81771ecc418804c0f3ab205ae82bd52c40899`. The local and GitHub branch
heads matched, and the working tree was clean before this audit.

## Preserved artifacts

- The immutable privileged dataset remains at 120,144 transitions with
  dataset SHA-256
  `06f872d819241563d3bf735d97c203d3de9633fbbc9cb3e5e99f95a1edbb1285`.
- The immutable systematic causal dataset remains at 250,128 transitions
  with dataset SHA-256
  `5278452907ed0f7a3ad1c4381caad643d9c03b74b3bff32747735223f6c7756f`.
- The v2 task-performance checkpoint was located and verified at
  `artifacts/strong_policy_upgrade/hrmppo_v2_search/anchor_lr5e6_100k/final_checkpoint.pt`
  with SHA-256
  `f296c5633ec6d3e9918474039ae079fb681e7dded02a4962c465010e2bf80436`.
- All prior failed candidates and negative reports remain present.

## Software baseline

Python compilation, Ruff check, Ruff format verification, and all 316 tests
passed.

## Pre-modification production regression

The required production command was executed before changing production
source. It failed before the Webots controller started because PowerShell
`Start-Process` encountered duplicate case variants of `Path` and `PATH` in
the inherited Windows environment. CUDA and Webots R2025a validation had
already passed. This is a launcher defect, not a policy or mission result,
and no production safety claim was derived from the failed run.

The duplicate environment-key defect was repaired and the Ultralytics
configuration directory was redirected to a repository-local ignored runtime
directory. Follow-up attempts reached controller startup, but Webots R2025a
then terminated while Qt attempted to allocate an invalid multi-million-pixel
desktop view. Batch, minimized, no-rendering, and normalized Qt scaling modes
did not eliminate that host display failure. The failed logs are preserved and
the production regression remains failed rather than being inferred from the
previously accepted evidence.

## Recovery decision

Recovery and artifact-integrity gates pass. Production remains on v1.
Expensive Safe RL training remains blocked until the 5,460-violation audit,
Safety Contract v3, vector-cost implementation, and shield/environment
consistency tests pass.
