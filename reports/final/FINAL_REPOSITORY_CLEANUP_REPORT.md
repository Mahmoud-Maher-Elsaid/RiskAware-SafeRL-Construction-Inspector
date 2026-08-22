# Final Repository Cleanup Report

Audit date: 2026-08-22

## Scope and preservation

- Branch: `fix/manual-first-person-free-exploration`
- HEAD and upstream: `97cec6e72c3f121b1cab42076ede21e9aca8476b`
- Tracked files inventoried: 879
- Untracked paths before cleanup: 10,059
- Untracked paths after cleanup: 1,465 visible to Git (canonical audit/research
  reports and required validation fixtures); local datasets/checkpoints and
  intermediate diagnostics are now covered by narrow ignore rules.
- Modified tracked files at audit close: 16
- Disposable caches, generated runtime directories, 184 superseded visible-demo
  attempt folders, and 70 intermediate strong-policy artifact directories were
  deleted after explicit provenance checks. Five tracked manifest files found
  inside deleted artifact directories were restored immediately from HEAD.
- A backup of every pre-existing modified tracked file was written under
  `.local-audit-backup/20260822_121816/`.

## Changes made

- Added professional Python/OS/temp ignores to `.gitignore`.
- Updated README and `docs/final_demo.md` to distinguish the static v4 overview
  demo from historical first-person Stage 5 evidence.
- Updated the paper's Webots wording from first-person rendering to rendered
  Webots operation; benchmark values and negative findings remain unchanged.
- Updated the camera contract test and v4 launcher overview template to match
  the user-approved site-centered overview configuration.

## Deletions

- Files removed by cleanup operations: 12,417; five tracked manifests were
  restored from HEAD, for a net deletion of 12,412 disposable files.
- Directories deleted: 350
- Approximate bytes removed: 23,598,270,872
- Cache files: 676 (`__pycache__`, pytest, and Ruff caches)
- Runtime-attempt files: 4,904, with the latest failure/smoke and approved
  overview evidence retained
- Intermediate artifact files: 6,719 (23.3 GB); required H4 checkpoint and
  accepted/final artifact directories retained
- Generated Webots worlds/markers/backups: 36 files
- Tracked files deleted: 0 (five accidentally selected manifests were restored)
- Duplicate checkpoints removed: 0
- Repair backups retained under `.local-audit-backup/` and
  `reports/final_refactor_backup/` because they contain pre-existing work or
  pre-change provenance.

## Final release validation

- `compileall`: passed
- Ruff lint: passed
- Ruff format check: passed (274 files already formatted)
- Full pytest: passed
- `pip check`: passed
- Paper citation-key and figure/table reference checks: passed
- Paper build: fresh four-page PDF rebuilt with repository-local output after
  repairing only the current user's `%LOCALAPPDATA%\\MiKTeX` ACL; no fatal,
  undefined citation/reference, or missing-figure diagnostics.
- Automated Webots GUI smoke: environment-limited under the non-interactive
  Qt session. The user's direct interactive Webots run and approved overview
  camera remain the acceptance evidence.

## Final disposition

Canonical research outputs, fixtures, audit reports, and required runtime files
are explicitly staged. Local datasets, checkpoints, caches, and runtime
workspaces remain ignored with narrow rules. Release integration and push are
performed only after the staged diff and final static gates pass.
