# GitHub Actions CI discovery

The repository contains `.github/workflows/ci.yml` on the research branch and on `origin/main`.

The workflow is named `CI`, runs on pushes to `main`, `stage-5b-live-perception-integration`, and `final/strong-policy-upgrade`, and runs on pull requests targeting `main` or the integration branch. It also supports `workflow_dispatch`.

The known failed run was CI run `31030581950`, job `92389966770`, whose direct failure was `ruff format --check .` on `scripts/train_hierarchical_hrmppo_mpc_v4.py`. The Node.js deprecation annotation was not the direct failure.

The local GitHub CLI is unavailable in this environment and the unauthenticated REST endpoint returned 404, so a current remote run conclusion cannot be independently queried here. Local equivalents pass after formatting the script.
