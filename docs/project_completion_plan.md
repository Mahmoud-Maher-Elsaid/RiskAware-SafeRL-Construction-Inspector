# Project Completion Plan

Last updated: 2026-08-06

## Live checklist

| Field | Current value |
|---|---|
| Current task | Final research release with RiskShield-PPO v1 production and hierarchical v4 experimental |
| Current state | The v4 experimental runtime, three-world validation, benchmark v2, ablations, paper, release package, final acceptance, integration merge, and main CI are complete. v4 replacement remains rejected by paired evidence. |
| Evidence collected | Causal BC summary and memory gate pass; three flat DAgger iterations pass; H2/H3/H4 paired evidence; real v4 Webots telemetry; 1,620-row benchmark integrity; CV audit; rebuilt four-page paper; release SHA-256 manifest |
| Files changed | Safety contract and migration; recurrent constrained PPO; vector-cost buffer; causal shield; hierarchical policy/planner/controller; experimental Webots bridge; benchmark/acceptance/release tooling; documentation and paper |
| Tests run | Compileall, Ruff lint/format, full pytest (369 tests), production Webots, three-world v4 Webots, benchmark integrity, paper validation, release validation, and GitHub Actions all pass |
| Remaining work | No further justified production replacement is available. Preserve rejected v2/v3 and hierarchical-Dagger evidence as negative research results; retain v1 production and v4 experimental. |
| Blockers | None for the accepted research release. Independent labeled CV metrics and a v4 production replacement remain unavailable/rejected and are documented limitations. |

## Phase checklist

- [x] Preserve current tracked, untracked, ignored runtime, and repair-backup state externally.
- [x] Inspect branch, HEAD, top-level tree, dependency definitions, CI, Stage 5 assets, route, and launcher baseline.
- [x] Create permanent repository working rules.
- [x] Recover Python 3.11 environment and record baseline Ruff/pytest/model checks.
- [x] Diagnose and repair Stage 5A3 launcher, physics, calibration, navigation, camera, and validation.
- [x] Verify Stage 5A and Stage 5A3 in real Webots; repeat Stage 5A3 three times.
- [x] Integrate Stage 5 sources cleanly and verify deterministic world generation.
- [x] Implement and test truthful modular semantic perception.
- [x] Validate checkpoint and implement gated safe policy integration.
- [x] Complete the named implementation roadmap through the dedicated Stage 5C policy-motor runtime.
- [x] Preserve existing reproducible benchmark JSON/CSV artifacts and migrate the Stage 5B benchmark into the repository.
- [x] Complete Ruff, full pytest, CI/test markers, documentation, and paper alignment.
- [x] Commit and push stage-specific completion work.
- [x] Recover and externally back up the interrupted strong-policy working tree.
- [x] Preserve and hash-verify the accepted 120,144 expert transitions.
- [x] Replace copied in-memory sequences with a two-chunk LRU mmap cache and compact index.
- [x] Pass the 1,000-batch CUDA Behavior Cloning memory regression gate.
- [x] Pass Behavior Cloning held-out accuracy and per-action recall gates.
- [x] Implement and validate a policy-observation-only causal expert.
- [x] Pass causal expert solvability gates across 1,440 environment episodes.
- [x] Generate and hash-validate the separate causal correction dataset.
- [x] Compare causal-only, privileged-pretraining, and confidence-filtered modes.
- [x] Pass three-seed causal Behavior Cloning and CUDA memory gates.
- [x] Complete three genuine DAgger iterations.
- [ ] Train and accept RiskShield-HRMPPO v2 against all replacement gates.
- [x] Audit and decompose all 5,460 legacy constraint increments.
- [x] Implement Safety Contract v3 and preserve the legacy inactivity-biased diagnostic.
- [x] Implement typed vector-cost transitions and five independent recurrent cost critics.
- [x] Implement and regression-test the predictive event-aware shield.
- [x] Generate and validate 51,000 targeted safety DAgger corrections with
  contiguous recurrent context and KL task anchoring.
- [ ] Train and accept RiskShield-HRMPPO-Safe v3 against paired task and safety gates.
- [x] Complete CV audit, three-world Webots validation, benchmark-v2, ablations, paper, and Git acceptance.
- [x] Recover exact hierarchical-v4 starting state and freeze six baseline manifests.
- [x] Implement and test recurrent mission policy, causal planner, predictive local controller, and independent shield composition.
- [x] Derive 63,080 causal hierarchical decisions without modifying the 250,128-transition source.
- [x] Pass three-seed hierarchical imitation gates on held-out causal episodes.
- [ ] Complete three genuine hierarchical DAgger iterations (9,000 policy-visited corrections exist, but the H1 beta-zero mission gate failed and is not accepted).
- [ ] Train and accept constrained option-level HRMPPO-MPC v4.

## Safety evidence

- Remote branch: `final/pre-cleanup-safety-20260729`
- Safety commit: `d87ebdc368bd27fd4cca9841490d38ac78476e92`
- The remote branch preserves the legitimate pre-cleanup Stage 5B working state.

## Initial state notes

- Source branch: `stage-5b-live-perception-integration` at `650da2b`.
- Pre-cleanup work contained legitimate Stage 5B viewpoint repairs plus ignored caches, environments, checkpoints, datasets, and runtime evidence.
- The original main viewport was orthographic, line-only, and rotated; static tests did not detect the visual failure.
- The Webots executable accepts the world as a positional argument, and supports `--batch`, `--minimize`, `--no-rendering`, `--stdout`, and `--stderr`.
- The original verified motor source was the deterministic Stage 5A3 waypoint controller, and RL motor control was false.
