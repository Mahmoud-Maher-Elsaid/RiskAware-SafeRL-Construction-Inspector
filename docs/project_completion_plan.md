# Project Completion Plan

Last updated: 2026-07-29

## Live checklist

| Field | Current value |
|---|---|
| Current task | Strong-policy upgrade: RiskShield-HRMPPO v2 |
| Current state | Causal imitation and three genuine policy-visited-state DAgger iterations passed |
| Evidence collected | Preserved original hashes; causal manifests; expert evaluation; mixed-mode comparison; three-seed BC metrics; CUDA memory gate; DAgger corrections, disagreements, checkpoints, and held-out rollouts |
| Files changed | Causal imitation stack, recurrent attention policy, memory-safe trainer, DAgger collector/retrainer, tests, manifests, reports, and docs |
| Tests run | Repository Ruff/full pytest; causal expert/generator/mixer/cache tests; 1,440-episode expert evaluation; 1,000-batch CUDA memory gate; DAgger CUDA smoke and full 36,000-correction run |
| Runtime result | Causal BC mean 91.07%; DAgger best held-out success 85.56%; final beta-zero success 76.67%, recall/coverage 95.94%, collision rate 0, invalid actions 0 |
| Remaining work | HRMPPO v2; CV/Webots/benchmark/paper/final acceptance |
| Blockers | None at the HRMPPO algorithm-smoke stage |

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
- [ ] Pass Behavior Cloning held-out accuracy and per-action recall gates.
- [x] Implement and validate a policy-observation-only causal expert.
- [x] Pass causal expert solvability gates across 1,440 environment episodes.
- [x] Generate and hash-validate the separate causal correction dataset.
- [x] Compare causal-only, privileged-pretraining, and confidence-filtered modes.
- [x] Pass three-seed causal Behavior Cloning and CUDA memory gates.
- [x] Complete three genuine DAgger iterations.
- [ ] Train and accept RiskShield-HRMPPO v2 against all replacement gates.
- [ ] Complete CV audit, three-world Webots validation, benchmark-v2, ablations, paper, and Git acceptance.

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
