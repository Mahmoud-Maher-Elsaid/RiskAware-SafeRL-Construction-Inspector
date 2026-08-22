# Repository Working Rules

## Scope and language

- Work directly in this repository and continue through implementation, tests, and locally available runtime verification; inspection alone is never completion.
- Use English only in source, comments, docstrings, documentation, configuration, filenames, commands, tests, logs, errors, and Git-related text.
- Normal commits and non-force pushes are allowed after validation. Normal
  fast-forward or merge integration into `main` is allowed after validation;
  pull requests are not required. Do not force-push, rewrite published history,
  destructively replace `main`, or modify/delete tag `v1.0.0-safe-shielded`.
- Preserve all current Stage 5A/5A3 work, including untracked files, runtime evidence, and repair backups. Never use `git clean`, `git reset --hard`, or discard user work.
- Back up material work externally with a hashed manifest before risky recovery or repair.

## Engineering workflow

- For important changes: inspect, back up, edit source directly, format, statically validate, run targeted tests, run broader tests, run applicable real runtime validation, inspect evidence, then document verified facts.
- Diagnose root causes from complete failures. Avoid fragile exact-text patchers and giant one-use repair scripts.
- Keep generated Webots worlds deterministic and retain one authoritative builder per generated world.
- Use repository-relative discovery; keep machine-specific external paths configurable.
- Do not weaken physical stability or safety thresholds to hide failures. Normal Stage 5A3 roll and pitch targets are below 10 degrees, with a real fail-fast guard no higher than 18 degrees.
- Never fabricate runtime, experiment, model, perception, policy-control, or safety results.
- Do not add TensorFlow. The project targets Python 3.11 and PyTorch with the CUDA 12.8 build when GPU packages are installed.

## Safety and control claims

- Preserve the mandatory policy execution order: observation -> MaskablePPO -> task-valid action mask -> policy proposal -> semantic safety shield -> deadlock-safe fallback -> executed action.
- Policy motor control must remain gated off except in a dedicated, bounded, verified runtime.
- Keep scripted Stage 4 and deterministic Stage 5A3 controllers as distinct baselines. Stage 5A3 is deterministic closed-loop navigation, not RL control or CV inference.
- Report perception backends truthfully: real only with a validated configured artifact, deterministic mock for tests, or disabled when unavailable.
- Avoid absolute safety claims. Prefer language such as "validated in simulation" and "observed in the tested configurations"; never claim real-world safety was established.

## Verification and artifacts

- CI must not require a GPU, interactive Webots GUI, private data, or private models. Mark local Webots, GPU, and long-running tests appropriately.
- Derive acceptance values from runtime evidence using an independent validator; never hardcode success results.
- Store machine-readable experiment results as JSON and CSV and generate plots reproducibly; do not hardcode paper results.
- Keep large logs/runtime evidence ignored unless intentionally curating a small artifact. Never ignore required source, tests, configs, docs, builders, launchers, validators, or worlds.
- Maintain `docs/project_completion_plan.md` throughout substantial project-completion work.
