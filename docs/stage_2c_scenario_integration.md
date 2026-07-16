# Stage 2C: Scenario Dataset Integration

## Objective

Connect the versioned construction scenario dataset to training and evaluation.

## Training protocol

The default full PPO run uses:

- 100 rollout updates
- 4 parallel environments
- 256 steps per environment and update
- 102,400 total environment timesteps
- 10 optimization epochs per rollout update
- 1,000 total optimization epochs across the run

A smoke test is available only for implementation validation. Smoke-test results
must not be reported as research results.

## Split usage

- `train`: policy optimization
- `validation`: model selection and development evaluation
- `test_seen`: final in-distribution evaluation
- `test_unseen`: out-of-distribution density evaluation
- `stress`: high-density robustness evaluation

The `test_seen` label means an unseen layout sampled from the training
distribution. It does not contain layouts used during training.

## Reproducibility

Each split is loaded from a versioned JSONL file and checked against the SHA256
digest stored in `manifest.json`. JSON and JSONL files are forced to LF line
endings to preserve stable hashes across operating systems.