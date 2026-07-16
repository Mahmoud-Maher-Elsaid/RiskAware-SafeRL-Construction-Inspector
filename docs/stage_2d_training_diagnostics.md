# Stage 2D: Training Diagnostics and Validation

## Purpose

This stage prevents long reinforcement-learning runs from failing silently.

## Added capabilities

- rollout-level action-frequency logging
- action-collapse detection
- safety-cost logging
- deterministic validation on fixed scenarios
- best-model selection
- periodic checkpoints
- resume-from-checkpoint support
- final validation evaluation
- mean, standard deviation, and 95% confidence intervals
- run metadata with Git commit, seed, device, and hyperparameters

## Default full run

- 100 rollout updates
- 4 parallel environments
- 256 steps per environment and rollout
- 102,400 environment timesteps
- 10 PPO optimization epochs per rollout
- 1,000 total optimization epochs
- validation every 10 rollout updates
- checkpoint every 10 rollout updates

## Model selection

The default selection metric is `safe_hazard_recall`.

A model is considered feasible when its mean validation safety cost is less than
or equal to the configured safety-cost limit. Among feasible models, hazard
recall is maximized. Infeasible models receive a negative selection score.

## Smoke tests

Smoke tests use two rollout updates and five validation scenarios. Their only
purpose is implementation validation. They must not be reported as research
results.
