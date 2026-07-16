# Stage 3D Maskable PPO

## Purpose

Stage 3C demonstrated inspect-action collapse in the unmasked PPO baseline.
Stage 3D introduces task-valid action masking while preserving semantic safety
risks for later Safe RL and shield ablations.

## Mask semantics

The mask removes only task-invalid actions:

- movement outside the grid
- movement into an obstacle
- inspection when no uninspected hazard is visible from the current viewpoint

Movement into worker-proximity cells or restricted zones remains available.
Those actions are safety-relevant rather than task-invalid and must remain
visible to the policy, safety constraints, and shield.

A trapped-state fallback keeps the inspect action available when every movement
is blocked and no hazard is inspectable. This guarantees that the mask never
contains only false values.

## Training

The curriculum training script supports:

```text
--algorithm ppo
--algorithm maskable_ppo
```

Maskable PPO uses the environment-provided `action_masks()` method during
training and deterministic validation.

## Diagnostics

Rollout diagnostics include:

- action frequencies
- inspect-collapse streak and warning state
- mean valid action count
- inspect availability frequency
- invalid masked action count and rate
- safety-cost components
- terminal recall, coverage, and success

The expected invalid masked action count is zero.

## Controlled ablation

The Stage 3C unmasked PPO run is retained as the negative baseline. The masked
run uses the same curriculum, network, optimizer, rollout size, PPO epochs, and
seed so that action masking is the primary experimental difference.
