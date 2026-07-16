# Stage 3D Maskable PPO Results

## Status

Stage 3D successfully eliminated inspect-action collapse and enabled meaningful
navigation and inspection learning.

The learned policy does not yet satisfy the semantic safety constraint.

## Training configuration

- Algorithm: MaskablePPO
- Seed: 42
- Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- CUDA runtime: 12.8
- Rollout updates: 100
- Environment timesteps: 102,400
- PPO optimization epochs: 1,000
- Parallel environments: 4
- Safety shield: disabled
- Task-valid action masking: enabled
- Safety-cost limit: 5.0

## Mask validation

Across training:

- Invalid masked actions remained zero.
- Collision cost remained zero.
- Inspect collapse was not detected.
- The policy learned navigation and valid viewpoint inspection.

## Best versus final checkpoint

| Model | Evaluation | Scenarios | Reward | Safety cost | Hazard recall | Coverage | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| Best | Feasible | 153 | -11.2267 | 26.3922 | 0.3499 | 0.0996 | 0.0065 |
| Best | Full | 200 | -11.3751 | 26.1500 | 0.3322 | 0.0979 | 0.0050 |
| Final | Feasible | 153 | -11.9761 | 30.4314 | 0.4282 | 0.1176 | 0.0588 |
| Final | Full | 200 | -10.8137 | 28.2350 | 0.4189 | 0.1176 | 0.0650 |

## Checkpoint decision

The final model is retained as the primary Stage 3D task-learning baseline.

It provides higher hazard recall, coverage, and success than the checkpoint
selected by the safety-aware periodic metric.

The periodic best model is retained as a lower-cost reference checkpoint, but
it is not considered safe because its mean safety cost remains substantially
above the required limit.

## Interpretation

Task-valid action masking solves the invalid-action and inspect-collapse
problems. It does not prevent movement near workers or through restricted
zones because those actions are semantic safety decisions rather than invalid
task actions.

Neither checkpoint satisfies the safety-cost limit of 5.0.

Stage 3E will add semantic safety control while keeping the task-valid masks
from Stage 3D.