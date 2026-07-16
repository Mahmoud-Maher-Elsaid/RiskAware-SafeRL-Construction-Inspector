# Stage 3D Maskable PPO Smoke Results

## Status

The MaskablePPO implementation passed the complete test suite and CUDA smoke
test.

## Smoke configuration

- Algorithm: MaskablePPO
- Seed: 42
- Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- CUDA runtime: 12.8
- Rollout updates: 3
- Environment timesteps: 3,072
- PPO optimization epochs: 30
- Parallel environments: 4
- Curriculum stages: easy, medium, full
- Safety shield: disabled

## Action-mask validation

| Metric | Result |
|---|---:|
| Invalid masked actions | 0 |
| Invalid masked action rate | 0.0000 |
| Mean valid action count | 3.2148 |
| Inspect availability frequency | 0.0371 |
| Final inspect action frequency | 0.0137 |
| Inspect collapse detected | False |

## Final rollout

| Metric | Result |
|---|---:|
| Hazard recall | 0.6714 |
| Coverage | 0.3264 |
| Success | 0.0000 |
| Mean step safety cost | 0.1758 |
| Collision cost | 0.0000 |
| Worker cost | 0.1377 |
| Restricted-zone cost | 0.0381 |

## Smoke validation

| Metric | Result |
|---|---:|
| Feasible validation recall | 0.2533 |
| Feasible validation safety cost | 99.0000 |
| Feasible validation success | 0.0000 |

## Interpretation

Task-valid action masking successfully removes the inspect-action collapse.
The policy cannot select an invalid inspection action or move outside the grid
or into an obstacle.

Worker-proximity and restricted-zone actions intentionally remain available.
The high safety cost therefore represents the unsolved semantic safety problem,
not a failure of the task mask.

The next controlled experiment uses the same curriculum, network, seed, and PPO
configuration as the Stage 3C baseline, replacing only PPO with MaskablePPO.