# Stage 3C Safe Curriculum PPO Results

## Status

The safe curriculum pipeline completed successfully, but the first unmasked
PPO baseline failed to learn a useful inspection policy.

## Training configuration

- Seed: 42
- Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- CUDA runtime: 12.8
- Rollout updates: 100
- Environment timesteps: 102,400
- PPO optimization epochs: 1,000
- Parallel environments: 4
- Rollout steps per environment: 256
- Batch size: 256
- Curriculum schedule:
  - Easy: 25 updates
  - Easy and medium: 25 updates
  - Full safe-feasible curriculum: 50 updates

## Curriculum dataset

- Safe-feasible training scenarios: 751
- Safe-feasible validation scenarios: 153
- Complete validation scenarios: 200
- Easy scenarios: 251
- Medium scenarios: 250
- Hard scenarios: 250

## Action-collapse diagnosis

The policy increasingly selected the inspect action instead of navigating.

| Metric | First rollout | Final rollout |
|---|---:|---:|
| Inspect action frequency | 0.2207 | 0.8457 |
| Hazard recall | 0.6167 | 0.5357 |
| Coverage | 0.3576 | 0.1667 |
| Episode success | 0.0000 | 0.0000 |
| Mean step safety cost | 0.3340 | 0.2139 |

Several intermediate rollouts exceeded an inspect-action frequency of 0.95.

## Deterministic validation

| Evaluation | Scenarios | Reward | Safety cost | Hazard recall | Coverage | Success |
|---|---:|---:|---:|---:|---:|---:|
| Best periodic model | 50 | -33.5200 | 15.0000 | 0.0800 | 0.0069 | 0.0000 |
| Final feasible validation | 153 | -36.1069 | 18.1895 | 0.0871 | 0.0093 | 0.0000 |
| Final full validation | 200 | -39.7243 | 24.9600 | 0.0819 | 0.0093 | 0.0000 |

## Interpretation

The run validates the curriculum, checkpointing, evaluation, diagnostics,
reporting, and CUDA training pipeline.

The learned policy is not a successful task policy. The weak invalid-inspection
penalty allows PPO to converge toward repeated inspection rather than learning
navigation and viewpoint selection.

This run is retained as the unmasked PPO failure baseline.

## Next stage

Stage 3D will introduce task-valid action masks with MaskablePPO while keeping
worker and restricted-zone safety constraints separate for controlled
ablation experiments.