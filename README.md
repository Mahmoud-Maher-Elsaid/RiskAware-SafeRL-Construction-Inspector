# RiskAware-SafeRL Construction Inspector

A research-oriented reinforcement learning project for autonomous inspection of dynamic construction sites.

## Research question

Can semantic risk observations, constrained objectives, and action shielding improve hazard discovery while reducing collisions, worker near-misses, and restricted-zone violations?

## Current stage

- [x] Reproducible Python project
- [x] Gymnasium construction-inspection benchmark
- [x] Explicit safety cost signals
- [x] Random-policy baseline
- [x] PPO baseline
- [x] Rule-based action shield
- [x] Unit tests and GitHub Actions
- [ ] PPO-Lagrangian
- [ ] Expert A* / frontier baseline
- [ ] Partial-observation recurrent agent
- [ ] Webots integration
- [ ] Semantic perception module
- [ ] Domain randomization
- [ ] Full benchmark and research paper

## Environment

The agent operates in a partially observable grid construction site.

Actions:

- `0`: move up
- `1`: move down
- `2`: move left
- `3`: move right
- `4`: inspect the current cell

Observation channels:

1. obstacles
2. hazards
3. workers
4. restricted zones
5. visited cells
6. agent position
7. semantic risk map

Safety costs:

- collision cost
- worker near-miss cost
- restricted-zone cost

## Setup

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe scripts\evaluate_random.py --episodes 20
.\.venv\Scripts\python.exe scripts\train_ppo.py --timesteps 100000 --device cuda
.\.venv\Scripts\python.exe scripts\train_ppo.py --timesteps 100000 --device cuda --shield
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## TensorBoard

```powershell
.\.venv\Scripts\tensorboard.exe --logdir artifacts\tensorboard
```

## Repository policy

Generated datasets, model checkpoints, and experiment logs are ignored by Git. Only code, configs, documentation, and small reproducibility artifacts should be committed.