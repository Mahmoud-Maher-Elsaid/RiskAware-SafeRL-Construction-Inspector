# Stage 3A: Expert Baselines and Reward Audit

## Purpose

Determine whether the environment is solvable and whether the reward function
encourages useful inspection behavior before launching long PPO runs.

## Baselines

- `inspect_only`: stationary policy that repeatedly inspects
- `random`: deterministic seeded random policy
- `greedy_astar`: full-information nearest-hazard A* planner
- `safe_greedy_astar`: A* planner that excludes restricted cells and worker
  near-miss cells

The A* planners are oracle upper-bound baselines because they use the full
scenario layout.

## Reward audit

Each transition is decomposed into:

- step penalty
- new-cell reward
- hazard discovery reward
- completion reward
- collision penalty
- worker near-miss penalty
- restricted-zone penalty
- invalid-inspection penalty

The wrapper verifies that the reconstructed components exactly match the
environment reward.

## Expert plan dataset

Expert plans are stored as compact action sequences linked to versioned
scenario identifiers. Full observations do not need to be duplicated because
the scenarios and environment are deterministic and replayable.