# Problem formulation

The task is modeled as a constrained partially observable Markov decision process.

The policy maximizes discounted inspection reward while respecting expected safety-cost budgets.

## Reward objectives

- discover and inspect hazards
- explore new cells
- finish the inspection efficiently

## Safety constraints

- collisions
- entering restricted zones
- near-misses with workers

## Planned research contribution

1. An open construction-inspection Safe RL benchmark.
2. A semantic risk representation.
3. A constrained RL agent.
4. An action shield.
5. Robustness tests under perception noise and unseen layouts.