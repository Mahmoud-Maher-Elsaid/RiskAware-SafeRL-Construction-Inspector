# Experiment protocol

Every reported result must include:

- at least 10 random seeds for final experiments
- mean and 95% confidence interval
- environment version and configuration
- training timesteps
- reward return
- hazard recall
- inspection coverage
- collision rate
- worker near-miss rate
- restricted-zone violations
- success rate
- wall-clock training time

Initial baselines:

1. Random policy
2. Frontier/A* expert
3. PPO
4. Shielded PPO
5. PPO-Lagrangian
6. Proposed shielded constrained agent