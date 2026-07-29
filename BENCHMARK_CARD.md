# Benchmark Card

The final matrix contains:

- 3 environment sizes;
- 3 hazard densities;
- 3 perception false-negative levels;
- 5 algorithms;
- 10 paired evaluation seeds;
- 1,350 primary episodes and 180 ablation episodes.

Each run has a unique immutable ID, canonical configuration hash, checkpoint
hash where applicable, atomic cache record, timestamps, failure field, and task
and safety metrics. Missing data are not imputed. Friedman and paired Wilcoxon
tests use Holm correction.

The benchmark is a grid simulation. Webots integration is validated separately
and must not be conflated with the 1,350 grid episodes.
