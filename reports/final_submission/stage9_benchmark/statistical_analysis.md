# Stage 9 Statistical Analysis

The analysis uses the complete paired benchmark matrix. No failed or missing run is imputed.

## Method

- Design: Repeated measures by site size, hazard density, noise, and seed.
- Omnibus Test: Friedman rank test because bounded and zero-inflated metrics need not be normal.
- Planned Pair: Paired Wilcoxon signed-rank: RiskShield-PPO versus PPO.
- Multiple Comparison Correction: Holm family-wise correction across six planned metrics.
- Effect Size: Standardized paired mean difference; zero when paired differences have zero variance.
- Confidence Intervals: Two-sided 95% Student-t intervals over ten evaluation seeds in each aggregate cell.
- Missing Data Policy: No imputation; Stage 9 fails if any primary run is missing.

## Results

| Metric | Friedman p | Wilcoxon Holm p | Paired effect size | n |
|---|---:|---:|---:|---:|
| hazard_recall | 2.48019e-206 | 2.94695e-16 | 0.5533 | 270 |
| inspection_coverage | 3.38438e-210 | 3.12772e-28 | 0.7716 | 270 |
| collision_rate | 2.60519e-168 | 1.54884e-05 | -0.2823 | 270 |
| constraint_violations | 2.21136e-104 | 8.42643e-14 | -0.3444 | 270 |
| safety_cost | 2.53736e-133 | 8.30985e-18 | -0.4183 | 270 |
| success_rate | 1.99103e-227 | 1 | 0.0000 | 270 |
