# Full Research Benchmark

The final matrix pairs five algorithms across three site sizes, three hazard
densities, three perception false-negative levels, and ten evaluation seeds.
It contains 1,350 primary episodes. Evaluation seeds are shared across
algorithms within each condition; policies are not retrained per cell.

Planner episodes run in a bounded thread pool. PPO, SAC, and RiskShield-PPO
episodes run serially because they share the CUDA device. Every run has a
canonical configuration hash, checkpoint hash where applicable, immutable run
identifier, atomic JSON cache record, timestamps, terminal state, failure
field, and complete safety and task metrics.

The entry point validates dependencies, CUDA, checkpoint files, the 1,350-entry
manifest, cache completeness, CSV/Parquet parity, ablations, figures, and
tables:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "F:\AI\My_Project\RiskAware-SafeRL-Construction-Inspector\scripts\run_full_research_benchmark.ps1"
```

Statistical comparisons use a Friedman repeated-measures test because the
bounded, zero-inflated metrics do not justify a normality assumption. The
planned RiskShield-PPO versus PPO comparison uses paired Wilcoxon signed ranks,
Holm correction across six metrics, and a standardized paired mean difference.
No missing run is imputed.

The benchmark's main negative result is important: PPO, SAC, and
RiskShield-PPO achieved zero complete-mission success under this evaluation
budget. RiskShield-PPO improved mean safety cost and coverage relative to PPO,
but did not solve the task. The planner baselines achieved much higher success,
recall, and coverage.
