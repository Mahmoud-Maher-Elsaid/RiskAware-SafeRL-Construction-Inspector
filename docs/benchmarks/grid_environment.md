# Final Grid Construction-Site Benchmark

`ResearchConstructionEnv` is a deterministic Gymnasium environment for rapid
algorithm and uncertainty studies. It extends the historical grid prototype
without changing the observation schema used by existing trained checkpoints.

## Actions

| Index | Action | Effect |
|---:|---|---|
| 0 | north | Move one cell north and face north |
| 1 | south | Move one cell south and face south |
| 2 | west | Move one cell west and face west |
| 3 | east | Move one cell east and face east |
| 4 | inspect | Inspect hazards within the configured radius |

## Observation

The observation is a typed `gymnasium.spaces.Dict`.

- `map`: ten `float32` semantic channels in `[0, 1]`: obstacles, uninspected
  hazards, workers, restricted zones, visited cells, robot position, fused risk,
  dynamic hazards, PPE-risk zones, and visibility.
- `state`: nine normalized values: row, column, orientation, elapsed fraction,
  hazard recall, coverage, accumulated cost, dynamic density, and perception
  false-negative rate.

Ground truth remains in environment state. Perception noise affects only
observable semantic channels, never labels or evaluation truth.

## Reward and cost

Reward and safety cost are separate. Reward encourages new coverage and verified
inspection while penalizing wasted actions and unsafe motion. Safety cost records
collisions, restricted-zone entry, worker/dynamic-hazard near misses, and PPE-risk
exposure. The `info` dictionary and `EpisodeTelemetry` expose cumulative metrics.

## Reproducibility

All layout sampling, perception dropout, and dynamic movement use Gymnasium's
seeded generator. Run:

```powershell
.venv\Scripts\python.exe scripts\validate_grid_environment.py
```

The validator runs Gymnasium's checker, deterministic reset/step comparisons,
space checks, action-mask checks, and JSON serialization for all three configs.
