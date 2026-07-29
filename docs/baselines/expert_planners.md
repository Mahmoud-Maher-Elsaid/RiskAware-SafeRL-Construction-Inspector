# Expert Planning Baselines

The final grid benchmark provides three deterministic, non-RL baselines.

- **Risk-aware A\*** is a privileged expert with ground-truth geometry. It
  minimizes path length plus configurable costs for restricted zones, PPE-risk
  cells, workers, and moving hazards.
- **Frontier exploration** selects the nearest safe unvisited boundary adjacent
  to explored space and inspects hazards when they enter inspection range.
- **Nearest-risk revisit** prioritizes the closest known uninspected hazard,
  falling back to deterministic exploration when no known risk is reachable.

All planners replan after every action. This makes dynamic-obstacle recovery
explicit and avoids executing a stale route. Ties use row-major positions and
fixed action order. An unreachable target results in a documented hold/inspect
action rather than an invalid move.

Run the real 90-episode Stage 2 evaluation:

```powershell
.venv\Scripts\python.exe scripts\evaluate_planner_baselines.py
```

The output includes hazard recall, inspection coverage, path length, collisions,
near misses, restricted-zone violations, time to inspect, energy proxy, success
rate, and planning latency. These are expert planning baselines and are never
described as reinforcement learning.
