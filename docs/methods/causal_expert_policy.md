# Observation-Consistent Causal Expert

The causal expert is a deterministic stateful planner whose complete input is the
same `map`, `state`, and `action_mask` observation exposed to the recurrent
policy. It never receives an environment object.

Its persistent memory is derived only from prior observations:

- observed and traversable cells;
- observed obstacles and restricted zones;
- detected hazard locations;
- inspected-target markers;
- visited cells;
- currently observed workers, dynamic hazards, and PPE risk;
- recent robot positions for loop avoidance.

The expert first inspects a detected, uninspected target when the observation
mask permits inspection. Otherwise, it runs risk-weighted A* over discovered
traversable cells to a viewpoint near the nearest reachable remembered target.
If no target is reachable, it plans to an unvisited discovered frontier and
finally expands into an adjacent unknown cell using only the action mask.

An observation-only shield scores candidate next cells from detected restricted
zones, workers, dynamic hazards, PPE risk, and remembered hazards. It replaces
an invalid or excessive-risk proposal with the lowest-cost valid observed
action. No future trajectory or hidden simulator state is consulted.

The following information is deliberately unavailable: undiscovered hazards,
hidden obstacles beyond the action mask, full simulator maps, oracle target
coordinates, future worker motion, and private environment collections.
