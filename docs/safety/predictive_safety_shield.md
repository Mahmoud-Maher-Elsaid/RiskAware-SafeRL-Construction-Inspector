# Predictive Safety Shield

The predictive shield runs after a policy proposes an action and before the
environment or Webots motor adapter executes it. It simulates a deterministic
trajectory for a configurable horizon and predicts collision, worker/dynamic
hazard near miss, restricted-zone entry, PPE-risk proximity, and budget excess.

Each decision records:

- proposed action and predicted trajectory;
- predicted cost and violation types;
- accept, replace, or stop decision;
- replacement and final action;
- confidence, emergency-stop state, and computation latency.

Candidate actions use deterministic safety-cost and action-deviation ordering.
If no candidate remains within budget, the shield emits the hold/inspect action
as an emergency stop. `PredictiveShieldWrapper` passes only the final action to
the environment. The existing production Stage 5C shield remains unchanged;
this research shield is regression-isolated until a later production upgrade is
separately validated.

The shield benchmark compares disabled, one-step, and three-step modes over
three environments and ten seeds. The reported unnecessary-intervention metric
is an explicitly named immediate-risk proxy: an intervention is counted when
the proposed action has no one-step violation even though the predictive model
forecasts a later violation. This is conservative and does not imply the
intervention was truly unnecessary.
