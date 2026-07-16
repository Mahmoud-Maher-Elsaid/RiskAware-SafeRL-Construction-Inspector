# Stage 3E-B Counterfactual Lagrangian Training

## Problem

The Stage 3E-A runtime shield produces zero executed safety cost, but the
unshielded policy remains unsafe. Optimizing only executed cost therefore
provides no learning signal for unsafe policy proposals that the shield
replaces.

## Counterfactual proposed-action cost

Before shield replacement, the training wrapper evaluates the action proposed
by the policy. Worker, restricted-zone, collision, and invalid-action
violations contribute to the counterfactual cost.

The constrained training reward is:

```text
constrained_reward = task_reward - lambda * proposed_action_cost
```

The environment still executes the shielded action. The policy therefore
receives a penalty for an unsafe proposal while the runtime system remains
protected.

## Adaptive multiplier

At the end of each rollout, the multiplier is updated from completed episode
costs:

```text
lambda = clip(
    lambda + learning_rate * (mean_episode_cost - cost_limit),
    0,
    maximum,
)
```

The multiplier is shared by all parallel environments.

## Model selection

Periodic validation and the primary final validation are unshielded during
Lagrangian training. The selected checkpoint must therefore satisfy intrinsic
policy safety rather than shielded runtime safety.

Additional shielded final evaluations measure deployment behavior separately.

## Stage 3E-B acceptance target

- unshielded feasible safety cost not greater than 5.0
- unshielded full safety cost not greater than 5.0
- zero collision cost
- zero invalid masked actions
- no inspect-action collapse
- lower runtime shield dependence
- maximum hazard recall subject to the safety constraint

## Calibration runs

Full curriculum experiments require at least 100 rollout updates. Shorter
diagnostic experiments must explicitly pass `--calibration-run`.

Calibration mode preserves the requested update count and curriculum stage
boundaries. It cannot be combined with `--smoke-test`, which always uses three
fixed rollout updates.
