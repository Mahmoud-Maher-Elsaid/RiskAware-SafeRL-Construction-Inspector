# Stage 3E-C Dual-Critic Maskable PPO-Lagrangian

## Motivation

The scalar reward-penalty baselines changed the reward scale whenever the
Lagrange multiplier changed. This destabilized the reward value function and
did not satisfy the episodic safety-cost constraint.

Stage 3E-C keeps task reward and safety cost in separate learning streams.

## Architecture

- Maskable PPO actor
- reward value function from the Maskable PPO policy
- independent semantic-map cost value function
- reward generalized advantage estimation
- cost generalized advantage estimation
- conservative clipped cost surrogate
- normalized reward and cost advantages
- bounded exponential-moving-average dual controller
- no semantic action replacement during training
- unshielded primary evaluation
- shielded deployment evaluation

## Policy objective

The reward surrogate uses the standard clipped PPO minimum. The cost surrogate
uses the conservative clipped maximum. The policy maximizes reward and
minimizes cost through:

```text
(reward_surrogate - lambda * cost_surrogate) / (1 + lambda)
```

Division by `1 + lambda` keeps the policy-gradient scale bounded as the dual
variable changes.

## Cost critic

The cost critic receives the same semantic map and robot state as the policy.
It is optimized independently with Smooth L1 loss and gradient clipping. Its
parameters are not included in the reward-policy optimizer.

## Dual controller

The dual controller:

- observes completed undiscounted episode costs
- smooths costs with an exponential moving average
- waits for a configurable critic warmup period
- updates from normalized constraint violation
- clips the multiplier to a configured maximum

## Acceptance criteria

The method is successful only when unshielded evaluation satisfies:

- mean feasible safety cost at most 5
- mean full safety cost at most 5
- collision cost equal to 0
- invalid masked actions equal to 0
- no inspect-action collapse
- the highest feasible hazard recall available under the constraint

Shielded evaluation is reported separately and cannot be used as evidence of
intrinsic policy safety.
