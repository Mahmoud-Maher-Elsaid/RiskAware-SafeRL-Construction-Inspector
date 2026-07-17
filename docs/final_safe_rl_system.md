# Final Safe RL System Decision

## Project objective

The system performs autonomous construction-site inspection while reducing
unsafe interaction with workers, restricted zones, obstacles, and other
semantic hazards.

The final Version 1 system prioritizes reliable runtime safety over unsupported
claims of intrinsic policy safety.

## Canonical deployment architecture

The production deployment baseline uses:

- MaskablePPO
- task-valid action masks
- the best Stage 3D policy checkpoint
- a deadlock-safe semantic runtime shield
- safe task-valid action projection
- emergency hold when no safe movement action exists
- least-unsafe task-valid fallback for unavoidable states
- shield-intervention diagnostics
- emergency-hold diagnostics
- unavoidable-violation diagnostics
- unshielded and shielded evaluation reports

## Canonical checkpoint

The deployment checkpoint is stored locally at:

~~~text
artifacts/runs/maskable_ppo_deadlock_safe_shield_seed42_u100/evaluations/best_model/best_model.zip
~~~

The checkpoint is generated as a training artifact and is not required to be
tracked directly by Git.

## Runtime control pipeline

Every action must pass through the following control pipeline:

~~~text
Observation
    -> MaskablePPO policy
    -> Task-valid action mask
    -> Proposed task-valid action
    -> Semantic safety shield
    -> Deadlock-safe fallback
    -> Executed robot action
~~~

The raw policy action must never be sent directly to the deployed robot.

## Action-mask responsibility

The task-valid action mask is responsible for preventing actions that are
structurally invalid, including:

- movement outside the grid
- movement into static obstacles
- invalid inspection actions
- other actions that are impossible in the current task state

The action mask does not independently guarantee semantic safety around
workers or restricted areas.

## Semantic-shield responsibility

The semantic safety shield evaluates the valid policy proposal before
execution.

The shield can:

- accept a safe policy proposal
- replace an unsafe proposal with a safe task-valid action
- select emergency hold when safe movement is unavailable
- select the least-unsafe task-valid action when no safe action exists
- record every intervention and fallback decision

The shield is a best-effort safety controller. It must not be described as a
mathematical guarantee for every possible real-world state.

## Validated findings

### Unmasked PPO

The unmasked PPO baseline developed inspect-action collapse and failed to
produce useful inspection behavior.

### MaskablePPO

Task-valid action masking eliminated invalid masked actions and prevented
inspect-action collapse.

MaskablePPO improved inspection recall, but the unshielded policy did not
satisfy the semantic safety constraint.

### Semantic safety shield

The semantic shield produced zero mean safety cost on the final shielded
feasible and full validation evaluations used in the project.

This is empirical validation on the evaluated scenario sets. It is not proof
of zero risk in every possible environment.

### Scalar Lagrangian training

The scalar reward-penalty method was rejected because:

- the Lagrange multiplier increased continuously
- the reward scale became strongly negative
- reward-value loss became unstable
- policy updates became very small
- the configured safety-cost constraint was not satisfied

### Dual-critic PPO-Lagrangian

The dual-critic implementation separated reward and safety-cost learning and
remained more numerically stable than the scalar reward-penalty baseline.

However, the calibrated dual-critic policy:

- did not satisfy the configured cost limit
- did not outperform the Stage 3D baseline on intrinsic safety
- did not justify replacing the canonical deployment architecture

The dual-critic implementation remains on its experimental branch for research
and ablation studies. It is not part of the Version 1 production baseline.

## Canonical Version 1 deployment rule

Version 1 deployment must use:

~~~text
Stage 3D best MaskablePPO policy
+ task-valid action masking
+ deadlock-safe semantic runtime shield
+ intervention and fallback monitoring
~~~

Deployment without the semantic shield is unsupported.

## Operational safety requirements

A deployment integration must:

1. load the validated MaskablePPO checkpoint
2. compute a task-valid action mask for every observation
3. request a masked policy action
4. pass the proposal through the semantic shield
5. execute only the action returned by the shield
6. record shield interventions
7. record emergency holds
8. record unavoidable safety violations
9. stop operation when observations or safety state are invalid
10. keep a human operator able to override or stop the robot

## Evaluation requirements

Every release evaluation must report shielded and unshielded results
separately.

The primary fields are:

- hazard recall
- coverage
- success rate
- total safety cost
- worker safety cost
- restricted-zone safety cost
- collision cost
- invalid-action rate
- inspect-action frequency
- shield-intervention rate
- emergency-hold rate
- least-unsafe fallback rate

Shielded zero cost must be described as enforced runtime safety on the
evaluated scenarios, not intrinsic learned safety.

## Research status

Intrinsic policy safety remains an open research objective for this project.

Future research may investigate:

- longer dual-critic training
- stronger cost-value estimation
- cost-aware curriculum design
- constrained-policy initialization
- risk-sensitive world models
- distributional cost critics
- offline safe-RL pretraining
- formal safety verification
- sim-to-real uncertainty estimation

These research directions are separate from the validated Version 1
deployment baseline.

## Final decision

The shielded MaskablePPO system is the canonical Version 1 deployment
baseline.

The project is considered operationally complete for the current grid-world
Safe RL stage when:

- all automated tests pass
- the canonical checkpoint is available
- task-valid masks remain active
- the semantic shield remains mandatory
- intervention and fallback diagnostics remain enabled
- shielded and unshielded results remain clearly separated

No claim of absolute or universal safety is made.
