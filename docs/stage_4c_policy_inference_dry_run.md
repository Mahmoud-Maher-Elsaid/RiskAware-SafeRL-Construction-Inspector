# Stage 4C Policy Inference Dry Run

## Goal

Stage 4C validates the trained MaskablePPO checkpoint against observations
created by the Webots bridge.

Policy actions are proposals only. They are not motor commands.

## Safety boundary

The Stage 4C execution boundary is:

~~~text
observation
    -> task-valid action mask
    -> MaskablePPO proposal
    -> proposal validation
    -> structured dry-run log
~~~

The following connection is prohibited in this stage:

~~~text
policy proposal -> wheel motors
~~~

## Proposal contract

Every proposal records:

- sample index
- proposed grid action
- action name
- deterministic or stochastic inference mode
- complete task-valid action mask
- number of valid actions
- mask compliance
- motor connection status

A proposal is rejected when:

- the observation shape is incompatible
- the observation contains non-finite values
- the observation is outside the normalized range
- the mask has an invalid shape
- the mask contains no valid action
- the checkpoint returns a masked-out action
- the checkpoint returns an invalid action index

## Current scope

Stage 4C1A provides the reusable policy proposal engine and validates it
against the real checkpoint.

Live Webots telemetry integration is completed in the following Stage 4C
runtime step.

This stage does not provide an absolute safety guarantee.
