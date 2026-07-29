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

## Stage 4C1B live policy sidecar

Stage 4C1B runs the trained MaskablePPO checkpoint against observations
reconstructed from live Webots telemetry.

The policy engine runs in a separate sidecar process. The Webots robot
controller remains unchanged and continues to use the scripted validation
motion sequence.

The runtime path is:

~~~text
live Webots sensors
    -> existing observation bridge
    -> live telemetry log
    -> isolated policy sidecar
    -> reconstructed observation verification
    -> MaskablePPO proposal
    -> structured proposal log
~~~

The prohibited path remains disconnected:

~~~text
MaskablePPO proposal -X-> wheel motors
~~~

Each live proposal records that:

- the semantic map was verified against live telemetry
- the state vector was verified against live telemetry
- the action mask was verified against live telemetry
- the proposal respected the action mask
- no motor command channel was available
- the proposal was not applied
- the robot controller was not modified

Stage 4C1B remains a dry run and does not provide an absolute safety
guarantee.
