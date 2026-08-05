# Forensic v4 Webots correction findings

The prior experimental controller used a synthetic `observation(step)` implementation, populated fixed semantic-map cells, passed an all-ones option mask, supplied zero structured state, emitted only three decisions, and wrote success assertions as constants. Those behaviors were startup evidence rather than experimental integration.

The corrected controller now reads the Webots camera and GPS at each decision, calls the configured live perception backend, projects only current detections into an observed local map, constructs the 32-element training structured state, calls `causal_option_mask`, updates recurrent state after every policy decision, records persistent targets, executes the causal planner/controller/shield chain, and writes per-decision telemetry.

The historical three-step outputs under `webots_v4_final` are preserved and are superseded evidence. They are not used by final acceptance.
