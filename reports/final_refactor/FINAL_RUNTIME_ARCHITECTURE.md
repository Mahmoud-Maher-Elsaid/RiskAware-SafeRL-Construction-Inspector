# Final runtime architecture

The bounded and until-closed controllers share the same robot navigation loop;
only termination is selected by `RISK_AWARE_EXPERIMENTAL_RUN_MODE`.

- Sensors: `riskaware_saferrl.webots.sensors.clearance_meters`, with the visible
  world lookup response defined directly in metres.
- Primitive contract: `MotionPrimitive` and `demo_primitive_to_wheels` in
  `src/riskaware_saferrl/webots/motion_primitives.py`.
- Planner: `RiskShieldHierarchicalSystem` and `CausalRiskAwarePlanner`, with
  persistent `_last_path` reuse unless a causal replan is requested.
- Recovery: robot-controller recovery episode state and translational escape,
  still routed through planner, shield, envelope, and actuator clamp.
- Safety: shield decision, clearance envelope, emergency backstop, then the
  final ±1.35 command clamp immediately before Webots motors.
- Camera: Supervisor-only `HUMAN_VIEWPOINT` composition; Mounted Shot is not
  configured in first-person mode.

The fixture worlds under this report are validation-only and are not selected
by the production launcher.
