# Pre-change audit

The visible world used a fixed external `Viewpoint` and the controller selected options and targets with masked argmax only. The worker trajectory had a fixed phase, and the runtime did not expose pose, route, or replanning traces. The correction adds a Webots `Mounted Shot` follow target, seeded policy sampling, scenario-controlled worker phase, and causal pose/route/replanning telemetry.
