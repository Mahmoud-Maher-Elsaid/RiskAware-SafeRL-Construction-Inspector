# Final Demonstration

For the production integration path, run
`scripts/run_complete_autonomous_inspection.ps1`. For the experimental v4
visible demonstration, use `scripts/run_real_v4_webots_demo.ps1` with its
default `CameraMode=overview`. The launchers validate the configured runtime
and preserve the production-v1 versus experimental-v4 boundary.

The v4 Webots main viewport is a static, site-centered overview. The robot
moves autonomously inside the construction scene; no external camera viewer,
follow camera, or manual overlay is required.
