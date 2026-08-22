# Runtime execution graph

Authoritative user command: `scripts/run_real_v4_webots_demo.ps1 -World site_dynamic -RunMode until_closed -CameraMode first_person -ScenarioSeed 42 -PolicyMode stochastic -PolicyTemperature 0.70`.

1. PowerShell launcher generates `webots/worlds/site_dynamic_v4_visible_demo.wbt`, injects experimental controllers and obstacle sensors, sets process-local environment, selects a Webots port, and launches `webots.exe --batch --mode=realtime --stdout --stderr --port=<free-port> <world>`.
2. Webots starts `hierarchical_experimental_robot`, `hierarchical_experimental_supervisor`, and `final_dynamic_worker`.
3. Robot resolves GPS, compass, camera, wheel motors, and seven obstacle sensors. Each loop reads camera/GPS/compass/sensors, runs live CV, builds the local semantic map, calls `structured_state`, `causal_option_mask`, recurrent policy, target logits, target lifecycle, planner, shield, controller, execution guard, and `primitive_to_wheels`, then writes motor velocities.
4. Supervisor reads `SHOWCASE_ROBOT`, writes `HUMAN_VIEWPOINT`, and exports native frames. This is visualization-only state; it is not passed to policy/planner.
5. Until-closed termination is controlled by `robot.step(...) == -1`/Webots closure. Bounded mode additionally uses decision/deadline termination and supervisor `simulationQuit`.

Primary files: `scripts/run_real_v4_webots_demo.ps1`; `webots/controllers/hierarchical_experimental_robot/hierarchical_experimental_robot.py`; `webots/controllers/hierarchical_experimental_supervisor/hierarchical_experimental_supervisor.py`; `src/riskaware_saferrl/hierarchical/planner.py`; `src/riskaware_saferrl/hierarchical/schemas.py`; `src/riskaware_saferrl/safety/*`.
