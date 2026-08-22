# Root-cause map

| ID | Severity | First faulty component | Root cause | Evidence |
|---|---|---|---|---|
| RC-01 | P0 | Planner/controller boundary | `PlannerResult(success=false, target=None, path=(start,))` is passed to the local controller, which selects `INSPECT`/`STOP`; no causal exploration recovery follows. | Final planner rows and motor rows in attempt `20260822_000519_143`. |
| RC-02 | P0 | Robot runtime loop | Only a short turn counter exists; it resets on small translation/replanning and there is no rolling physical spin detector. | 1,000--3,999 and late-run pose/motor windows. |
| RC-03 | P1 | Structured-state construction | `decision_index / max(DECISIONS,1)` saturates immediately in until-closed mode, changing recurrent input semantics. | Current controller source line 558. |
| RC-04 | P0 | Actuator boundary | Fixed wheel speeds are sent after shielding; sensor clearance never imposes a final clamp/stop. | Current controller source lines 780--793; no emergency path. |
| RC-05 | P0 | Visible validation | Window handle and GUI exports are treated as rendered camera evidence; captured images are splash/loading overlays. | `rendered_start.png`, `rendered_middle.png`, launcher summary. |
| RC-06 | P1 | Camera ownership | First-person mode disables Supervisor updates while WBT Mounted Shot remains unproven as the active GUI viewpoint. | Current supervisor lines 112--116 and WBT viewpoint. |
| RC-07 | P1 | Acceptance launcher | `until_closed` prints PASS from normal user shutdown/summary without behavioral gates. | `run_real_v4_webots_demo.ps1` lines 212--213. |
| RC-08 | P1 | Runtime freshness | No timestamp monotonicity/stale-frame guard and unbounded recurrent hash list. | Current controller loop and summary. |
| RC-09 | P0 | Planner no-route handling | A failed route is represented as a one-cell path, which is indistinguishable from a completed inspection and causes stop oscillation. | Final planner trace. |
| RC-10 | P0 | Safety runtime | No rolling translation/yaw spin detector that survives target/replan changes. | Long-run windows show repeated turns/stops and near-zero translation. |
| RC-11 | P0 | Safety runtime | No guaranteed recovery transition from stop/hold/no-route states. | Final 638 zero-wheel decisions. |
| RC-12 | P0 | Motor command path | No final velocity clamp or clearance-aware speed envelope. | Fixed `-5.0/-5.5/-1.5` commands and user collision report. |
| RC-13 | P0 | Sensor-to-actuator path | Distance readings enter observations but cannot block an imminent forward command at the motor boundary. | Current controller has no clearance backstop. |
| RC-14 | P1 | Camera evidence | GUI capture can be a splash/loading surface; no active-client/render freshness check. | Rendered screenshots and launcher state. |
| RC-15 | P1 | Acceptance logic | `USER_CLOSED_WEBOTS=TRUE` is conflated with behavioral success. | Launcher unconditional `UNTIMED_FIRST_PERSON_DEMO=PASSED`. |
