# Long-run visible runtime forensic analysis

Source attempt: `reports/strong_policy_upgrade/webots_v4_visible_demo/attempt_20260822_000519_143`.

## Evidence timeline

The run produced 6,638 policy decisions over 212.416 s of simulation time. Decisions 0--499 travelled 8.004 m. Decisions 1,000--3,999 travelled only 1.053 m combined while emitting repeated turn/stop/forward recovery commands. Decisions 4,000--4,999 travelled 4.683 m, then decisions 5,000--6,637 travelled only 0.157 m and finally 0 m. The last 638 decisions were all zero wheel commands.

The first sustained stop state is caused by the planner returning `success=false`, `output_target=null` while the option remains `EXPLORE_FRONTIER`; the local controller then selects `INSPECT`/`STOP` because the path is `(start,)`. This is visible in the final planner trace and motor trace, not inferred from the summary.

The late-run planner trace reports `replanned=true` on every decision with reason `policy_urgency_or_observed_change`, despite `semantic_changed=false` on the same records. This is stale historical evidence from the user run and demonstrates that the executed path was still urgency-driven rather than path-state-driven. The current source has since set urgency telemetry-only, so this report distinguishes the historical run from the current working tree.

## Long-horizon state findings

The structured-state call uses `decision_index / max(DECISIONS, 1)`. In `until_closed` mode `DECISIONS` is empty/zero, so the progress feature saturates to 1.0 after the first decision. This makes the bounded and indefinite state distributions different and is a long-run recurrent-policy freshness defect. There is no stale sensor/timestamp guard, no rolling spin detector, and no actuator-level obstacle backstop in the current controller.

## Camera findings

The attempt's `rendered_start.png` is a Webots splash screen and `rendered_middle.png` is a Webots loading/scene-tree overlay, not a clean 3-D client frame. The launcher therefore accepted a window/capture proxy rather than the active rendered viewpoint. In first-person mode the Supervisor deliberately does not update `HUMAN_VIEWPOINT`, while the WBT uses `follow "SHOWCASE_ROBOT"`/`Mounted Shot`; this leaves the GUI's active viewpoint unproven and explains the user's static/incorrect view.

## Collision and acceleration findings

The controller sends fixed-magnitude wheel commands (`-5.0`, `-5.5`, `-1.5`) directly to motors after the shield. There is no final sensor-based speed envelope or emergency clearance clamp. Distance sensors are sampled into the observation but are not enforced at the actuator boundary. This permits a recovery/forward command to continue into an obstacle and explains the observed collision and abrupt speed changes.

## Root-cause conclusion

The P0 failures are: (1) planner no-route is converted to a permanent stop instead of a causal exploration recovery; (2) no long-window physical spin/stale-state guard; (3) no final actuator collision backstop/speed envelope; and (4) rendered validation does not prove the active GUI viewpoint. The next repair must address these coherently and derive PASS from runtime evidence.
