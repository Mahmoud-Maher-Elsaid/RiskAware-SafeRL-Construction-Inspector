# HRMPPO v2 violation audit

Status: **PASSED**

- Episodes: 90
- Total transitions: 18834
- Reproduced legacy constraint count: 5460
- Mean legacy raw safety cost: 35.636111
- Invalid policy actions: 0
- Invalid motor commands: 0

## Step counts

- controlled_inspection_exposure: 20
- dynamic_hazard_proximity: 1585
- emergency_stop: 2
- restricted_zone: 743
- shield_intervention: 1158
- uncontrolled_ppe_exposure: 403
- worker_near_miss: 3405

## Unique events

- controlled_inspection_exposure: 20
- dynamic_hazard_proximity: 804
- emergency_stop: 2
- restricted_zone: 697
- shield_intervention: 1062
- uncontrolled_ppe_exposure: 399
- worker_near_miss: 1245

## Category contract audit

| Type | Severity | Legacy weight | Counter unit | Mission conflict |
|---|---|---:|---|---|
| collision | hard | 1.0 | one increment per colliding transition | False |
| controlled_inspection_exposure | bounded mission exposure | 0.25 | candidate classification; v3 adds dwell and retreat checks | True |
| dynamic_hazard_proximity | soft exposure unless minimum clearance is breached | 0.5 | shares one increment with worker proximity per timestep | possibly |
| emergency_stop | hard when unresolved | 0.0 | one event per contiguous stop episode | False |
| restricted_zone | hard entry; subsequent occupancy is dwell exposure | 1.0 | one increment per occupied timestep | False |
| shield_intervention | soft operational event unless generated action violates a hard constraint | 0.0 | one step plus contiguous intervention event | may impede progress when repeated |
| uncontrolled_ppe_exposure | soft semantic-risk exposure | 0.25 | one raw-cost increment per timestep | True |
| worker_near_miss | hard when below clearance; otherwise soft proximity exposure | 0.5 | one shared near-miss increment per timestep | False |

## Accounting findings

- legacy_counter_unit: per timestep, not unique physical event
- near_miss_combines_worker_and_dynamic_hazard: True
- worker_dynamic_overlap_single_legacy_increment: True
- ppe_in_legacy_constraint_count: False
- ppe_in_legacy_raw_cost: True
- repeated_unchanged_exposure_recharged: True
- post_termination_accounting: False
- initial_state_accounting_before_first_action: False

The legacy 5,460 value is an exposure-duration counter. It is not a count of unique physical safety events. PPE exposure contributes raw cost but is not included in that legacy counter. Worker and dynamic-hazard proximity share one legacy near-miss increment even when both categories overlap.
