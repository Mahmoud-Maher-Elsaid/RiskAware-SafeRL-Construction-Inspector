# Safety Contract v3

## Purpose

Safety Contract v3 supplements, but never overwrites, the historical raw
safety metrics. The former raw-cost comparison is retained under the label
`LEGACY_INACTIVITY_BIASED_DIAGNOSTIC`: RiskShield-PPO v1 achieved low
exposure partly by failing every mission and moving very little.

The v3 contract requires useful mission performance and stronger absolute
hard-safety limits. It distinguishes unique physical events from event
duration and separates prohibited violations, controlled inspection
exposure, semantic mission risk, and perception uncertainty.

## Hard constraints

- Collision.
- Restricted-zone entry. Continued occupancy is recorded as event duration,
  not additional unique entries.
- Minimum human-clearance breach.
- An unresolved no-safe-action emergency stop.
- Invalid motor command.

Hard constraints are never reclassified as controlled inspection exposure.

## Soft constraints

- Near-miss exposure above the hard-clearance boundary.
- Elevated semantic risk.
- Perception uncertainty exposure.
- Excessive risk dwell.
- Repeated or unnecessary shield intervention.

Soft events retain duration, peak severity, and integrated severity.

## Controlled inspection exposure

Approaching an observed risk target is controlled only when all conditions
hold:

1. The target is recognized in the causal observation.
2. Inspection intent is active.
3. The approach direction is assessed as safe.
4. Speed is at or below the configured limit.
5. Human clearance is at least the configured minimum.
6. Dwell remains within the configured bound.
7. A retreat route is available.
8. The predictive Safety Shield is active.

Controlled exposure receives a small operational cost and remains fully
logged. Collision, restricted entry, and human-clearance breach are never
waived.

Human contact (clearance below one grid cell) is a hard clearance breach.
One-cell separation remains a soft near-miss exposure with duration and
integrated severity. The legacy per-step near-miss counter is preserved.

## Event schema and metrics

Every event records its identifier, type, start and end steps, duration, peak
and integrated severity, robot position, available target identity, proposed
and executed actions, shield decision, inspection intent, controlled status,
resolution status, and resolution action.

The runtime retains the legacy raw fields plus hard, soft, event,
success-conditioned, progress-normalized, and per-100-step v3 costs. Old
reports receive derived v3 values only when their raw traces are sufficient;
otherwise the value is unavailable.

This contract is validated in simulation in the tested configurations. It
does not establish real-world construction-site safety.
