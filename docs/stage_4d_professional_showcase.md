# Stage 4D Professional Construction Showcase

## Purpose

Stage 4D provides a dedicated visual presentation world for the
RiskAware SafeRL Construction Inspector project.

The existing Stage 4B2 world remains unchanged and continues to serve
as the deterministic runtime-validation environment.

## Visual contents

The showcase world includes:

- a large reinforced-concrete construction slab
- a marked safe robot walkway
- a PPE preparation zone
- a restricted excavation zone
- an unfinished concrete structure
- multi-level scaffolding
- a tower crane
- a site office container
- timber pallets and steel pipes
- workers wearing visible PPE
- inspection checkpoints
- perimeter safety fencing
- a larger inspection robot model
- an on-screen simulation status overlay

## Controller isolation

The showcase robot controller imports only the Webots controller API.

It does not import:

- Gymnasium
- Stable-Baselines3
- SB3-Contrib
- MaskablePPO
- riskaware_saferrl

This allows the showcase to open independently from the research
runtime environment.

## Scope boundary

The Stage 4D movement sequence is scripted for visual presentation.

It is not presented as a live reinforcement-learning execution.

The validated research path remains separate:

```text
live sensors
    -> observation bridge
    -> task-valid action mask
    -> MaskablePPO proposal
    -> semantic shield
    -> deadlock-safe fallback
    -> executed action
```

At the current Stage 4D milestone, MaskablePPO does not control the
showcase robot motors.

The visual showcase does not constitute an absolute safety guarantee.
