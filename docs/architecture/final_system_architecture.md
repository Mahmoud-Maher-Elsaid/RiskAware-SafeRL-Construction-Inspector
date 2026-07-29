# Final System Architecture

```mermaid
flowchart TD
  Camera --> CV[CUDA detector]
  CV --> Filter[Temporal filter]
  Filter --> Risk[Semantic risk map]
  Sensors --> Obs[Typed observation adapter]
  Risk --> Obs
  Obs --> Policy[Trained RL policy]
  Policy --> Shield[Predictive Safety Shield]
  Shield --> Motors[Webots wheel motors]
  Motors --> Sensors
  Shield --> Logs[Action and intervention evidence]
```

The research grid benchmark and Webots runtime share observation, risk, and
safety concepts but are separate evaluation domains. Planner actions are never
labeled RL. Simulator truth is labeled separately from CUDA detections.
