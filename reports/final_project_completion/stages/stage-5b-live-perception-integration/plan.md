# Stage 5B Live Perception Integration Plan

1. Reproduce the current Webots main-viewport and CUDA perception runtime.
2. Reject unusable visual output through manual inspection.
3. Repair projection state, authoritative orientation-matrix tracking, human-eye leveling, turn evidence, and final route heading.
4. Preserve the 22-frame controller-synchronized CUDA perception stream.
5. Add independent visual validation and regression tests.
6. Run the complete Stage 5B mission, inspect all required images, and publish deterministic evidence.
