# Data Card

The repository tracks configuration and metadata, not large raw datasets or
videos. The research grid benchmark is generated deterministically from seeded
scenario configurations. Webots provides rendered simulation frames and
simulator-ground-truth semantic layers.

The PPE detector's source dataset remains a local licensed artifact. Its class
list and checkpoint metadata are recorded without committing the raw dataset.
Users must independently verify dataset licensing and worker-privacy
requirements before reuse.

Perception experiments preserve three distinct layers: clean simulator truth,
seeded perturbed perception, and the observation delivered to the agent.
