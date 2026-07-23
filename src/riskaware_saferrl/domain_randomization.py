from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DomainParameters:
    lighting_intensity: float
    obstacle_dropout_probability: float
    sensor_noise_std: float
    perception_false_negative_probability: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class DomainRandomizer:
    """Seeded sampler for documented simulation-domain parameters."""

    def __init__(self, seed: int, *, enabled: bool = True) -> None:
        self.seed = int(seed)
        self.enabled = enabled
        self._generator = np.random.default_rng(self.seed)

    def sample(self) -> DomainParameters:
        if not self.enabled:
            return DomainParameters(1.0, 0.0, 0.0, 0.0)
        return DomainParameters(
            lighting_intensity=float(self._generator.uniform(0.7, 1.3)),
            obstacle_dropout_probability=float(self._generator.uniform(0.0, 0.05)),
            sensor_noise_std=float(self._generator.uniform(0.0, 0.02)),
            perception_false_negative_probability=float(self._generator.uniform(0.0, 0.10)),
        )
