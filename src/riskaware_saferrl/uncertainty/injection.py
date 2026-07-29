from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class PerceptionUncertaintyConfig:
    false_negative_rate: float = 0.0
    visual_perturbation: str = "normal"
    seed: int = 0

    def validate(self) -> None:
        if self.false_negative_rate not in {0.0, 0.1, 0.2, 0.3}:
            raise ValueError("false_negative_rate must be one of 0, 0.1, 0.2, 0.3")
        if self.visual_perturbation not in {
            "normal",
            "low_light",
            "motion_blur",
            "camera_noise",
        }:
            raise ValueError("Unsupported visual perturbation")


def perturb_semantic_observation(
    observation: dict[str, np.ndarray],
    config: PerceptionUncertaintyConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return a perturbed copy while preserving clean environment truth."""
    config.validate()
    result = {key: np.array(value, copy=True) for key, value in observation.items()}
    semantic = result["map"]
    perceptual_channels = (0, 1, 2, 3, 6, 7, 8)
    active = semantic[list(perceptual_channels)] > 0
    drops = rng.random(active.shape) < config.false_negative_rate
    selected = semantic[list(perceptual_channels)]
    selected[active & drops] = 0.0
    if config.visual_perturbation == "low_light":
        selected *= 0.55
        selected[rng.random(selected.shape) < 0.08] = 0.0
    elif config.visual_perturbation == "motion_blur":
        selected[:] = (
            np.roll(selected, -1, axis=2) + selected + np.roll(selected, 1, axis=2)
        ) / 3.0
    elif config.visual_perturbation == "camera_noise":
        selected[:] = np.clip(selected + rng.normal(0.0, 0.08, selected.shape), 0.0, 1.0)
    semantic[list(perceptual_channels)] = selected
    result["map"] = semantic.astype(np.float32)
    return result


class PerceptionUncertaintyWrapper(gym.ObservationWrapper):
    """Deterministically perturb only the observation delivered to the agent."""

    def __init__(self, environment: gym.Env, config: PerceptionUncertaintyConfig):
        super().__init__(environment)
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.clean_observation: dict[str, np.ndarray] | None = None
        self.perturbed_observation: dict[str, np.ndarray] | None = None

    def reset(self, *, seed=None, options=None):
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        return super().reset(seed=seed, options=options)

    def observation(self, observation):
        self.clean_observation = {
            key: np.array(value, copy=True) for key, value in observation.items()
        }
        self.perturbed_observation = perturb_semantic_observation(
            observation, self.config, self.rng
        )
        return self.perturbed_observation
