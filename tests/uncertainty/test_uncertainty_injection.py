from __future__ import annotations

import numpy as np

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv
from riskaware_saferrl.uncertainty import (
    PerceptionUncertaintyConfig,
    PerceptionUncertaintyWrapper,
    perturb_semantic_observation,
)


def test_injection_is_deterministic_and_does_not_mutate_clean_input() -> None:
    environment = ResearchConstructionEnv(GridEnvironmentConfig(size=8))
    observation, _ = environment.reset(seed=4)
    clean = {key: value.copy() for key, value in observation.items()}
    config = PerceptionUncertaintyConfig(0.2, "camera_noise", seed=9)
    first = perturb_semantic_observation(observation, config, np.random.default_rng(9))
    second = perturb_semantic_observation(observation, config, np.random.default_rng(9))
    np.testing.assert_array_equal(first["map"], second["map"])
    np.testing.assert_array_equal(observation["map"], clean["map"])


def test_all_required_visual_perturbations_execute() -> None:
    environment = ResearchConstructionEnv(GridEnvironmentConfig(size=8))
    observation, _ = environment.reset(seed=2)
    outputs = []
    for visual in ("normal", "low_light", "motion_blur", "camera_noise"):
        outputs.append(
            perturb_semantic_observation(
                observation,
                PerceptionUncertaintyConfig(0.1, visual, seed=3),
                np.random.default_rng(3),
            )["map"]
        )
    assert all(output.shape == observation["map"].shape for output in outputs)
    assert not np.array_equal(outputs[0], outputs[1])


def test_wrapper_keeps_clean_and_agent_observation_separate() -> None:
    base = ResearchConstructionEnv(GridEnvironmentConfig(size=8))
    wrapper = PerceptionUncertaintyWrapper(
        base, PerceptionUncertaintyConfig(0.3, "low_light", seed=7)
    )
    perturbed, _ = wrapper.reset(seed=7)
    assert wrapper.clean_observation is not None
    assert wrapper.perturbed_observation is not None
    assert base.hazards
    assert not np.shares_memory(wrapper.clean_observation["map"], perturbed["map"])


def test_required_false_negative_rates_are_valid() -> None:
    for rate in (0.0, 0.1, 0.2, 0.3):
        PerceptionUncertaintyConfig(rate, "normal").validate()
