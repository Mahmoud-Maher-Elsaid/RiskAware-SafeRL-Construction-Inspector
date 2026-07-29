from __future__ import annotations

import numpy as np

from riskaware_saferrl.envs import GridEnvironmentConfig
from riskaware_saferrl.training import (
    ContinuousActionAdapter,
    EpisodeMetricsCallback,
    flatten_research_environment,
)


def test_continuous_action_bins_cover_all_discrete_actions() -> None:
    config = GridEnvironmentConfig(size=8)
    environment = flatten_research_environment(config, continuous_actions=True)
    adapter = environment.env
    assert isinstance(adapter, ContinuousActionAdapter)
    actions = [
        adapter.action(np.array([value], dtype=np.float32)) for value in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ]
    assert actions == [0, 1, 2, 3, 4]


def test_flattened_schema_is_cross_site_compatible() -> None:
    small = flatten_research_environment(GridEnvironmentConfig(size=8))
    large = flatten_research_environment(GridEnvironmentConfig(size=16))
    assert small.observation_space == large.observation_space
    assert small.reset(seed=4)[0].shape == large.reset(seed=4)[0].shape


def test_early_stopping_configuration_is_explicit(tmp_path) -> None:
    callback = EpisodeMetricsCallback(
        tmp_path / "metrics.jsonl",
        early_stopping_patience_episodes=5,
        minimum_episodes=10,
    )
    assert callback.early_stopping_patience_episodes == 5
    assert callback.minimum_episodes == 10
