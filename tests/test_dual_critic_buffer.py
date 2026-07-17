from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from riskaware_saferrl.algorithms import (
    DualCriticMaskableDictRolloutBuffer,
)


def test_cost_gae_is_separate_from_reward_gae() -> None:
    observation_space = spaces.Dict(
        {
            "map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )
    buffer = DualCriticMaskableDictRolloutBuffer(
        buffer_size=2,
        observation_space=observation_space,
        action_space=spaces.Discrete(2),
        device="cpu",
        gae_lambda=1.0,
        gamma=1.0,
        n_envs=1,
        cost_gamma=1.0,
        cost_gae_lambda=1.0,
    )

    observation = {
        "map": np.zeros((1, 4), dtype=np.float32),
        "state": np.zeros(
            (1, 2),
            dtype=np.float32,
        ),
    }

    buffer.add(
        observation,
        np.asarray([[0]]),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        torch.asarray([0.0]),
        torch.asarray([0.0]),
        costs=np.asarray([1.0]),
        cost_values=torch.asarray([0.0]),
        action_masks=np.asarray([[True, True]]),
    )
    buffer.add(
        observation,
        np.asarray([[1]]),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        torch.asarray([0.0]),
        torch.asarray([0.0]),
        costs=np.asarray([2.0]),
        cost_values=torch.asarray([0.0]),
        action_masks=np.asarray([[True, True]]),
    )

    buffer.compute_cost_returns_and_advantage(
        last_cost_values=torch.asarray([0.0]),
        dones=np.asarray([True]),
    )

    assert np.allclose(
        buffer.cost_returns[:, 0],
        np.asarray([3.0, 2.0]),
    )
