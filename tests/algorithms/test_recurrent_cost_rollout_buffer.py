from __future__ import annotations

import torch

from riskaware_saferrl.buffers import RecurrentCostRolloutBuffer


def test_reward_and_cost_advantages_are_independent() -> None:
    buffer = RecurrentCostRolloutBuffer(gamma=1.0, gae_lambda=1.0)
    for index in range(3):
        buffer.add(
            map=torch.zeros(11, 16, 16),
            state=torch.zeros(17),
            recurrent_state=torch.zeros(256),
            episode_start=torch.tensor(index == 0),
            action=torch.tensor(0),
            action_mask=torch.ones(5, dtype=torch.bool),
            reward=torch.tensor(1.0),
            cost=torch.tensor(float(index == 1)),
            reward_value=torch.tensor(0.0),
            cost_value=torch.tensor(0.0),
            log_probability=torch.tensor(0.0),
            done=torch.tensor(index == 2),
        )
    batch = buffer.finalize(torch.tensor(0.0), torch.tensor(0.0))
    torch.testing.assert_close(batch.reward_advantages, torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(batch.cost_advantages, torch.tensor([1.0, 1.0, 0.0]))
    assert not torch.equal(batch.reward_advantages, batch.cost_advantages)
