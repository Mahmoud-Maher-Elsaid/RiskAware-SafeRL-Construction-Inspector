from __future__ import annotations

import torch

from riskaware_saferrl.algorithms import PIDLagrangeController, RiskShieldHRMPPOV2
from riskaware_saferrl.buffers import RecurrentCostRolloutBuffer
from riskaware_saferrl.policies import RecurrentMaskedPolicy


def rollout(policy: RecurrentMaskedPolicy, steps: int = 8):
    buffer = RecurrentCostRolloutBuffer()
    hidden = policy.initial_state(1, "cpu")
    for index in range(steps):
        maps = torch.rand(1, 1, 11, 16, 16)
        states = torch.rand(1, 1, 17)
        masks = torch.ones(1, 1, 5, dtype=torch.bool)
        output = policy(maps, states, masks, hidden)
        action = output.distribution.sample()
        buffer.add(
            map=maps[0, 0],
            state=states[0, 0],
            recurrent_state=hidden,
            episode_start=torch.tensor(index == 0),
            action=action[0, 0],
            action_mask=masks[0, 0],
            reward=torch.tensor(1.0),
            cost=torch.tensor(float(index % 3 == 0)),
            reward_value=output.reward_value[0, 0].detach(),
            cost_value=output.cost_value[0, 0].detach(),
            log_probability=output.distribution.log_prob(action)[0, 0].detach(),
            done=torch.tensor(index == steps - 1),
        )
        hidden = output.recurrent_state.detach()
    return buffer.finalize(torch.tensor(0.0), torch.tensor(0.0))


def test_pid_lagrange_is_projected_and_stateful() -> None:
    controller = PIDLagrangeController(maximum=2.0)
    assert controller.update(30.0, 20.0) > 0.0
    assert controller.update(1000.0, 20.0) == 2.0
    state = controller.state_dict()
    restored = PIDLagrangeController()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_hrmppo_updates_reward_cost_hierarchy_and_actor() -> None:
    torch.manual_seed(4)
    policy = RecurrentMaskedPolicy()
    batch = rollout(policy)
    before = policy.actor.weight.detach().clone()
    algorithm = RiskShieldHRMPPOV2(policy, safety_budget=2.0)
    result = algorithm.update(batch, epochs=2)
    assert result.total_loss == result.total_loss
    assert result.reward_value_loss >= 0.0
    assert result.cost_value_loss >= 0.0
    assert result.hierarchy_loss >= 0.0
    assert not torch.equal(before, policy.actor.weight)
    assert result.lagrange_multiplier >= 0.0


def test_hrmppo_checkpoint_restores_pid_and_optimizer() -> None:
    policy = RecurrentMaskedPolicy()
    algorithm = RiskShieldHRMPPOV2(policy)
    algorithm.pid_lagrange.update(30.0, 20.0)
    state = algorithm.state_dict()
    restored = RiskShieldHRMPPOV2(RecurrentMaskedPolicy())
    restored.load_state_dict(state)
    assert restored.pid_lagrange.value == algorithm.pid_lagrange.value
    assert state["algorithm"] == "RiskShield-HRMPPO-v2"
