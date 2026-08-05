from __future__ import annotations

import torch

from riskaware_saferrl.algorithms import (
    AntiWindupPIDMultiplier,
    RecurrentMaskedSafePolicyV3,
    RiskShieldHRMPPOSafeV3,
)
from riskaware_saferrl.buffers import RecurrentVectorCostBuffer
from riskaware_saferrl.policies import RecurrentMaskedPolicy
from riskaware_saferrl.safety.safety_contract_v3 import VECTOR_COST_NAMES


def budgets() -> dict[str, float]:
    return {
        "collision": 0.0,
        "restricted_zone": 0.1,
        "worker_near_miss": 0.1,
        "uncontrolled_semantic_risk": 1.0,
        "uncertainty": 1.0,
    }


def rollout(policy: RecurrentMaskedSafePolicyV3, steps: int = 8):
    buffer = RecurrentVectorCostBuffer()
    hidden = policy.initial_state(1, "cpu")
    for index in range(steps):
        maps = torch.rand(1, 1, 11, 16, 16)
        states = torch.rand(1, 1, 22)
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
            vector_cost=torch.tensor([float(index == 2), 0.0, 0.1, 0.2, 0.05]),
            reward_value=output.reward_value[0, 0].detach(),
            vector_cost_value=output.vector_cost_values[0, 0].detach(),
            log_probability=output.distribution.log_prob(action)[0, 0].detach(),
            done=torch.tensor(index == steps - 1),
        )
        hidden = output.recurrent_state.detach()
    return buffer.finalize(torch.tensor(0.0), torch.zeros(5))


def test_vector_buffer_keeps_five_advantages_separate() -> None:
    policy = RecurrentMaskedSafePolicyV3()
    batch = rollout(policy)
    assert batch.vector_costs.shape == (8, 5)
    assert batch.vector_cost_advantages.shape == (8, 5)
    assert batch.vector_cost_returns.shape == (8, 5)
    assert not torch.equal(
        batch.vector_cost_advantages[:, 0],
        batch.vector_cost_advantages[:, 1],
    )


def test_v2_checkpoint_initializes_shared_v3_policy() -> None:
    v2 = RecurrentMaskedPolicy()
    checkpoint = {"model_state_dict": v2.state_dict()}
    v3 = RecurrentMaskedSafePolicyV3.from_v2_checkpoint(checkpoint, "cpu")
    assert torch.equal(v3.actor.weight, v2.actor.weight)
    assert torch.equal(
        v3.state_encoder[0].weight[:, :17],
        v2.state_encoder[0].weight,
    )
    assert v3.state_encoder[0].weight.shape[1] == 22
    assert set(v3.cost_critics) == set(VECTOR_COST_NAMES)


def test_anti_windup_warmup_projection_and_restore() -> None:
    multiplier = AntiWindupPIDMultiplier(maximum=1.0, warmup_updates=2)
    assert multiplier.update(10.0, 0.0) == 0.0
    assert multiplier.update(10.0, 0.0) == 0.0
    multiplier.update(10.0, 0.0)
    assert multiplier.update(10.0, 0.0) == 1.0
    assert multiplier.saturated
    state = multiplier.state_dict()
    restored = AntiWindupPIDMultiplier()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_safe_v3_updates_all_cost_critics_and_multipliers() -> None:
    torch.manual_seed(3)
    policy = RecurrentMaskedSafePolicyV3()
    batch = rollout(policy)
    algorithm = RiskShieldHRMPPOSafeV3(
        policy,
        budgets=budgets(),
        multiplier_warmup_updates=0,
    )
    before = {name: critic.weight.detach().clone() for name, critic in policy.cost_critics.items()}
    update = algorithm.update(
        batch,
        observed_episode_costs={name: 2.0 for name in VECTOR_COST_NAMES},
        epochs=2,
    )
    assert set(update.cost_value_losses) == set(VECTOR_COST_NAMES)
    assert set(update.constraint_residuals) == set(VECTOR_COST_NAMES)
    assert set(update.multipliers) == set(VECTOR_COST_NAMES)
    assert all(
        not torch.equal(before[name], policy.cost_critics[name].weight)
        for name in VECTOR_COST_NAMES
    )
    assert update.gradient_norm >= 0.0


def test_safe_v3_checkpoint_restores_each_multiplier() -> None:
    algorithm = RiskShieldHRMPPOSafeV3(
        RecurrentMaskedSafePolicyV3(),
        budgets=budgets(),
        multiplier_warmup_updates=0,
    )
    for multiplier in algorithm.multipliers.values():
        multiplier.update(2.0, 0.0)
    state = algorithm.state_dict()
    restored = RiskShieldHRMPPOSafeV3(
        RecurrentMaskedSafePolicyV3(),
        budgets=budgets(),
    )
    restored.load_state_dict(state)
    assert {
        name: multiplier.state_dict() for name, multiplier in restored.multipliers.items()
    } == state["multipliers"]
