from __future__ import annotations

import torch

from riskaware_saferrl.algorithms.hierarchical_v4 import (
    RiskShieldHierarchicalHRMPPOV4,
)
from riskaware_saferrl.buffers import RecurrentVectorCostBuffer
from riskaware_saferrl.hierarchical import HierarchicalMissionPolicy
from riskaware_saferrl.hierarchical.schemas import VECTOR_COST_NAMES


def budgets() -> dict[str, float]:
    return {
        "collision": 0.0,
        "restricted_zone": 0.0,
        "human_clearance": 0.0,
        "uncontrolled_semantic_risk": 1.0,
        "uncertainty": 1.0,
        "shield_dependence": 1.0,
        "deadlock": 0.0,
    }


def rollout(policy: HierarchicalMissionPolicy, steps: int = 8):
    buffer = RecurrentVectorCostBuffer(cost_count=len(VECTOR_COST_NAMES))
    hidden = policy.initial_state(1, "cpu")
    for index in range(steps):
        maps = torch.zeros(1, 1, 11, 16, 16)
        maps[0, 0, 5, 4, 4] = 1
        states = torch.rand(1, 1, 32)
        masks = torch.ones(1, 1, 9, dtype=torch.bool)
        output = policy(maps, states, masks, hidden)
        option = output.option_distribution.sample()
        buffer.add(
            map=maps[0, 0],
            state=states[0, 0],
            recurrent_state=hidden,
            episode_start=torch.tensor(index == 0),
            action=option[0, 0],
            action_mask=masks[0, 0],
            reward=torch.tensor(1.0),
            vector_cost=torch.tensor([float(index == 2), 0.0, 0.1, 0.2, 0.05, 0.1, 0.0]),
            reward_value=output.reward_value[0, 0].detach(),
            vector_cost_value=output.vector_cost_values[0, 0].detach(),
            log_probability=output.option_distribution.log_prob(option)[0, 0].detach(),
            done=torch.tensor(index == steps - 1),
        )
        hidden = output.recurrent_state.detach()
    return buffer.finalize(torch.tensor(0.0), torch.zeros(len(VECTOR_COST_NAMES)))


def test_option_level_update_keeps_seven_constraints_independent() -> None:
    torch.manual_seed(9)
    policy = HierarchicalMissionPolicy()
    batch = rollout(policy)
    algorithm = RiskShieldHierarchicalHRMPPOV4(
        policy,
        budgets=budgets(),
        multiplier_warmup_updates=0,
    )
    before = [critic.weight.detach().clone() for critic in policy.cost_critics]
    update = algorithm.update(
        batch,
        observed_episode_costs={name: 2.0 for name in VECTOR_COST_NAMES},
    )
    assert set(update.cost_value_losses) == set(VECTOR_COST_NAMES)
    assert set(update.multipliers) == set(VECTOR_COST_NAMES)
    assert set(update.constraint_residuals) == set(VECTOR_COST_NAMES)
    assert all(
        not torch.equal(previous, critic.weight)
        for previous, critic in zip(before, policy.cost_critics, strict=True)
    )
    assert update.gradient_norm >= 0


def test_option_level_checkpoint_restores_optimizer_anchor_and_pid_state() -> None:
    algorithm = RiskShieldHierarchicalHRMPPOV4(
        HierarchicalMissionPolicy(),
        budgets=budgets(),
        multiplier_warmup_updates=0,
    )
    for multiplier in algorithm.multipliers.values():
        multiplier.update(2.0, 0.0)
    state = algorithm.state_dict()
    restored = RiskShieldHierarchicalHRMPPOV4(
        HierarchicalMissionPolicy(),
        budgets=budgets(),
    )
    restored.load_state_dict(state)
    assert restored.state_dict()["multipliers"] == state["multipliers"]


def test_option_level_update_rejects_missing_constraint_budget() -> None:
    incomplete = budgets()
    incomplete.pop("deadlock")
    try:
        RiskShieldHierarchicalHRMPPOV4(
            HierarchicalMissionPolicy(),
            budgets=incomplete,
        )
    except ValueError as error:
        assert "deadlock" in str(error)
    else:
        raise AssertionError("Missing vector budget was accepted")
