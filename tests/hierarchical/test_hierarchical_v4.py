from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from riskaware_saferrl.hierarchical import (
    CausalRiskAwarePlanner,
    HierarchicalMissionPolicy,
    MissionOption,
    PredictiveLocalController,
    RiskShieldHierarchicalSystem,
)
from riskaware_saferrl.safety import EventAwarePredictiveShieldV3, SafetyContractV3


def observation() -> dict[str, np.ndarray]:
    semantic_map = np.zeros((11, 16, 16), dtype=np.float32)
    semantic_map[5, 4, 4] = 1.0
    semantic_map[9, :9, :9] = 1.0
    return {
        "map": semantic_map,
        "state": np.zeros(17, dtype=np.float32),
        "action_mask": np.ones(5, dtype=np.int8),
    }


def test_policy_heads_shapes_and_exact_option_masking() -> None:
    policy = HierarchicalMissionPolicy()
    maps = torch.zeros(2, 3, 11, 16, 16)
    states = torch.zeros(2, 3, 32)
    masks = torch.ones(2, 3, 9, dtype=torch.bool)
    masks[:, :, 2] = False
    output = policy(maps, states, masks)
    assert output.option_distribution.probs.shape == (2, 3, 9)
    assert torch.all(output.option_distribution.probs[:, :, 2] == 0)
    assert output.target_logits.shape == (2, 3, 257)
    assert output.duration_logits.shape == (2, 3, 8)
    assert output.risk_budgets.shape == (2, 3, 7)
    assert output.vector_cost_values.shape == (2, 3, 7)
    assert output.recurrent_state.shape == (1, 2, 256)


def test_target_prior_selects_nearest_observed_uninspected_risk() -> None:
    policy = HierarchicalMissionPolicy()
    maps = torch.zeros(1, 11, 16, 16)
    maps[0, 5, 4, 4] = 1
    maps[0, 1, 4, 5] = 1
    maps[0, 1, 12, 12] = 1
    prior = policy._causal_target_prior(maps)
    assert prior[0, 4, 5] > prior[0, 12, 12]
    assert prior[0, 0, 0] <= -32


def test_policy_recurrent_reset_and_deterministic_cpu_inference() -> None:
    torch.manual_seed(7)
    policy = HierarchicalMissionPolicy().eval()
    maps = torch.zeros(1, 2, 11, 16, 16)
    states = torch.zeros(1, 2, 32)
    masks = torch.ones(1, 2, 9, dtype=torch.bool)
    first, _ = policy.predict(maps, states, masks)
    second, _ = policy.predict(maps, states, masks)
    assert torch.equal(first, second)
    output = policy(
        maps,
        states,
        masks,
        recurrent_state=torch.ones(1, 1, 256),
        episode_starts=torch.tensor([[True, False]]),
    )
    baseline = policy(
        maps,
        states,
        masks,
        recurrent_state=torch.zeros(1, 1, 256),
    )
    assert torch.allclose(output.reward_value, baseline.reward_value)


def test_planner_uses_only_public_observation_and_finds_frontier() -> None:
    planner = CausalRiskAwarePlanner()
    value = observation()
    result = planner.plan(
        value,
        request=__import__(
            "riskaware_saferrl.hierarchical", fromlist=["PlannerRequest"]
        ).PlannerRequest(
            option=MissionOption.EXPLORE_FRONTIER,
            target=None,
            risk_budget=0.25,
            inspection_intent=False,
        ),
    )
    assert result.success
    assert "frontier" in result.target_type
    assert all(value["map"][9, position[0], position[1]] > 0 for position in result.path)


def test_hidden_hazard_does_not_change_causal_plan_before_discovery() -> None:
    first = observation()
    second = observation()
    first["map"][1, 12, 12] = 0.0
    second["map"][1, 12, 12] = 0.0
    planner_a = CausalRiskAwarePlanner()
    planner_b = CausalRiskAwarePlanner()
    from riskaware_saferrl.hierarchical import PlannerRequest

    request = PlannerRequest(MissionOption.EXPLORE_FRONTIER, None, 0.2, False)
    assert planner_a.plan(first, request).path == planner_b.plan(second, request).path


def test_worker_avoidance_retreats_instead_of_accepting_current_cell() -> None:
    value = observation()
    value["map"][2, 4, 5] = 1
    planner = CausalRiskAwarePlanner()
    from riskaware_saferrl.hierarchical import PlannerRequest

    result = planner.plan(
        value,
        PlannerRequest(MissionOption.AVOID_DYNAMIC_WORKER, None, 0.1, False),
    )
    assert result.success
    assert len(result.path) >= 2
    worker = (4, 5)
    assert sum(abs(a - b) for a, b in zip(result.path[-1], worker, strict=True)) >= 2


def test_reached_risk_target_executes_controlled_inspection_instead_of_retreating() -> None:
    value = observation()
    value["map"][8, 4, 4] = 1
    controller = PredictiveLocalController()
    decision = controller.decide(value, ((4, 4),), inspection_intent=True)
    assert decision.primitive == 4


def test_known_target_without_discovered_corridor_advances_to_causal_frontier() -> None:
    value = observation()
    value["map"][1, 12, 12] = 1
    planner = CausalRiskAwarePlanner()
    from riskaware_saferrl.hierarchical import PlannerRequest

    result = planner.plan(
        value,
        PlannerRequest(MissionOption.INSPECT_KNOWN_RISK, (12, 12), 0.2, True),
    )
    assert result.success
    assert result.path[-1] != (4, 4)
    assert result.target_type.endswith("route_frontier")


def test_unobserved_dynamic_worker_belief_expires() -> None:
    planner = CausalRiskAwarePlanner()
    value = observation()
    value["map"][2, 4, 5] = 1
    planner.update(value)
    hidden = observation()
    hidden["map"][9, 4, 5] = 0
    for _ in range(20):
        planner.update(hidden)
    assert planner._worker[4, 5] == 0


def test_restricted_zone_is_never_in_planned_path() -> None:
    value = observation()
    value["map"][3, 4, 5] = 1.0
    from riskaware_saferrl.hierarchical import PlannerRequest

    result = CausalRiskAwarePlanner().plan(
        value, PlannerRequest(MissionOption.EXPLORE_FRONTIER, None, 0.5, False)
    )
    assert (4, 5) not in result.path


def test_controller_uses_valid_primitive_and_avoids_observed_worker() -> None:
    value = observation()
    value["map"][2, 4, 5] = 1.0
    decision = PredictiveLocalController().decide(value, ((4, 4), (4, 5)), inspection_intent=False)
    assert value["action_mask"][decision.primitive]
    assert decision.primitive != 3


def test_inspection_intent_does_not_stop_before_reaching_approach_cell() -> None:
    decision = PredictiveLocalController().decide(
        observation(), ((4, 4), (4, 5), (4, 6)), inspection_intent=True
    )
    assert decision.primitive == 3


def test_exploration_does_not_hold_at_current_frontier() -> None:
    value = observation()
    planner = CausalRiskAwarePlanner()
    from riskaware_saferrl.hierarchical import PlannerRequest

    route = planner.plan(value, PlannerRequest(MissionOption.EXPLORE_FRONTIER, None, 0.2, False))
    decision = PredictiveLocalController().decide(value, route.path, inspection_intent=False)
    assert len(route.path) > 1
    assert decision.primitive != 4


def test_four_layer_system_executes_shielded_primitive() -> None:
    contract = SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    system = RiskShieldHierarchicalSystem(
        CausalRiskAwarePlanner(),
        PredictiveLocalController(),
        EventAwarePredictiveShieldV3(contract),
    )
    decision = system.execute_option(
        observation(),
        option=MissionOption.EXPLORE_FRONTIER,
        target=None,
        risk_budget=0.25,
        inspection_intent=False,
    )
    assert decision.executed_primitive == decision.shield.final_action
    assert decision.planner.success
