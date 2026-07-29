from __future__ import annotations

import ast
import inspect
from copy import deepcopy

import numpy as np

from riskaware_saferrl.baselines.causal_expert import CausalObservationExpert


def observation(
    *,
    agent: tuple[int, int] = (3, 3),
    hazard: tuple[int, int] | None = None,
    inspected: tuple[int, int] | None = None,
    mask: tuple[int, ...] = (1, 1, 1, 1, 0),
) -> dict[str, np.ndarray]:
    semantic_map = np.zeros((11, 16, 16), dtype=np.float32)
    semantic_map[9, :8, :8] = 1
    semantic_map[4, 3, 3] = 1
    semantic_map[5, agent[0], agent[1]] = 1
    if hazard is not None:
        semantic_map[1, hazard[0], hazard[1]] = 1
    if inspected is not None:
        semantic_map[10, inspected[0], inspected[1]] = 1
    state = np.zeros(17, dtype=np.float32)
    state[0] = agent[0] / 7
    state[1] = agent[1] / 7
    state[3] = 1
    state[9] = 0.5
    return {
        "map": semantic_map,
        "state": state,
        "action_mask": np.asarray(mask, dtype=np.int8),
    }


def test_source_has_no_private_environment_access() -> None:
    tree = ast.parse(
        inspect.getsource(__import__("riskaware_saferrl.baselines.causal_expert", fromlist=["*"]))
    )
    forbidden = {
        "hazards",
        "obstacles",
        "workers",
        "dynamic_hazards",
        "restricted",
        "ppe_risk",
        "agent",
        "np_random",
    }
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint(forbidden)


def test_hidden_hazard_cannot_affect_decision_before_discovery() -> None:
    first = CausalObservationExpert().decide(observation())
    second_input = observation()
    second_input["map"][1, 12, 12] = 0
    second = CausalObservationExpert().decide(second_input)
    assert first == second


def test_identical_histories_produce_identical_decisions() -> None:
    history = [observation(), observation(agent=(3, 4)), observation(agent=(4, 4))]
    experts = (CausalObservationExpert(), CausalObservationExpert())
    decisions = [[expert.decide(deepcopy(item)) for item in history] for expert in experts]
    assert decisions[0] == decisions[1]


def test_action_is_always_valid_under_mask() -> None:
    expert = CausalObservationExpert()
    for mask in ((1, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 0, 0, 1)):
        item = observation(mask=mask)
        decision = expert.decide(item)
        assert item["action_mask"][decision.final_action]


def test_reset_clears_episode_memory() -> None:
    expert = CausalObservationExpert()
    expert.decide(observation(hazard=(3, 5)))
    assert expert.discovered_target_count == 1
    expert.reset()
    assert expert.discovered_target_count == 0
    assert expert.memory_snapshot()["step"] == 0


def test_memory_updates_only_from_observations() -> None:
    expert = CausalObservationExpert()
    item = observation(hazard=(2, 3))
    expert.decide(item)
    snapshot = expert.memory_snapshot()
    assert snapshot["known_hazards"][2, 3]
    assert int(np.count_nonzero(snapshot["known_hazards"])) == 1


def test_no_future_information_is_consumed() -> None:
    expert = CausalObservationExpert()
    item = observation()
    decision = expert.decide(item)
    item["future_map"] = np.ones((11, 16, 16), dtype=np.float32)
    with np.testing.assert_raises(ValueError):
        expert.decide(item)
    assert decision == CausalObservationExpert().decide(observation())


def test_unreachable_target_falls_back_to_exploration() -> None:
    item = observation(hazard=(1, 1))
    item["map"][0, 2, :] = 1
    item["map"][0, :, 2] = 1
    decision = CausalObservationExpert().decide(item)
    assert decision.target_type == "frontier"
    assert item["action_mask"][decision.final_action]


def test_explores_frontier_when_no_target_known() -> None:
    decision = CausalObservationExpert().decide(observation())
    assert decision.target_type == "frontier"
    assert decision.reason in {"navigate_to_frontier", "expand_discovered_map"}


def test_inspects_valid_observed_target_in_range() -> None:
    item = observation(hazard=(3, 4), mask=(1, 1, 1, 1, 1))
    decision = CausalObservationExpert().decide(item)
    assert decision.proposed_action == 4
    assert decision.final_action == 4
    assert decision.reason == "inspect_observed_target"
