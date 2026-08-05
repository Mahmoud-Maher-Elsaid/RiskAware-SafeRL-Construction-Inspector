from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from riskaware_saferrl.safety import (
    ControlledInspectionState,
    EventAwarePredictiveShieldV3,
    SafetyContractV3,
)


def observation() -> dict[str, np.ndarray]:
    semantic_map = np.zeros((11, 16, 16), dtype=np.float32)
    semantic_map[5, 4, 4] = 1.0
    semantic_map[9, :8, :8] = 1.0
    return {
        "map": semantic_map,
        "state": np.zeros(17, dtype=np.float32),
        "action_mask": np.ones(5, dtype=np.int8),
    }


def shield() -> EventAwarePredictiveShieldV3:
    return EventAwarePredictiveShieldV3(
        SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    )


def test_safe_action_is_minimally_accepted_and_deterministic() -> None:
    first_shield = shield()
    second_shield = shield()
    first = first_shield.decide(observation(), 3)
    second = second_shield.decide(observation(), 3)
    assert first.final_action == 3
    assert first.shield_decision == "accept"
    assert first.predicted_trajectory == second.predicted_trajectory


def test_avoidable_collision_is_replaced() -> None:
    value = observation()
    value["map"][0, 4, 5] = 1.0
    value["action_mask"][3] = 0
    decision = shield().decide(value, 3)
    assert decision.final_action != 3
    assert decision.shield_decision in {"replace", "modify"}
    assert "invalid_or_obstructed_action" in decision.rejection_reasons


def test_safe_replacement_prefers_minimal_action_change_before_soft_risk() -> None:
    value = observation()
    value["map"][0, 4, 5] = 1.0
    value["action_mask"][3] = 0
    value["map"][8, 4, 3] = 0.25
    decision = shield().decide(value, 3)
    assert decision.final_action == 2


def test_restricted_boundary_and_worker_clearance_are_hard() -> None:
    value = observation()
    value["map"][3, 4, 5] = 1.0
    value["map"][2, 3, 4] = 1.0
    restricted = shield().decide(value, 3)
    worker = shield().decide(value, 0)
    assert restricted.final_action != 3
    assert worker.final_action != 0


def test_observed_worker_memory_survives_a_false_negative() -> None:
    instance = shield()
    visible = observation()
    visible["map"][2, 4, 5] = 1.0
    instance.decide(visible, 0)
    false_negative = observation()
    decision = instance.decide(false_negative, 3)
    assert decision.final_action != 3
    assert "human_clearance_breach" in decision.rejection_reasons


def test_noisy_human_clearance_requires_two_observations() -> None:
    instance = shield()
    noisy = observation()
    noisy["state"][8] = 0.2
    first = instance.decide(noisy, 3)
    later = [instance.decide(noisy, 3) for _ in range(8)]
    assert first.final_action != 3
    assert "unconfirmed_human_clearance" in first.rejection_reasons
    assert later[-1].final_action == 3


def test_controlled_inspection_is_not_blocked_by_semantic_risk() -> None:
    value = observation()
    value["map"][8, 4, 4] = 1.0
    inspection = ControlledInspectionState(
        target_recognized=True,
        inspection_intent=True,
        safe_approach=True,
        speed=0.0,
        human_clearance_cells=2.0,
        dwell_steps=1,
        retreat_route_available=True,
        shield_active=True,
    )
    decision = shield().decide(value, 4, inspection)
    assert decision.controlled_inspection
    assert decision.final_action == 4


def test_dynamic_hazard_prediction_increases_horizon_and_is_logged() -> None:
    value = observation()
    value["map"][7, 4, 6] = 1.0
    decision = shield().decide(value, 3)
    assert decision.adaptive_horizon > 1
    assert decision.predicted_vector_cost.uncontrolled_semantic_risk >= 0.0


def test_deadlock_escape_does_not_repeat_stop() -> None:
    value = observation()
    value["map"][0, 3, 4] = 1.0
    value["action_mask"][0] = 0
    instance = shield()
    decision = None
    for _ in range(10):
        decision = instance.decide(value, 0)
    assert decision is not None
    assert decision.final_action != 4
    assert decision.shield_decision == "retreat"


def test_no_safe_action_uses_emergency_stop() -> None:
    value = observation()
    value["action_mask"][:] = 0
    value["action_mask"][4] = 1
    value["map"][3, 4, 4] = 1.0
    decision = shield().decide(value, 0)
    assert decision.final_action == 4
    assert decision.emergency_stop


def test_latency_is_bounded_in_unit_environment() -> None:
    instance = shield()
    started = time.perf_counter()
    decisions = [instance.decide(observation(), 3) for _ in range(100)]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert all(decision.computation_time_ms >= 0.0 for decision in decisions)
    assert elapsed_ms < 1000.0
