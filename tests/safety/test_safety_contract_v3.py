from __future__ import annotations

from pathlib import Path

from riskaware_saferrl.safety import ControlledInspectionState, SafetyContractV3

CONFIG = Path("configs/safety/safety_contract_v3.yaml")


def valid_inspection(**overrides) -> ControlledInspectionState:
    values = {
        "target_recognized": True,
        "inspection_intent": True,
        "safe_approach": True,
        "speed": 0.0,
        "human_clearance_cells": 2.0,
        "dwell_steps": 1,
        "retreat_route_available": True,
        "shield_active": True,
    }
    values.update(overrides)
    return ControlledInspectionState(**values)


def test_contract_loads_and_preserves_legacy_label() -> None:
    contract = SafetyContractV3.from_yaml(CONFIG)
    contract.add_legacy(raw_cost=2.5, constraint_count=3)
    summary = contract.summary()
    assert summary["legacy_raw_safety_cost"] == 2.5
    assert summary["legacy_constraint_count"] == 3
    assert summary["legacy_diagnostic_label"] == "LEGACY_INACTIVITY_BIASED_DIAGNOSTIC"


def test_controlled_inspection_suppresses_only_semantic_vector_cost() -> None:
    contract = SafetyContractV3.from_yaml(CONFIG)
    vector = contract.vector_cost(
        collision=True,
        restricted_zone=False,
        human_clearance_breach=False,
        semantic_risk=0.8,
        uncertainty=0.2,
        inspection=valid_inspection(),
    )
    assert vector.collision == 1.0
    assert vector.uncontrolled_semantic_risk == 0.0
    assert vector.uncertainty == 0.2


def test_uncontrolled_or_excessive_dwell_remains_penalized() -> None:
    contract = SafetyContractV3.from_yaml(CONFIG)
    vector = contract.vector_cost(
        collision=False,
        restricted_zone=False,
        human_clearance_breach=False,
        semantic_risk=0.7,
        uncertainty=0.0,
        inspection=valid_inspection(dwell_steps=4),
    )
    assert vector.uncontrolled_semantic_risk == 0.7


def test_event_duration_and_unique_count_are_separate() -> None:
    contract = SafetyContractV3.from_yaml(CONFIG)
    for step in (1, 2, 3):
        contract.add_legacy(raw_cost=1.0, constraint_count=1)
        contract.update_events(
            episode_id="episode",
            step=step,
            position=(2, 2),
            active={"restricted_zone_entry": ("hard", 1.0, False)},
            proposed_action=3,
            executed_action=3,
            shield_decision="accept",
            inspection_intent=False,
        )
    contract.update_events(
        episode_id="episode",
        step=4,
        position=(2, 1),
        active={},
        proposed_action=2,
        executed_action=2,
        shield_decision="accept",
        inspection_intent=False,
    )
    summary = contract.summary()
    assert summary["legacy_constraint_count"] == 3
    assert summary["hard_violations_per_episode"] == 1
    assert summary["events"][0]["duration"] == 3
    assert summary["events"][0]["integrated_severity"] == 3.0


def test_failed_mission_does_not_claim_success_conditioned_cost() -> None:
    contract = SafetyContractV3.from_yaml(CONFIG)
    contract.update_events(
        episode_id="episode",
        step=1,
        position=(1, 1),
        active={"near_miss": ("soft", 0.5, False)},
        proposed_action=0,
        executed_action=0,
        shield_decision="accept",
        inspection_intent=False,
    )
    contract.finish_episode(success=False, mission_progress=0.5, resolution_action=2)
    summary = contract.summary()
    assert summary["success_conditioned_safety_cost_v3"] is None
    assert summary["mission_progress_normalized_cost_v3"] == 1.0
