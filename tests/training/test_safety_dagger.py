from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from riskaware_saferrl.safety import (
    EventAwarePredictiveShieldV3,
    SafetyContractV3,
)
from riskaware_saferrl.training.safety_dagger import (
    SafetyCorrectionWriter,
    select_safety_state,
)


def observation() -> dict[str, np.ndarray]:
    semantic_map = np.zeros((11, 16, 16), dtype=np.float32)
    semantic_map[5, 4, 4] = 1
    return {
        "map": semantic_map,
        "state": np.zeros(17, dtype=np.float32),
        "action_mask": np.ones(5, dtype=np.int8),
    }


def test_selector_uses_proposed_risk_and_disagreement() -> None:
    shield = EventAwarePredictiveShieldV3(
        SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    )
    decision = shield.decide(observation(), 0)
    risky = replace(
        decision,
        proposed_vector_cost=replace(decision.proposed_vector_cost, worker_near_miss=1.0),
    )
    selected = select_safety_state(
        policy_action=0,
        teacher_action=1,
        policy_shield=risky,
        recent_positions=((4, 4),),
        include_safe_example=False,
    )
    assert selected.selected
    assert "teacher_disagreement" in selected.reasons
    assert "pre_worker_near_miss" in selected.reasons


def test_safe_state_requires_explicit_anchor_sampling() -> None:
    decision = EventAwarePredictiveShieldV3(
        SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    ).decide(observation(), 0)
    omitted = select_safety_state(
        policy_action=0,
        teacher_action=0,
        policy_shield=decision,
        recent_positions=((4, 4),),
        include_safe_example=False,
    )
    included = select_safety_state(
        policy_action=0,
        teacher_action=0,
        policy_shield=decision,
        recent_positions=((4, 4),),
        include_safe_example=True,
    )
    assert not omitted.selected
    assert included.reasons == ("task_anchor",)


def test_writer_rejects_invalid_teacher_and_writes_hash(tmp_path) -> None:
    writer = SafetyCorrectionWriter(tmp_path, chunk_size=1)
    record = {
        "map": observation()["map"],
        "state": observation()["state"],
        "action_mask": np.asarray([1, 1, 1, 1, 1]),
        "teacher_action": 1,
        "policy_action": 0,
        "shield_action": 1,
        "episode_id": "episode",
        "selection_reasons": ("teacher_disagreement",),
        "selection_priority": 2.0,
        "predicted_vector_cost": {
            "collision": 0.0,
            "restricted_zone": 0.0,
            "worker_near_miss": 0.0,
            "uncontrolled_semantic_risk": 0.0,
            "uncertainty": 0.0,
        },
    }
    writer.add(record)
    assert writer.chunks[0]["rows"] == 1
    assert len(writer.chunks[0]["sha256"]) == 64
    record["action_mask"] = np.asarray([1, 0, 1, 1, 1])
    try:
        writer.add(record)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid teacher action was accepted")
