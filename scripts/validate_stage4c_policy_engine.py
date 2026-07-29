from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from riskaware_saferrl.webots import (
    BridgeState,
    ObservationBridge,
    SemanticScene,
)
from riskaware_saferrl.webots.policy_dry_run import (
    PolicyDryRunEngine,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "maskable_ppo_deadlock_safe_shield_seed42_u100"
    / "evaluations"
    / "best_model"
    / "best_model.zip"
)

REPORT_PATH = PROJECT_ROOT / "webots" / "logs" / "stage4c_policy_engine_audit.json"

EXPECTED_SHA256 = "172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7"


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


engine = PolicyDryRunEngine.from_checkpoint(
    CHECKPOINT_PATH,
    expected_sha256=EXPECTED_SHA256,
    device="cpu",
    seed=42,
)

scene = SemanticScene(
    size=12,
    obstacles=frozenset(
        {
            (5, 6),
            (7, 5),
        }
    ),
    hazards=frozenset(
        {
            (6, 8),
            (4, 7),
        }
    ),
    workers=frozenset(
        {
            (7, 7),
        }
    ),
    restricted=frozenset(
        {
            (6, 4),
        }
    ),
)

observation_bridge = ObservationBridge(
    vision_radius=4,
    inspection_radius=2,
)

audit_positions = (
    (6, 6),
    (6, 5),
    (5, 5),
    (4, 5),
    (4, 6),
    (4, 7),
    (5, 7),
    (6, 7),
    (6, 8),
    (7, 8),
    (8, 8),
    (8, 7),
)

visited: set[tuple[int, int]] = set()
proposals: list[dict[str, object]] = []

masked_state_count = 0
inspect_enabled_state_count = 0

for state_index, position in enumerate(audit_positions):
    visited.add(position)

    bridge_state = BridgeState(
        agent_position=position,
        visited=frozenset(visited),
        inspected=frozenset(),
        steps=state_index,
        max_steps=250,
    )

    observation = observation_bridge.build_observation(
        scene,
        bridge_state,
    )

    action_mask = observation_bridge.action_mask(
        scene,
        bridge_state,
    )

    if not bool(np.all(action_mask)):
        masked_state_count += 1

    if bool(action_mask[4]):
        inspect_enabled_state_count += 1

    modes = (
        ("deterministic", True),
        ("stochastic_1", False),
        ("stochastic_2", False),
        ("stochastic_3", False),
    )

    for mode_name, deterministic in modes:
        proposal = engine.propose(
            sample_index=len(proposals),
            observation=observation,
            action_mask=action_mask,
            deterministic=deterministic,
        )

        proposal_payload = proposal.to_dict()

        proposal_payload["mode"] = mode_name

        proposal_payload["agent_position"] = list(position)

        proposals.append(proposal_payload)

require(
    len(proposals) == 48,
    (f"Expected 48 policy proposals. Received {len(proposals)}."),
)

require(
    masked_state_count > 0,
    "No partially masked state was tested.",
)

require(
    inspect_enabled_state_count > 0,
    "No INSPECT-enabled state was tested.",
)

require(
    all(bool(proposal["mask_respected"]) for proposal in proposals),
    "A policy proposal violated the mask.",
)

require(
    all(not bool(proposal["motors_connected"]) for proposal in proposals),
    "A policy proposal was connected to motors.",
)

unique_actions = sorted({int(proposal["action"]) for proposal in proposals})

summary = {
    "audit_schema_version": 1,
    "algorithm_class": (engine.algorithm_class),
    "policy_class": (engine.policy_class),
    "device": engine.device,
    "checkpoint_path": (engine.checkpoint_path),
    "checkpoint_sha256": (engine.checkpoint_hash),
    "tested_state_count": len(audit_positions),
    "masked_state_count": (masked_state_count),
    "inspect_enabled_state_count": (inspect_enabled_state_count),
    "proposal_count": len(proposals),
    "invalid_proposal_count": 0,
    "unique_proposed_actions": (unique_actions),
    "all_masks_respected": True,
    "motors_connected": False,
    "policy_engine_verified": True,
    "proposals": proposals,
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print(
    json.dumps(
        {key: value for key, value in summary.items() if key != "proposals"},
        indent=2,
        sort_keys=True,
    )
)
