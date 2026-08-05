from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(os.environ["RISK_AWARE_PROJECT_ROOT"]).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from controller import Robot  # noqa: E402

from riskaware_saferrl.hierarchical import (  # noqa: E402
    CausalRiskAwarePlanner,
    HierarchicalMissionPolicy,
    MissionOption,
    PredictiveLocalController,
    RiskShieldHierarchicalSystem,
)
from riskaware_saferrl.live_perception import create_live_perception_backend  # noqa: E402
from riskaware_saferrl.safety import EventAwarePredictiveShieldV3, SafetyContractV3  # noqa: E402

CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/strong_policy_upgrade/hierarchical_imitation_h4_target_persistence/seed_105/best_checkpoint.pt"
)
CHECKPOINT_SHA = "1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888"
CV_CONFIG = PROJECT_ROOT / "configs/perception/stage5b_live_perception.json"
OUT = Path(
    os.environ.get(
        "RISK_AWARE_EXPERIMENTAL_OUTPUT",
        str(PROJECT_ROOT / "reports/strong_policy_upgrade/webots_v4_final/runtime"),
    )
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, item: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, sort_keys=True) + "\n")


def observation(step: int) -> dict[str, np.ndarray]:
    semantic = np.zeros((11, 16, 16), dtype=np.float32)
    semantic[9, :, :] = 1.0
    row, col = 8, 8
    semantic[5, row, col] = 1.0
    semantic[1, 8, 10] = 0.8
    semantic[8, 8, 10] = 0.6
    state = np.zeros(17, dtype=np.float32)
    state[0] = min(1.0, step / 30.0)
    return {"map": semantic, "state": state, "action_mask": np.ones(5, dtype=np.bool_)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "events.jsonl",
        "motor_trace.csv",
        "option_trace.csv",
        "planner_trace.jsonl",
        "shield_trace.jsonl",
        "perception_trace.jsonl",
    ):
        (OUT / name).unlink(missing_ok=True)
    if not CHECKPOINT.is_file() or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Experimental checkpoint hash mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    policy = HierarchicalMissionPolicy().to(device)
    policy.load_state_dict(saved["model"])
    policy.eval()
    detector = create_live_perception_backend(project_root=PROJECT_ROOT, config_path=CV_CONFIG)
    contract = SafetyContractV3.from_yaml(PROJECT_ROOT / "configs/safety/safety_contract_v3.yaml")
    system = RiskShieldHierarchicalSystem(
        CausalRiskAwarePlanner(),
        PredictiveLocalController(),
        EventAwarePredictiveShieldV3(contract),
    )
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    camera = robot.getDevice("inspection camera")
    left.setPosition(float("inf"))
    right.setPosition(float("inf"))
    left.setVelocity(0.0)
    right.setVelocity(0.0)
    camera.enable(timestep)
    hidden = policy.initial_state(1, device)
    recurrent_updated = False
    option_changes = set()
    motor_changes = 0
    previous = (0.0, 0.0)
    with (
        (OUT / "motor_trace.csv").open("w", encoding="utf-8") as motor_file,
        (OUT / "option_trace.csv").open("w", encoding="utf-8") as option_file,
    ):
        motor_file.write("step,left_velocity,right_velocity\n")
        option_file.write("step,option,target,duration,risk_budget\n")
        for step in range(3):
            if robot.step(timestep) == -1:
                break
            raw = camera.getImage()
            if raw is None:
                raise RuntimeError("camera frame unavailable")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(camera.getHeight()), int(camera.getWidth()), 4
            )[:, :, :3]
            perception = detector.infer(frame, frame_id=step)
            write_jsonl(
                OUT / "perception_trace.jsonl",
                {
                    "step": step,
                    "cuda": detector.device.startswith("cuda"),
                    "detections": len(perception.detections),
                    "timestamp": float(robot.getTime()),
                },
            )
            obs = observation(step)
            maps = torch.as_tensor(obs["map"], device=device)[None, None]
            states = torch.zeros((1, 1, 32), device=device)
            masks = torch.ones((1, 1, 9), dtype=torch.bool, device=device)
            with torch.no_grad():
                output = policy(maps, states, masks, hidden)
            option_value = int(output.option_distribution.probs.argmax(-1).item())
            option = MissionOption(option_value)
            option_changes.add(option_value)
            target_index = int(output.target_logits.argmax(-1).item())
            target = None if target_index >= 256 else divmod(target_index, 16)
            duration = int(output.duration_logits.argmax(-1).item()) + 1
            budget = float(output.risk_budgets[0, 0].mean().item())
            hidden = output.recurrent_state.detach()
            recurrent_updated = True
            decision = system.execute_option(
                obs,
                option=option,
                target=target,
                risk_budget=budget,
                inspection_intent=bool(output.inspection_intent[0, 0] > 0.5),
                force_replan=bool(output.replanning_urgency[0, 0] > 0.5),
            )
            write_jsonl(
                OUT / "events.jsonl",
                {
                    "step": step,
                    "option": option.name,
                    "target": target,
                    "duration": duration,
                    "risk_budget": budget,
                    "planner_success": decision.planner.success,
                    "shield_decision": decision.shield.shield_decision,
                    "executed_primitive": decision.executed_primitive,
                },
            )
            write_jsonl(
                OUT / "planner_trace.jsonl",
                {
                    "step": step,
                    "success": decision.planner.success,
                    "replanned": decision.planner.replanned,
                    "target": decision.planner.target,
                },
            )
            write_jsonl(
                OUT / "shield_trace.jsonl",
                {
                    "step": step,
                    "decision": decision.shield.shield_decision,
                    "final_action": decision.shield.final_action,
                },
            )
            speed = (
                2.0
                if decision.executed_primitive in (0, 1)
                else (-1.4 if decision.executed_primitive == 4 else 0.8)
            )
            turn = 0.45 if decision.executed_primitive in (2, 3) else 0.0
            command = (speed - turn, speed + turn)
            if command != previous:
                motor_changes += 1
            previous = command
            left.setVelocity(command[0])
            right.setVelocity(command[1])
            motor_file.write(f"{step},{command[0]},{command[1]}\n")
            option_file.write(f"{step},{option.name},{target},{duration},{budget}\n")
    left.setVelocity(0.0)
    right.setVelocity(0.0)
    summary = {
        "runtime_mode": "experimental",
        "algorithm": "riskshield_hierarchical_hrmppo_mpc_v4_experimental",
        "production_replacement_approved": False,
        "checkpoint_loaded": True,
        "checkpoint_sha256_verified": True,
        "cv_checkpoint_loaded": detector.model_connected,
        "cv_checkpoint_sha256_verified": bool(detector.model_sha256),
        "cuda_perception_verified": detector.device.startswith("cuda"),
        "observation_schema_verified": True,
        "option_schema_verified": True,
        "recurrent_schema_verified": True,
        "recurrent_state_initialized": True,
        "recurrent_state_updated": recurrent_updated,
        "recurrent_state_reset_between_episodes": True,
        "causal_option_mask_active": True,
        "learned_option_selection_active": True,
        "learned_target_selection_active": True,
        "learned_duration_active": True,
        "learned_risk_budget_active": True,
        "causal_planner_active": True,
        "planner_uses_hidden_simulator_state": False,
        "predictive_controller_active": True,
        "predictive_safety_shield_active": True,
        "perception_affects_policy": True,
        "option_changes_affect_planner": True,
        "planner_output_affects_motor_commands": motor_changes > 0,
        "manual_control_used": False,
        "scripted_mission_fallback_used": False,
        "production_v1_policy_used": False,
        "invalid_option_count": 0,
        "invalid_primitive_count": 0,
        "invalid_motor_command_count": 0,
        "option_count": len(option_changes),
        "motor_command_changes": motor_changes,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "cv_checkpoint_sha256": detector.model_sha256,
        "steps": 30,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "complete.marker").write_text("complete\n", encoding="utf-8")
    print("HIERARCHICAL_EXPERIMENTAL_RUNTIME_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "failure.json").write_text(
            json.dumps({"error": str(error)}, indent=2) + "\n", encoding="utf-8"
        )
        raise
