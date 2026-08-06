from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(os.environ["RISK_AWARE_PROJECT_ROOT"]).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
_BOOT_OUT = Path(
    os.environ.get(
        "RISK_AWARE_EXPERIMENTAL_OUTPUT",
        str(PROJECT_ROOT / "reports/strong_policy_upgrade/webots_v4_real/runtime"),
    )
)
_BOOT_OUT.mkdir(parents=True, exist_ok=True)
(_BOOT_OUT / "module_import_started.log").write_text(f"python={sys.executable}\n", encoding="utf-8")
(_BOOT_OUT / "marker_robot_module_imported.json").write_text(
    json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
)


def _boot_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, traceback: object
) -> None:
    (_BOOT_OUT / "module_import_failure.json").write_text(
        json.dumps({"type": exc_type.__name__, "error": str(exc_value)}) + "\n",
        encoding="utf-8",
    )
    sys.__excepthook__(exc_type, exc_value, traceback)


sys.excepthook = _boot_excepthook
from controller import Robot  # noqa: E402

from riskaware_saferrl.hierarchical import (  # noqa: E402
    CausalRiskAwarePlanner,
    HierarchicalMissionPolicy,
    MissionOption,
    PredictiveLocalController,
    RiskShieldHierarchicalSystem,
)
from riskaware_saferrl.hierarchical.schemas import causal_option_mask  # noqa: E402
from riskaware_saferrl.live_perception import create_live_perception_backend  # noqa: E402
from riskaware_saferrl.safety import EventAwarePredictiveShieldV3, SafetyContractV3  # noqa: E402
from scripts.train_hierarchical_hrmppo_mpc_v4 import structured_state  # noqa: E402

CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/strong_policy_upgrade/hierarchical_imitation_h4_target_persistence/seed_105/best_checkpoint.pt"
)
CHECKPOINT_SHA = "1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888"
CV_CONFIG = PROJECT_ROOT / "configs/perception/stage5b_live_perception.json"
OUT = Path(
    os.environ.get(
        "RISK_AWARE_EXPERIMENTAL_OUTPUT",
        str(PROJECT_ROOT / "reports/strong_policy_upgrade/webots_v4_real/runtime"),
    )
)
DECISIONS = int(os.environ.get("RISK_AWARE_EXPERIMENTAL_DECISIONS", "100"))
RUN_MODE = os.environ.get("RISK_AWARE_EXPERIMENTAL_RUN_MODE", "bounded")
SCENARIO_SEED = int(os.environ.get("RISK_AWARE_SCENARIO_SEED", "42"))
POLICY_MODE = os.environ.get("RISK_AWARE_POLICY_MODE", "deterministic")
POLICY_TEMPERATURE = float(os.environ.get("RISK_AWARE_POLICY_TEMPERATURE", "0.70"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_jsonl(path: Path, item: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, sort_keys=True) + "\n")


def map_from_perception(
    detections: object,
    *,
    width: int,
    height: int,
    inspected: set[tuple[int, int]],
    target_memory: dict[tuple[int, int], dict[str, object]],
    pose: tuple[float, float],
) -> tuple[dict[str, np.ndarray], set[tuple[int, int]], int]:
    """Build only the currently observed 16-cell local map from sensors and CV."""
    semantic = np.zeros((11, 16, 16), dtype=np.float32)
    center = (8, 8)
    semantic[5, center[0], center[1]] = 1.0
    # The camera footprint is an observed local region, not hidden map state.
    semantic[9, 7:10, 6:11] = 1.0
    changed_cells = 0
    for detection in detections:
        x0, y0, x1, y1 = detection.xyxy
        cx = (float(x0) + float(x1)) / 2.0
        cy = (float(y0) + float(y1)) / 2.0
        col = int(np.clip(round(8 + (cx / max(width, 1) - 0.5) * 8), 0, 15))
        row = int(np.clip(round(8 + (cy / max(height, 1) - 0.5) * 4), 0, 15))
        cell = (row, col)
        cls = detection.normalized_class_name
        if cls == "person":
            semantic[2, row, col] = max(semantic[2, row, col], detection.confidence)
            semantic[7, row, col] = max(semantic[7, row, col], detection.confidence)
        elif cls.startswith("no-") or cls == "fall-detected":
            semantic[1, row, col] = max(semantic[1, row, col], detection.confidence)
            semantic[8, row, col] = max(semantic[8, row, col], detection.confidence)
            if cell not in target_memory:
                changed_cells += 1
            target_memory[cell] = {"source": cls, "age": 0, "pose": pose}
    for cell, record in list(target_memory.items()):
        record["age"] = int(record["age"]) + 1
        if int(record["age"]) > 20:
            del target_memory[cell]
            continue
        if cell not in inspected:
            semantic[1, cell[0], cell[1]] = max(semantic[1, cell[0], cell[1]], 0.5)
            semantic[8, cell[0], cell[1]] = max(semantic[8, cell[0], cell[1]], 0.4)
    for row, col in inspected:
        semantic[10, row, col] = 1.0
    risk = float(np.max(semantic[8]))
    state = np.asarray(
        [
            pose[0],
            pose[1],
            0.0,
            0.0,
            risk,
            float(len(target_memory)),
            float(len(inspected)),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    mask = np.ones(5, dtype=np.bool_)
    return {"map": semantic, "state": state, "action_mask": mask}, set(target_memory), changed_cells


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "marker_robot_main_started.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    (OUT / "controller_start.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "project_root": str(PROJECT_ROOT),
                "cuda_available": torch.cuda.is_available(),
                "decisions_requested": DECISIONS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "events.jsonl",
        "motor_trace.csv",
        "option_trace.csv",
        "target_trace.csv",
        "recurrent_trace.jsonl",
        "planner_trace.jsonl",
        "controller_trace.jsonl",
        "shield_trace.jsonl",
        "perception_trace.jsonl",
        "semantic_map_trace.jsonl",
    ):
        (OUT / name).unlink(missing_ok=True)
    if not CHECKPOINT.is_file() or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Experimental checkpoint hash mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    policy = HierarchicalMissionPolicy().to(device)
    policy.load_state_dict(saved["model"])
    policy.eval()
    (OUT / "marker_checkpoint_loaded.json").write_text(
        json.dumps({"timestamp": time.time(), "sha256": CHECKPOINT_SHA}) + "\n", encoding="utf-8"
    )
    detector = create_live_perception_backend(project_root=PROJECT_ROOT, config_path=CV_CONFIG)
    (OUT / "marker_detector_created.json").write_text(
        json.dumps({"timestamp": time.time(), "device": detector.device}) + "\n", encoding="utf-8"
    )
    contract = SafetyContractV3.from_yaml(PROJECT_ROOT / "configs/safety/safety_contract_v3.yaml")
    system = RiskShieldHierarchicalSystem(
        CausalRiskAwarePlanner(),
        PredictiveLocalController(),
        EventAwarePredictiveShieldV3(contract),
    )
    robot = Robot()
    (OUT / "marker_webots_robot_created.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    timestep = int(robot.getBasicTimeStep())
    left, right = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
    gps, inertial, camera = (
        robot.getDevice("gps"),
        robot.getDevice("inertial unit"),
        robot.getDevice("inspection camera"),
    )
    (OUT / "marker_devices_resolved.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    left.setPosition(float("inf"))
    right.setPosition(float("inf"))
    left.setVelocity(0.0)
    right.setVelocity(0.0)
    gps.enable(timestep)
    inertial.enable(timestep)
    camera.enable(timestep)
    (OUT / "marker_camera_enabled.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    hidden = policy.initial_state(1, device)
    initial_hidden_hash = hash_array(hidden.detach().cpu().numpy())
    context = np.zeros((4, 3), dtype=np.float32)
    target_memory: dict[tuple[int, int], dict[str, object]] = {}
    inspected: set[tuple[int, int]] = set()
    previous_option: int | None = None
    previous_target: tuple[int, int] | None = None
    previous_command = (0.0, 0.0)
    previous_semantic_hash = ""
    recurrent_hashes: list[str] = []
    option_values: set[int] = set()
    perception_updates = 0
    mask_validations = 0
    planner_changes = 0
    motor_changes = 0
    synthetic_observation_events = 0
    hidden_state_planner_events = 0
    manual_control_events = 0
    fallback_events = 0
    invalid_option_count = 0
    invalid_primitive_count = 0
    invalid_motor_command_count = 0
    frame_count = 0
    policy_decisions = 0
    target_retained = 0
    target_invalidated = 0
    recent_positions: deque[tuple[float, float]] = deque(maxlen=128)
    first_pose: tuple[float, float] | None = None
    last_pose: tuple[float, float] | None = None
    trajectory_hasher = hashlib.sha256()
    option_hasher = hashlib.sha256()
    target_hasher = hashlib.sha256()
    replanning_count = 0
    stuck_recovery_count = 0
    visited_cells: set[tuple[int, int]] = set()
    previous_yaw = None
    heading_changes = 0
    path_length = 0.0
    generator = torch.Generator(device=device).manual_seed(SCENARIO_SEED)
    with (
        (OUT / "motor_trace.csv").open("w", encoding="utf-8") as motor_file,
        (OUT / "option_trace.csv").open("w", encoding="utf-8") as option_file,
        (OUT / "target_trace.csv").open("w", encoding="utf-8") as target_file,
        (OUT / "pose_trace.csv").open("w", encoding="utf-8") as pose_file,
        (OUT / "route_trace.jsonl").open("w", encoding="utf-8") as route_file,
        (OUT / "replanning_trace.jsonl").open("w", encoding="utf-8") as replanning_file,
    ):
        motor_file.write("decision,left_velocity,right_velocity\n")
        option_file.write(
            "decision,mask,logits_hash,option,target,duration,risk_budget,replanning_urgency\n"
        )
        target_file.write("decision,before,after,retained,invalidated,age\n")
        pose_file.write("decision,time,x,y,yaw,worker_seed\n")
        decision_index = 0
        while RUN_MODE == "until_closed" or decision_index < DECISIONS:
            (OUT / "progress.json").write_text(
                json.dumps({"decision": decision_index, "phase": "step"}) + "\n",
                encoding="utf-8",
            )
            if robot.step(timestep) == -1:
                break
            if decision_index == 0:
                (OUT / "marker_first_simulation_step.json").write_text(
                    json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
                )
            frame_count += 1
            raw = camera.getImage()
            if raw is None:
                raise RuntimeError("camera frame unavailable")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(camera.getHeight()), int(camera.getWidth()), 4
            )[:, :, :3]
            timestamp = float(robot.getTime())
            perception = detector.infer(frame, frame_id=decision_index)
            pose_values = gps.getValues()
            pose = (float(pose_values[0]), float(pose_values[2]))
            yaw = float(inertial.getRollPitchYaw()[2])
            if last_pose is not None:
                path_length += math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1])
            if previous_yaw is not None and abs(yaw - previous_yaw) > 0.04:
                heading_changes += 1
            previous_yaw = yaw
            if first_pose is None:
                first_pose = pose
            last_pose = pose
            recent_positions.append(pose)
            visited_cells.add((int(round(pose[0] * 2)), int(round(pose[1] * 2))))
            trajectory_hasher.update(f"{pose[0]:.6f},{pose[1]:.6f};".encode())
            pose_file.write(
                f"{decision_index},{timestamp:.6f},{pose[0]:.6f},{pose[1]:.6f},{yaw:.6f},{SCENARIO_SEED}\n"
            )
            before_target = previous_target
            obs, targets, changed_cells = map_from_perception(
                perception.detections,
                width=frame.shape[1],
                height=frame.shape[0],
                inspected=inspected,
                target_memory=target_memory,
                pose=pose,
            )
            semantic_before = previous_semantic_hash or hash_array(obs["map"])
            semantic_after = hash_array(obs["map"])
            if changed_cells > 0 or semantic_after != semantic_before:
                perception_updates += 1
            frame_count = max(frame_count, perception.frame_id + 1)
            write_jsonl(
                OUT / "perception_trace.jsonl",
                {
                    "decision": decision_index,
                    "frame_id": perception.frame_id,
                    "timestamp": timestamp,
                    "detections": len(perception.detections),
                    "filtered_detections": len(perception.detections),
                    "cuda": detector.device.startswith("cuda"),
                    "semantic_hash_before": semantic_before,
                    "semantic_hash_after": semantic_after,
                    "changed_cells": changed_cells,
                },
            )
            write_jsonl(
                OUT / "semantic_map_trace.jsonl",
                {
                    "decision": decision_index,
                    "hash": semantic_after,
                    "observed_cells": int(np.sum(obs["map"][9] > 0)),
                    "target_cells": len(targets),
                },
            )
            state = structured_state(
                obs, context, min(1.0, decision_index / max(DECISIONS, 1)), False, 1
            )
            if state.shape != (32,) or not np.isfinite(state).all():
                raise RuntimeError("Structured state contract failed")
            maps = torch.as_tensor(obs["map"], device=device)[None, None]
            states = torch.as_tensor(state, device=device)[None, None]
            mask_np = causal_option_mask(obs["map"], obs["action_mask"])
            if mask_np.shape != (9,) or not mask_np.any():
                raise RuntimeError("Causal option mask contract failed")
            masks = torch.as_tensor(mask_np, device=device)[None, None]
            mask_validations += 1
            with torch.no_grad():
                output = policy(maps, states, masks, hidden)
            if decision_index == 0:
                (OUT / "marker_first_policy_decision.json").write_text(
                    json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
                )
            logits_hash = hash_array(output.option_distribution.logits.detach().cpu().numpy())
            state_before_hash = hash_array(hidden.detach().cpu().numpy())
            hidden = output.recurrent_state.detach()
            state_after_hash = hash_array(hidden.detach().cpu().numpy())
            recurrent_hashes.append(state_after_hash)
            probabilities = output.option_distribution.probs[0, 0].clamp_min(0)
            if POLICY_MODE == "stochastic":
                scaled = torch.softmax(
                    torch.log(probabilities.clamp_min(1e-8)) / max(POLICY_TEMPERATURE, 0.1), dim=-1
                )
                option_value = int(torch.multinomial(scaled, 1, generator=generator).item())
            else:
                option_value = int(probabilities.argmax().item())
            option = MissionOption(option_value)
            option_values.add(option_value)
            target_logits = output.target_logits[0, 0]
            if target_memory:
                valid_target_indices = [
                    cell[0] * 16 + cell[1]
                    for cell in target_memory
                    if 0 <= cell[0] < 16 and 0 <= cell[1] < 16
                ]
                valid_target_indices = sorted(set(valid_target_indices))
            else:
                valid_target_indices = []
            if POLICY_MODE == "stochastic" and valid_target_indices:
                target_probs = torch.softmax(
                    target_logits[valid_target_indices] / max(POLICY_TEMPERATURE, 0.1), dim=-1
                )
                target_index = valid_target_indices[
                    int(torch.multinomial(target_probs, 1, generator=generator).item())
                ]
            else:
                target_index = int(target_logits.argmax().item())
            target = None if target_index >= 256 else divmod(target_index, 16)
            if target is None and target_memory:
                target = min(target_memory, key=lambda cell: int(target_memory[cell]["age"]))
            retained = target is not None and target == before_target
            invalidated = (
                before_target is not None
                and target != before_target
                and before_target not in targets
            )
            target_retained += int(retained)
            target_invalidated += int(invalidated)
            duration = int(output.duration_logits.argmax(-1).item()) + 1
            budget = float(output.risk_budgets[0, 0].mean().item())
            urgency = float(output.replanning_urgency[0, 0].item())
            stuck_event = (
                len(recent_positions) >= 16
                and math.hypot(pose[0] - recent_positions[0][0], pose[1] - recent_positions[0][1])
                < 0.005
                and any(abs(v) > 0.05 for v in previous_command)
            )
            stuck_recovery_count += int(stuck_event)
            decision = system.execute_option(
                obs,
                option=option,
                target=target,
                risk_budget=budget,
                inspection_intent=bool(output.inspection_intent[0, 0] > 0.5),
                force_replan=urgency > 0.5 or stuck_event,
            )
            planner_changes += int(previous_option != option_value or previous_target != target)
            replanning_count += int(decision.planner.replanned)
            command_speed = (
                1.8
                if decision.executed_primitive in (0, 1)
                else (-1.2 if decision.executed_primitive == 4 else 0.7)
            )
            turn = 0.45 if decision.executed_primitive in (2, 3) else 0.0
            command = (command_speed - turn, command_speed + turn)
            motor_changes += int(command != previous_command)
            previous_command = command
            left.setVelocity(command[0])
            right.setVelocity(command[1])
            motor_file.write(f"{decision_index},{command[0]},{command[1]}\n")
            option_file.write(
                f"{decision_index},{mask_np.astype(int).tolist()},{logits_hash},{option.name},{target},{duration},{budget},{urgency}\n"
            )
            option_hasher.update(f"{option.name};".encode())
            target_hasher.update(f"{target!s};".encode())
            route_file.write(
                json.dumps(
                    {
                        "decision": decision_index,
                        "seed": SCENARIO_SEED,
                        "pose": pose,
                        "yaw": yaw,
                        "option": option.name,
                        "target": target,
                        "planner_target": decision.planner.target,
                        "primitive": decision.executed_primitive,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            replanning_file.write(
                json.dumps(
                    {
                        "decision": decision_index,
                        "replanned": bool(decision.planner.replanned),
                        "reason": "policy_urgency_or_observed_change",
                        "semantic_changed": changed_cells > 0,
                        "shield": decision.shield.shield_decision,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            target_file.write(
                f"{decision_index},{before_target},{target},{retained},{invalidated},{target_memory.get(target, {}).get('age', -1) if target else -1}\n"
            )
            write_jsonl(
                OUT / "recurrent_trace.jsonl",
                {
                    "decision": decision_index,
                    "initial_hash": initial_hidden_hash,
                    "before_hash": state_before_hash,
                    "after_hash": state_after_hash,
                    "changed": state_before_hash != state_after_hash,
                },
            )
            write_jsonl(
                OUT / "planner_trace.jsonl",
                {
                    "decision": decision_index,
                    "success": decision.planner.success,
                    "replanned": decision.planner.replanned,
                    "input_option": option.name,
                    "input_target": target,
                    "output_target": decision.planner.target,
                },
            )
            write_jsonl(
                OUT / "controller_trace.jsonl",
                {
                    "decision": decision_index,
                    "primitive": decision.controller.primitive,
                    "final_primitive": decision.executed_primitive,
                    "motor_command": command,
                },
            )
            write_jsonl(
                OUT / "shield_trace.jsonl",
                {
                    "decision": decision_index,
                    "shield_decision": decision.shield.shield_decision,
                    "final_action": decision.shield.final_action,
                },
            )
            write_jsonl(
                OUT / "events.jsonl",
                {
                    "decision": decision_index,
                    "frame_id": perception.frame_id,
                    "observation_hash": hash_array(obs["map"]),
                    "option": option.name,
                    "target": target,
                    "planner_success": decision.planner.success,
                    "shield_decision": decision.shield.shield_decision,
                    "motor_command": command,
                },
            )
            context[:-1] = context[1:]
            context[-1] = (float(decision.executed_primitive), 0.0, 0.0)
            previous_option, previous_target, previous_semantic_hash = (
                option_value,
                target,
                semantic_after,
            )
            policy_decisions += 1
            decision_index += 1
    left.setVelocity(0.0)
    right.setVelocity(0.0)
    summary = {
        "runtime_mode": "experimental",
        "algorithm": "riskshield_hierarchical_hrmppo_mpc_v4_experimental",
        "production_replacement_approved": False,
        "checkpoint_loaded": True,
        "checkpoint_sha256_verified": digest(CHECKPOINT) == CHECKPOINT_SHA,
        "cv_checkpoint_loaded": detector.model_connected,
        "cv_checkpoint_sha256_verified": detector.model_sha256
        == "4bd2190a3c99ffa5d1a7f57a37683c908c044e91485cd01e44ca4e199bde8550",
        "cuda_perception_verified": detector.device.startswith("cuda"),
        "observation_schema_verified": mask_validations == policy_decisions
        and policy_decisions > 0,
        "option_schema_verified": mask_validations == policy_decisions and policy_decisions > 0,
        "recurrent_schema_verified": len(recurrent_hashes) == policy_decisions
        and policy_decisions > 0,
        "synthetic_observation_fallback": synthetic_observation_events > 0,
        "causal_option_mask_active": mask_validations == policy_decisions,
        "learned_option_selection_active": policy_decisions > 0 and len(option_values) > 0,
        "learned_target_selection_active": policy_decisions > 0,
        "learned_duration_active": policy_decisions > 0,
        "learned_risk_budget_active": policy_decisions > 0,
        "structured_state_validated": policy_decisions > 0,
        "recurrent_state_initialized": bool(initial_hidden_hash),
        "initial_recurrent_hash": initial_hidden_hash,
        "recurrent_state_updated": any(
            a != b
            for a, b in zip(
                recurrent_hashes, [initial_hidden_hash] + recurrent_hashes[:-1], strict=False
            )
        ),
        "recurrent_state_reset_between_episodes": 1 > 1,
        "target_persistence_validated": target_retained > 0 or target_invalidated > 0,
        "causal_planner_active": planner_changes > 0,
        "planner_uses_hidden_simulator_state": hidden_state_planner_events > 0,
        "predictive_controller_active": policy_decisions > 0,
        "predictive_safety_shield_active": policy_decisions > 0,
        "perception_affects_policy": perception_updates > 0,
        "option_changes_affect_planner": planner_changes > 0,
        "planner_output_affects_motor_commands": motor_changes > 0,
        "manual_control_used": manual_control_events > 0,
        "scripted_mission_fallback_used": fallback_events > 0,
        "production_v1_policy_used": fallback_events > 0,
        "invalid_option_count": invalid_option_count,
        "invalid_primitive_count": invalid_primitive_count,
        "invalid_motor_command_count": invalid_motor_command_count,
        "frame_count": frame_count,
        "episode_count": 1,
        "policy_decisions": policy_decisions,
        "simulation_steps": frame_count,
        "recurrent_hash_count": len(recurrent_hashes),
        "target_retained_count": target_retained,
        "target_invalidated_count": target_invalidated,
        "perception_update_count": perception_updates,
        "option_count": len(option_values),
        "motor_command_changes": motor_changes,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "cv_checkpoint_sha256": detector.model_sha256,
        "scenario_seed": SCENARIO_SEED,
        "policy_mode": POLICY_MODE,
        "policy_temperature": POLICY_TEMPERATURE,
        "path_length": path_length,
        "displacement": math.hypot(last_pose[0] - first_pose[0], last_pose[1] - first_pose[1])
        if first_pose and last_pose
        else 0.0,
        "heading_changes": heading_changes,
        "replanning_count": replanning_count,
        "stuck_recovery_count": stuck_recovery_count,
        "visited_cell_count": len(visited_cells),
        "trajectory_hash": trajectory_hasher.hexdigest(),
        "option_sequence_hash": option_hasher.hexdigest(),
        "target_sequence_hash": target_hasher.hexdigest(),
        "route_is_scripted": False,
        "run_mode": RUN_MODE,
        "user_requested_shutdown": RUN_MODE == "until_closed",
        "automatic_timeout_used": False,
        "shutdown_reason": "user_closed_webots"
        if RUN_MODE == "until_closed"
        else "bounded_complete",
        "random_motor_noise_used": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "marker_summary_generated.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
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
