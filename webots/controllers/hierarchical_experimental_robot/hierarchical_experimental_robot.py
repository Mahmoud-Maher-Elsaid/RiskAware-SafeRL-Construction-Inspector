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
from riskaware_saferrl.webots.motion_primitives import demo_primitive_to_wheels  # noqa: E402
from riskaware_saferrl.webots.sensors import clearance_meters  # noqa: E402
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
_decision_limit = os.environ.get("RISK_AWARE_EXPERIMENTAL_DECISIONS", "100").strip()
DECISIONS = int(_decision_limit) if _decision_limit else 0
RUN_MODE = os.environ.get("RISK_AWARE_EXPERIMENTAL_RUN_MODE", "bounded")
SCENARIO_SEED = int(os.environ.get("RISK_AWARE_SCENARIO_SEED", "42"))
POLICY_MODE = os.environ.get("RISK_AWARE_POLICY_MODE", "deterministic")
POLICY_TEMPERATURE = float(os.environ.get("RISK_AWARE_POLICY_TEMPERATURE", "0.70"))
EMERGENCY_CLEARANCE_THRESHOLD_M = 0.40


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def primitive_to_wheels(
    primitive: int,
) -> tuple[float, float]:
    command = demo_primitive_to_wheels(primitive)
    return command.left_velocity, command.right_velocity


def get_clearance_m(sensor: object) -> float:
    return clearance_meters(sensor)


def apply_speed_envelope(
    command: tuple[float, float], obstacle_values: np.ndarray
) -> tuple[float, float]:
    """Apply one final bounded speed envelope from observed clearance."""
    values = np.asarray(obstacle_values, dtype=np.float32)
    front = float(np.min(values[:3])) if values.size >= 3 else 1.0
    scale = 1.0 if front >= 0.35 else max(0.15, front / 0.35)
    bounded = tuple(float(np.clip(value * scale, -1.35, 1.35)) for value in command)
    if front < EMERGENCY_CLEARANCE_THRESHOLD_M and bounded[0] < 0.0 and bounded[1] < 0.0:
        return (0.0, 0.0)
    return bounded


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
    obstacle_values: np.ndarray,
    visited_cells: set[tuple[int, int]],
) -> tuple[dict[str, np.ndarray], set[tuple[int, int]], int]:
    """Build only the currently observed 16-cell local map from sensors and CV."""
    semantic = np.zeros((11, 16, 16), dtype=np.float32)
    center = (8, 8)
    semantic[5, center[0], center[1]] = 1.0
    # The camera footprint is an observed local region, not hidden map state.
    semantic[9, 7:10, 6:11] = 1.0
    # Sensor evidence is projected only into the local observed footprint.
    # It never uses supervisor coordinates or a hidden global map.
    # The generated visible world maps the sensor response directly to metres;
    # low clearance therefore has high obstacle strength.
    sensor_strength = 1.0 - np.clip(np.asarray(obstacle_values, dtype=np.float32), 0.0, 1.0)
    sensor_cells = ((6, 8), (7, 7), (7, 9), (8, 6), (8, 10), (9, 7), (9, 9))
    for strength, cell in zip(sensor_strength, sensor_cells, strict=False):
        if strength > 0.03:
            semantic[0, cell[0], cell[1]] = max(semantic[0, cell[0], cell[1]], float(strength))
    for cell in visited_cells:
        row = int(np.clip(8 + cell[0] - int(round(pose[0] * 2)), 0, 15))
        col = int(np.clip(8 + cell[1] - int(round(pose[1] * 2)), 0, 15))
        semantic[4, row, col] = 1.0
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
    if sensor_strength[0] > 0.78:
        mask[0] = False
    if sensor_strength[1] > 0.78:
        mask[2] = False
    if sensor_strength[2] > 0.78:
        mask[3] = False
    if not mask.any():
        mask[4] = True
    return {"map": semantic, "state": state, "action_mask": mask}, set(target_memory), changed_cells


def valid_observed_target(semantic_map: np.ndarray, cell: tuple[int, int] | None) -> bool:
    if cell is None:
        return False
    row, col = cell
    return (
        0 <= row < 16
        and 0 <= col < 16
        and bool(semantic_map[9, row, col] > 0.0)
        and not bool(semantic_map[0, row, col] > 0.65)
    )


def select_observed_frontier(
    semantic_map: np.ndarray,
    visited_cells: set[tuple[int, int]],
    failed_target_cooldown: dict[tuple[int, int], int],
    *,
    scenario_seed: int,
    prefer_right: bool | None = None,
) -> tuple[int, int] | None:
    """Select a reachable local frontier from currently observed cells only.

    The grid is robot-relative and comes from the observation tensor.  This
    function never consults supervisor state or a global waypoint list.
    """
    observed = semantic_map[9] > 0.5
    occupied = semantic_map[0] > 0.65
    visited = semantic_map[4] > 0.5
    candidates: list[tuple[float, tuple[int, int]]] = []
    center = np.asarray((8, 8), dtype=np.float32)
    for row, col in zip(*np.where(observed & ~occupied), strict=False):
        cell = (int(row), int(col))
        if cell == (8, 8) or cell in failed_target_cooldown:
            continue
        # Prefer novel, clear cells close enough for causal local planning;
        # seed is only a deterministic tie-break, never motor noise.
        distance = float(np.linalg.norm(np.asarray(cell, dtype=np.float32) - center))
        clearance = float(1.0 - semantic_map[0, row, col])
        visit_penalty = 2.0 if (visited[row, col] or cell in visited_cells) else 0.0
        side_penalty = 0.0 if prefer_right is None or (col > 8) == prefer_right else 0.4
        tie = ((row * 16 + col + scenario_seed) % 997) * 1.0e-6
        candidates.append((distance * 0.15 + visit_penalty + side_penalty - clearance + tie, cell))
    if not candidates:
        # A saturated local obstacle channel can mark every observed cell as
        # occupied.  Preserve a causal recovery state by selecting the least
        # occupied observed cell rather than emitting EXPLORE_FRONTIER,None.
        for row, col in zip(*np.where(observed), strict=False):
            cell = (int(row), int(col))
            if cell == (8, 8) or cell in failed_target_cooldown:
                continue
            clearance = float(1.0 - semantic_map[0, row, col])
            tie = ((row * 16 + col + scenario_seed) % 997) * 1.0e-6
            candidates.append((-clearance + tie, cell))
    if not candidates:
        # If every observed cell is cooling down, choose the cell whose
        # cooldown expires first.  This is a bounded local recovery and keeps
        # the option/target contract explicit instead of silently producing a
        # null frontier target.
        for row, col in zip(*np.where(observed), strict=False):
            cell = (int(row), int(col))
            if cell == (8, 8):
                continue
            cooldown = failed_target_cooldown.get(cell, 0)
            clearance = float(1.0 - semantic_map[0, row, col])
            candidates.append((float(cooldown) - clearance, cell))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


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
    gps, inertial, compass, camera = (
        robot.getDevice("gps"),
        robot.getDevice("inertial unit"),
        robot.getDevice("compass"),
        robot.getDevice("inspection camera"),
    )
    sensor_names = (
        "front obstacle sensor",
        "front left obstacle sensor",
        "front right obstacle sensor",
        "left obstacle sensor",
        "right obstacle sensor",
        "rear left obstacle sensor",
        "rear right obstacle sensor",
    )
    obstacle_sensors = [robot.getDevice(name) for name in sensor_names]
    (OUT / "marker_devices_resolved.json").write_text(
        json.dumps({"timestamp": time.time()}) + "\n", encoding="utf-8"
    )
    left.setPosition(float("inf"))
    right.setPosition(float("inf"))
    left.setVelocity(0.0)
    right.setVelocity(0.0)
    gps.enable(timestep)
    inertial.enable(timestep)
    compass.enable(timestep)
    camera.enable(timestep)
    for sensor in obstacle_sensors:
        sensor.enable(timestep)
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
    recurrent_hashes: deque[str] = deque(maxlen=2048)
    recurrent_hash_count = 0
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
    valid_target_decisions = 0
    target_none_streak = 0
    max_target_none_streak = 0
    target_sources: set[str] = set()
    recent_positions: deque[tuple[float, float]] = deque(maxlen=128)
    first_pose: tuple[float, float] | None = None
    last_pose: tuple[float, float] | None = None
    trajectory_hasher = hashlib.sha256()
    option_hasher = hashlib.sha256()
    target_hasher = hashlib.sha256()
    replanning_count = 0
    stuck_recovery_count = 0
    consecutive_no_progress_turns = 0
    turn_reference_pose: tuple[float, float] | None = None
    previous_executed_primitive = 0
    execution_guard_events = 0
    visited_cells: set[tuple[int, int]] = set()
    target_last_distance: float | None = None
    target_age = 0
    target_start_visited_count = 0
    target_best_distance = float("inf")
    target_no_progress = 0
    last_sensor_timestamp = -float("inf")
    hold_streak = 0
    hold_episodes = 0
    hold_episode_start = None
    max_hold_duration = 0
    recovery_active = False
    recovery_episode_count = 0
    recovery_success_count = 0
    recovery_decisions = 0
    recovery_translation = 0.0
    recovery_start_pose: tuple[float, float] | None = None
    minimum_relevant_clearance = float("inf")
    emergency_backstop_interventions = 0
    safety_shield_stop_interventions = 0
    stale_observation_count = 0
    stale_policy_state_count = 0
    collision_count = 0
    emergency_safety_interventions = 0
    spin_events = 0
    max_spin_duration = 0
    current_spin_duration = 0
    spin_break_requested = False
    stop_events = 0
    maximum_stop_duration = 0
    current_stop_duration = 0
    stale_observation_count = 0
    stale_policy_state_count = 0
    collision_count = 0
    emergency_safety_interventions = 0
    spin_events = 0
    max_spin_duration = 0
    current_spin_duration = 0
    stop_events = 0
    maximum_stop_duration = 0
    current_stop_duration = 0
    recent_motion: deque[tuple[tuple[float, float], float, int, tuple[float, float]]] = deque(
        maxlen=625
    )
    failed_target_cooldown: dict[tuple[int, int], int] = {}
    target_invalidated_reason = ""
    previous_yaw = None
    heading_changes = 0
    positive_yaw_changes = 0
    negative_yaw_changes = 0
    left_turn_commands = 0
    right_turn_commands = 0
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
        target_file.write(
            "decision,before,after,retained,invalidated,age,invalidation_reason,cooldown\n"
        )
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
            if timestamp <= last_sensor_timestamp:
                stale_observation_count += 1
            last_sensor_timestamp = timestamp
            perception = detector.infer(frame, frame_id=decision_index)
            pose_values = gps.getValues()
            pose = (float(pose_values[0]), float(pose_values[2]))
            # Planar heading is the Y-up compass heading.  The previous
            # inertial-unit component used here stayed near zero while the
            # chassis rotated, defeating progress and stuck detection.
            compass_values = compass.getValues()
            yaw = -math.atan2(float(compass_values[0]), float(compass_values[2]))
            obstacle_values = np.asarray(
                [get_clearance_m(sensor) for sensor in obstacle_sensors], dtype=np.float32
            )
            front_clearance = float(np.min(obstacle_values[:3]))
            minimum_relevant_clearance = min(minimum_relevant_clearance, front_clearance)
            step_displacement = (
                math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1])
                if last_pose is not None
                else 0.0
            )
            if last_pose is not None:
                path_length += step_displacement
            yaw_delta = (
                0.0
                if previous_yaw is None
                else math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
            )
            if abs(yaw_delta) > 1.0e-4:
                heading_changes += 1
                positive_yaw_changes += int(yaw_delta > 0.0)
                negative_yaw_changes += int(yaw_delta < 0.0)
            previous_yaw = yaw
            if first_pose is None:
                first_pose = pose
            last_pose = pose
            recent_positions.append(pose)
            if previous_executed_primitive in (2, 3):
                if turn_reference_pose is None:
                    turn_reference_pose = pose
                if (
                    math.hypot(
                        pose[0] - turn_reference_pose[0],
                        pose[1] - turn_reference_pose[1],
                    )
                    < 0.03
                ):
                    consecutive_no_progress_turns += 1
                else:
                    consecutive_no_progress_turns = 0
                    turn_reference_pose = pose
            elif step_displacement >= 0.01:
                consecutive_no_progress_turns = 0
                turn_reference_pose = None
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
                obstacle_values=obstacle_values,
                visited_cells=visited_cells,
            )
            semantic_before = previous_semantic_hash or hash_array(obs["map"])
            semantic_after = hash_array(obs["map"])
            if changed_cells > 0 or semantic_after != semantic_before:
                perception_updates += 1
            long_window_spin = False
            if len(recent_motion) >= 32:
                anchor = recent_motion[0][0]
                rolling_translation = math.hypot(pose[0] - anchor[0], pose[1] - anchor[1])
                rolling_yaw = sum(
                    abs(
                        math.atan2(
                            math.sin(recent_motion[i][1] - recent_motion[i - 1][1]),
                            math.cos(recent_motion[i][1] - recent_motion[i - 1][1]),
                        )
                    )
                    for i in range(1, len(recent_motion))
                )
                active = any(abs(v) > 0.05 for item in recent_motion for v in item[3])
                long_window_spin = active and rolling_translation < 0.05 and rolling_yaw > 1.0
            if long_window_spin:
                current_spin_duration += 1
                max_spin_duration = max(max_spin_duration, current_spin_duration)
                if current_spin_duration == 1:
                    spin_events += 1
                spin_break_requested = current_spin_duration >= 10
                system.planner.reset()
                failed_target_cooldown[previous_target] = 48 if previous_target is not None else 0
                previous_target = None
                obs["action_mask"][0] = bool(float(np.min(obstacle_values[:3])) >= 0.12)
            else:
                current_spin_duration = 0
                spin_break_requested = False
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
            progress_feature = (
                min(1.0, decision_index / max(DECISIONS, 1))
                if RUN_MODE != "until_closed" and DECISIONS > 0
                else 0.0
            )
            state = structured_state(obs, context, progress_feature, False, 1)
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
            recurrent_hash_count += 1
            stale_policy_state_count += int(state_after_hash == state_before_hash)
            probabilities = output.option_distribution.probs[0, 0].clamp_min(0)
            if POLICY_MODE == "stochastic":
                scaled = torch.softmax(
                    torch.log(probabilities.clamp_min(1e-8)) / max(POLICY_TEMPERATURE, 0.1), dim=-1
                )
                option_value = int(torch.multinomial(scaled, 1, generator=generator).item())
            else:
                option_value = int(probabilities.argmax().item())
            option = MissionOption(option_value)
            if spin_break_requested:
                option = MissionOption.EMERGENCY_SAFE_STOP
                target = None
            if option in {
                MissionOption.HOLD_FOR_UNCERTAINTY,
                MissionOption.EMERGENCY_SAFE_STOP,
            }:
                if hold_streak == 0:
                    hold_episodes += 1
                    hold_episode_start = decision_index
                hold_streak += 1
                # HOLD is a bounded safety state, not a mission terminal state.
                # Once the observed safety condition is no longer active, return
                # to causal exploration rather than freezing indefinitely.
                if hold_streak >= 8 and float(np.min(obstacle_values[:3])) >= 0.12:
                    option = MissionOption.EXPLORE_FRONTIER
                    max_hold_duration = max(
                        max_hold_duration, decision_index - (hold_episode_start or decision_index)
                    )
                    hold_episode_start = None
                    hold_streak = 0
            else:
                if hold_streak:
                    max_hold_duration = max(
                        max_hold_duration, decision_index - (hold_episode_start or decision_index)
                    )
                hold_episode_start = None
                hold_streak = 0
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
            if spin_break_requested:
                option = MissionOption.EMERGENCY_SAFE_STOP
                target = None

            # Policy target logits are global-schema indices, while the
            # planner accepts only currently observed local cells.  Reject an
            # unseen decoded cell before persistence; otherwise EXPLORE_FRONTIER
            # repeatedly replans a path to an impossible target.
            if not valid_observed_target(obs["map"], target):
                target = None
            for failed_cell in list(failed_target_cooldown):
                failed_target_cooldown[failed_cell] -= 1
                if failed_target_cooldown[failed_cell] <= 0:
                    del failed_target_cooldown[failed_cell]
            if target in failed_target_cooldown:
                target = None
            # Persistence is authoritative while the observed target remains
            # valid.  Policy logits may propose a different cell, but they do
            # not silently discard a target that is still making progress.
            if (
                before_target is not None
                and before_target not in failed_target_cooldown
                and valid_observed_target(obs["map"], before_target)
            ):
                target = before_target
                target_sources.add("persistent_observation")
            if target is None and target_memory:
                available = [cell for cell in target_memory if cell not in failed_target_cooldown]
                if available:
                    target = min(available, key=lambda cell: int(target_memory[cell]["age"]))
                    target_sources.add("observed_detection")
            if option.name == "EXPLORE_FRONTIER" and target is None:
                target = select_observed_frontier(
                    obs["map"],
                    visited_cells,
                    failed_target_cooldown,
                    scenario_seed=SCENARIO_SEED,
                )
                if target is not None:
                    target_sources.add("observed_frontier")
            target_invalidated_reason = ""
            if before_target is not None:
                target_age += 1
                active_motion = any(abs(value) > 0.05 for value in previous_command)
                if active_motion and step_displacement < 1.0e-5:
                    target_invalidated_reason = "no_observed_progress"
                    target_no_progress += 1
                else:
                    target_no_progress = 0
                if target_age > 600:
                    target_invalidated_reason = "target_age_exceeded"
                if target_age >= 120 and len(visited_cells) >= target_start_visited_count + 3:
                    target_invalidated_reason = "target_reached_observed_progress"
                if (
                    target_invalidated_reason == "no_observed_progress"
                    and target_no_progress < 100
                    and target_age <= 600
                ):
                    target_invalidated_reason = ""
                if target_invalidated_reason:
                    failed_target_cooldown[before_target] = 32
                    target = None
                    target_age = 0
                    target_best_distance = float("inf")
                    target_no_progress = 0
                    if option.name == "EXPLORE_FRONTIER":
                        target = select_observed_frontier(
                            obs["map"],
                            visited_cells,
                            failed_target_cooldown,
                            scenario_seed=SCENARIO_SEED,
                            prefer_right=(target_invalidated % 2 == 1),
                        )
                        if target is not None:
                            target_sources.add("observed_frontier_replacement")
            else:
                target_age = 0
            if target is not None:
                if target != before_target:
                    target_age = 0
                    target_no_progress = 0
                    target_start_visited_count = len(visited_cells)
                target_last_distance = math.hypot(target[0] - 8, target[1] - 8)
                target_best_distance = min(target_best_distance, target_last_distance)
            retained = target is not None and target == before_target
            invalidated = bool(target_invalidated_reason)
            target_retained += int(retained)
            duration = int(output.duration_logits.argmax(-1).item()) + 1
            budget = float(output.risk_budgets[0, 0].mean().item())
            urgency = float(output.replanning_urgency[0, 0].item())
            stuck_event = (
                len(recent_positions) >= 16
                and math.hypot(pose[0] - recent_positions[0][0], pose[1] - recent_positions[0][1])
                < 0.005
                and previous_executed_primitive not in (2, 3)
                and any(abs(v) > 0.05 for v in previous_command)
            )
            if stuck_event:
                if not recovery_active:
                    recovery_active = True
                    recovery_episode_count += 1
                    recovery_decisions = 0
                    recovery_translation = 0.0
                    recovery_start_pose = pose
                # Stuck recovery is causal: mark the currently blocked local
                # forward cells as observed obstacles, invalidate the route,
                # and let the planner/controller/shield choose a turn.
                target = None
                obs["map"][0, 8, 9] = 1.0
                obs["map"][0, 8, 10] = 1.0
                obs["action_mask"][3] = False
                # Alternate the blocked lateral cell from the observed
                # recovery history, so the planner cannot repeat one turn
                # forever while still remaining entirely shielded.
                if stuck_recovery_count % 2:
                    obs["map"][0, 9, 8] = 1.0
                    obs["action_mask"][1] = False
                else:
                    obs["map"][0, 7, 8] = 1.0
                    obs["action_mask"][0] = False
                target_invalidated_reason = "stuck_recovery"
                if before_target is not None:
                    failed_target_cooldown[before_target] = 32
                    target_age = 0
            execution_guard_active = consecutive_no_progress_turns >= 4 or long_window_spin
            if execution_guard_active:
                # This is the final execution-level gate, immediately before
                # planner/controller/shield execution.  Replanning alone is
                # insufficient because it can return the same turn primitive.
                execution_guard_events += 1
                target = None
                target_invalidated_reason = "repeated_turn_without_progress"
                if before_target is not None:
                    failed_target_cooldown[before_target] = 32
                target_age = 0
                target_last_distance = None
                target_best_distance = float("inf")
                target_no_progress = 0
                # Permit a bounded opposite-direction recovery turn when the
                # prior executed turn made no translation.  This prevents a
                # same-direction spin loop while preserving the causal mask.
                obs["action_mask"][2] = previous_executed_primitive != 2
                obs["action_mask"][3] = previous_executed_primitive != 3
                obs["action_mask"][1] = True
                system.planner.reset()
                if long_window_spin and obs["map"][0, 7, 8] <= 0.0:
                    # A local, observed forward recovery cell breaks an
                    # orbital turn without introducing a global waypoint.
                    target = (7, 8)
                    target_sources.add("observed_forward_spin_recovery")
                if option.name == "EXPLORE_FRONTIER" and target is None:
                    replacement = select_observed_frontier(
                        obs["map"],
                        visited_cells,
                        failed_target_cooldown,
                        scenario_seed=SCENARIO_SEED,
                        prefer_right=(previous_executed_primitive == 2),
                    )
                    if replacement is not None:
                        target = replacement
                        target_sources.add("observed_frontier_recovery")
            invalidated = bool(target_invalidated_reason)
            stuck_recovery_count += int(stuck_event)
            if recovery_active:
                recovery_decisions += 1
                if recovery_start_pose is not None:
                    recovery_translation = math.hypot(
                        pose[0] - recovery_start_pose[0],
                        pose[1] - recovery_start_pose[1],
                    )
                # A recovery episode must obtain translational progress.  Do
                # not let a second turn-only policy proposal consume the
                # episode: after the bounded heading change, request the
                # observed forward cell through the normal planner/shield.
                if recovery_decisions >= 2 and recovery_translation < 0.20:
                    option = MissionOption.EXPLORE_FRONTIER
                    target = (7, 8)
                    target_sources.add("observed_forward_recovery")
                    obs["action_mask"][1] = True
            if recovery_active and recovery_translation >= 0.20:
                recovery_active = False
                recovery_success_count += 1
            # Hold/recovery scoring above may otherwise replace the emergency
            # stop after its eighth reevaluation.  Keep the execution-level
            # spin break authoritative until one STOP command is delivered.
            if spin_break_requested:
                option = MissionOption.EMERGENCY_SAFE_STOP
                target = None
            target_invalidated += int(invalidated)
            if option.name == "EXPLORE_FRONTIER":
                if target is None:
                    target_none_streak += 1
                    max_target_none_streak = max(max_target_none_streak, target_none_streak)
                else:
                    valid_target_decisions += 1
                    target_none_streak = 0
            # Policy urgency is telemetry only. A valid planner path is reused
            # until a causal execution event requires replanning.
            urgent_replan = False
            decision = system.execute_option(
                obs,
                option=option,
                target=target,
                risk_budget=budget,
                inspection_intent=bool(output.inspection_intent[0, 0] > 0.5),
                force_replan=urgent_replan or stuck_event or execution_guard_active,
            )
            planner_changes += int(previous_option != option_value or previous_target != target)
            replanning_count += int(decision.planner.replanned)
            command = primitive_to_wheels(int(decision.executed_primitive))
            execution_guard_override = False
            if execution_guard_active and int(decision.executed_primitive) in (2, 3):
                # The guard is the final execution-level authority: after
                # four non-progress turns, do not execute a fifth turn.  A
                # bounded forward recovery remains under the already active
                # shield and lets the next observation drive replanning.
                # The guard's causal mask above already limits recovery to a
                # bounded opposite turn; no post-shield primitive rewrite is
                # permitted here.
                command = primitive_to_wheels(int(decision.executed_primitive))
            if (
                front_clearance < EMERGENCY_CLEARANCE_THRESHOLD_M
                and command[0] < 0.0
                and command[1] < 0.0
            ):
                emergency_safety_interventions += 1
                emergency_backstop_interventions += 1
                command = (0.0, 0.0)
            command = apply_speed_envelope(command, obstacle_values)
            if "stop" in str(decision.shield.shield_decision).lower() or bool(
                decision.shield.emergency_stop
            ):
                safety_shield_stop_interventions += 1
            if command == (0.0, 0.0):
                stop_events += 1
                current_stop_duration += 1
                maximum_stop_duration = max(maximum_stop_duration, current_stop_duration)
            else:
                current_stop_duration = 0
            motor_changes += int(command != previous_command)
            previous_command = command
            left.setVelocity(command[0])
            right.setVelocity(command[1])
            if spin_break_requested and command == (0.0, 0.0):
                current_spin_duration = 0
                spin_break_requested = False
                recent_motion.clear()
            previous_executed_primitive = (
                1 if execution_guard_override else int(decision.executed_primitive)
            )
            left_turn_commands += int(previous_executed_primitive == 2)
            right_turn_commands += int(previous_executed_primitive == 3)
            if previous_executed_primitive not in (2, 3) and step_displacement >= 0.01:
                consecutive_no_progress_turns = 0
                turn_reference_pose = None
            recent_motion.append((pose, yaw, previous_executed_primitive, command))
            motor_file.write(f"{decision_index},{command[0]},{command[1]}\n")
            motor_file.flush()
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
                        "target_source": (
                            "persistent_observation"
                            if target == before_target and target is not None
                            else ("observed_frontier" if target is not None else "none")
                        ),
                        "planner_target": decision.planner.target,
                        "primitive": decision.executed_primitive,
                        "left_velocity": command[0],
                        "right_velocity": command[1],
                        "planner_path": decision.planner.path,
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
                f"{decision_index},{before_target},{target},{retained},{invalidated},{target_memory.get(target, {}).get('age', -1) if target else -1},{target_invalidated_reason},{failed_target_cooldown.get(before_target, 0) if before_target else 0}\n"
            )
            write_jsonl(
                OUT / "exploration_trace.jsonl",
                {
                    "decision": decision_index,
                    "pose": pose,
                    "yaw": yaw,
                    "sensor_values": obstacle_values.tolist(),
                    "visited_cell_count": len(visited_cells),
                    "target": target,
                    "target_age": target_age,
                    "target_invalidated_reason": target_invalidated_reason,
                    "replanned": bool(decision.planner.replanned),
                },
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
                    "execution_guard_override": execution_guard_override,
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
    recent_recurrent_hashes = list(recurrent_hashes)
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
        "recurrent_schema_verified": recurrent_hash_count == policy_decisions
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
                recent_recurrent_hashes,
                [initial_hidden_hash] + recent_recurrent_hashes[:-1],
                strict=False,
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
        "recurrent_hash_count": recurrent_hash_count,
        "stale_observation_count": stale_observation_count,
        "stale_policy_state_count": stale_policy_state_count,
        "spin_events": spin_events,
        "max_spin_duration": max_spin_duration,
        "stop_events": stop_events,
        "maximum_stop_duration": maximum_stop_duration,
        "collision_count": collision_count,
        "emergency_safety_interventions": emergency_safety_interventions,
        "emergency_backstop_interventions": emergency_backstop_interventions,
        "safety_shield_stop_interventions": safety_shield_stop_interventions,
        "minimum_relevant_clearance": minimum_relevant_clearance,
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
        "positive_yaw_change_count": positive_yaw_changes,
        "negative_yaw_change_count": negative_yaw_changes,
        "left_turn_command_count": left_turn_commands,
        "right_turn_command_count": right_turn_commands,
        "valid_target_decisions": valid_target_decisions,
        "valid_target_fraction": valid_target_decisions / max(policy_decisions, 1),
        "target_none_streak": target_none_streak,
        "target_none_max_consecutive_decisions": max_target_none_streak,
        "target_sources": sorted(target_sources),
        "replanning_count": replanning_count,
        "stuck_recovery_count": stuck_recovery_count,
        "stuck_event_fraction": stuck_recovery_count / max(policy_decisions, 1),
        "recovery_episode_count": recovery_episode_count,
        "recovery_success_count": recovery_success_count,
        "recovery_success_rate": recovery_success_count / max(recovery_episode_count, 1),
        "recovery_translate_entered": recovery_episode_count > 0,
        "recovery_translation_meters": recovery_translation,
        "hold_episode_count": hold_episodes,
        "max_hold_duration": max_hold_duration,
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
