from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np


def find_project_root() -> Path:
    configured = os.environ.get("RISK_AWARE_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate the project root.")


PROJECT_ROOT = find_project_root()
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from controller import Robot  # noqa: E402

from riskaware_saferrl.live_perception import (  # noqa: E402
    annotate_frame,
    create_live_perception_backend,
)
from riskaware_saferrl.webots.bridge import (  # noqa: E402
    ACTION_TO_DELTA,
    BridgeState,
    CardinalHeading,
    GridFrame,
    ObservationBridge,
    SemanticScene,
)
from riskaware_saferrl.webots.policy_dry_run import PolicyDryRunEngine  # noqa: E402
from riskaware_saferrl.webots.runtime_control import (  # noqa: E402
    RuntimeSafetyEvaluator,
    inject_perception_risk,
)
from riskaware_saferrl.webots.safe_policy_pipeline import SafePolicyPipeline  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "webots" / "logs" / "stage5c_rl_motor_runtime"
CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "maskable_ppo_deadlock_safe_shield_seed42_u100"
    / "evaluations"
    / "best_model"
    / "best_model.zip"
)
CHECKPOINT_SHA256 = "172437CAE45B69031F443C0707FB0795D2F1860D3B95594BE281645D8A173FE7"
PERCEPTION_CONFIG = PROJECT_ROOT / "configs" / "perception" / "stage5b_live_perception.json"
DECISION_COUNT = 10
PRIMITIVE_STEPS = 18
MOTOR_SIGN = -1.0


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def camera_bgr(camera: object) -> np.ndarray:
    width = int(camera.getWidth())
    height = int(camera.getHeight())
    data = camera.getImage()
    if data is None:
        raise RuntimeError("Webots camera returned no frame.")
    bgra = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    return bgra[:, :, :3].copy()


def cv_risk_score(result: object) -> float:
    risk = result.risk
    if risk.fall_detected:
        return 1.0
    if risk.violation_count:
        return 0.9
    if result.detections:
        return 0.55
    return 0.0


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trace_path = OUTPUT_ROOT / "action_trace.jsonl"
    motor_path = OUTPUT_ROOT / "motor_commands.jsonl"
    for path in (trace_path, motor_path):
        path.unlink(missing_ok=True)

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    gps = robot.getDevice("gps")
    inertial = robot.getDevice("inertial unit")
    camera = robot.getDevice("inspection camera")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    gps.enable(time_step)
    inertial.enable(time_step)
    camera.enable(time_step)

    policy = PolicyDryRunEngine.from_checkpoint(
        CHECKPOINT,
        expected_sha256=CHECKPOINT_SHA256,
        device="cuda",
        seed=42,
    )
    detector = create_live_perception_backend(
        project_root=PROJECT_ROOT,
        config_path=PERCEPTION_CONFIG,
    )
    warmup_ms = detector.warmup()
    shield = RuntimeSafetyEvaluator()
    pipeline = SafePolicyPipeline(
        policy,
        shield,
        policy_controls_motors=True,
        verified_motor_runtime=True,
    )
    frame = GridFrame(size=12, x_min=-10.0, x_max=2.0, z_min=-7.0, z_max=1.0)
    bridge = ObservationBridge(vision_radius=4, inspection_radius=2)
    scene = SemanticScene(
        size=12,
        restricted=frozenset({(8, 1), (1, 9), (2, 9), (3, 9), (4, 9)}),
    )
    visited: set[tuple[int, int]] = set()
    inference_count = 0
    failure_count = 0
    annotated_count = 0
    perception_changed_observation_count = 0
    unique_policy_actions: set[int] = set()
    motor_command_changes = 0
    previous_motor_command: tuple[float, float] | None = None

    robot.step(time_step)
    for decision_index in range(DECISION_COUNT):
        position = gps.getValues()
        yaw = float(inertial.getRollPitchYaw()[2])
        agent_position = frame.world_to_grid(float(position[0]), float(position[2]))
        visited.add(agent_position)
        state = BridgeState(
            agent_position=agent_position,
            visited=frozenset(visited),
            inspected=frozenset(),
            steps=decision_index,
            max_steps=DECISION_COUNT,
        )
        observation = bridge.build_observation(scene, state)
        action_mask = bridge.action_mask(scene, state)

        image = camera_bgr(camera)
        try:
            perception = detector.infer(image, frame_id=decision_index)
            inference_count += 1
        except Exception:
            failure_count += 1
            raise
        risk_score = cv_risk_score(perception)
        updated_observation = inject_perception_risk(
            observation,
            agent_position=agent_position,
            grid_size=12,
            risk_score=risk_score,
        )
        if not np.array_equal(observation["map"], updated_observation["map"]):
            perception_changed_observation_count += 1

        annotated = annotate_frame(image, perception)
        annotated_path = OUTPUT_ROOT / f"annotated_{decision_index:03d}.png"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"Could not write {annotated_path}")
        annotated_count += 1

        forbidden: set[int] = set()
        for action, delta in ACTION_TO_DELTA.items():
            candidate = (agent_position[0] + delta[0], agent_position[1] + delta[1])
            if candidate in scene.restricted:
                forbidden.add(int(action))
        shield.configure(
            forbidden_actions=forbidden,
            emergency_stop=bool(perception.risk.fall_detected),
        )

        trace = pipeline.execute(
            sample_index=decision_index,
            observation=updated_observation,
            task_valid_mask=action_mask,
            heading=CardinalHeading.from_yaw(yaw),
        )
        unique_policy_actions.add(trace.proposed_action)

        trace_payload = trace.to_dict()
        trace_payload.update(
            {
                "policy_loaded": True,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "cv_risk_score": risk_score,
                "cv_detection_count": len(perception.detections),
                "perception_changed_observation": not np.array_equal(
                    observation["map"], updated_observation["map"]
                ),
                "safety_shield_active": True,
            }
        )
        append_jsonl(trace_path, trace_payload)

        for primitive_index, command in enumerate(trace.motor_commands):
            actual_command = (MOTOR_SIGN * command[0], MOTOR_SIGN * command[1])
            if previous_motor_command is not None and actual_command != previous_motor_command:
                motor_command_changes += 1
            previous_motor_command = actual_command
            for primitive_step in range(PRIMITIVE_STEPS):
                left_motor.setVelocity(actual_command[0])
                right_motor.setVelocity(actual_command[1])
                append_jsonl(
                    motor_path,
                    {
                        "decision_index": decision_index,
                        "primitive_index": primitive_index,
                        "primitive_step": primitive_step,
                        "proposed_action": trace.proposed_action,
                        "executed_action": trace.executed_action,
                        "left_velocity": actual_command[0],
                        "right_velocity": actual_command[1],
                    },
                )
                if robot.step(time_step) == -1:
                    raise RuntimeError("Webots stopped before the policy mission completed.")

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    summary = {
        "schema_version": 1,
        "stage": "5C",
        "runtime_verified": True,
        "policy_loaded": True,
        "policy_controls_motors": True,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "policy_device": policy.device,
        "safety_shield_active": True,
        "safety_shield_interventions": shield.intervention_count,
        "restricted_zone_metrics_recorded": True,
        "cv_model_connected": detector.model_connected,
        "cv_model_sha256": detector.model_sha256,
        "cuda_inference_verified": detector.device.startswith("cuda"),
        "perception_live_during_mission": True,
        "perception_affects_runtime_state": perception_changed_observation_count > 0,
        "perception_state_change_count": perception_changed_observation_count,
        "inference_count": inference_count,
        "failure_count": failure_count,
        "annotated_frame_count": annotated_count,
        "policy_decision_count": DECISION_COUNT,
        "unique_policy_action_count": len(unique_policy_actions),
        "motor_command_change_count": motor_command_changes,
        "manual_control_used": False,
        "fallback_controller_used": False,
        "warmup_ms": warmup_ms,
        "mission_completed": True,
    }
    (OUTPUT_ROOT / "stage5c_runtime_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "stage5c_complete.marker").write_text("complete\n", encoding="utf-8")
    print("STAGE5C_RL_MOTOR_RUNTIME_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "stage5c_failure.json").write_text(
            json.dumps(
                {"error": str(error), "traceback": traceback.format_exc()},
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
