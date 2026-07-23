from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"

WORLD_PATH = (
    PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a3_closed_loop_mission.wbt"
)

ROUTE_CONFIG_PATH = PROJECT_ROOT / "configs" / "webots" / "stage5a3_closed_loop_route.json"

ROBOT_CONTROLLER_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_robot"
    / "closed_loop_inspection_robot.py"
)

SUPERVISOR_CONTROLLER_PATH = (
    PROJECT_ROOT
    / "webots"
    / "controllers"
    / "closed_loop_inspection_supervisor"
    / "closed_loop_inspection_supervisor.py"
)

VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_stage5a3_closed_loop_mission.py"

LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_stage5a3_closed_loop_mission.ps1"
LAUNCHER_RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_stage5a3_closed_loop_mission.py"


def test_stage5a_world_is_preserved() -> None:
    content = SOURCE_WORLD_PATH.read_text(encoding="utf-8")

    assert 'controller "live_camera_robot"' in content
    assert 'controller "live_camera_supervisor"' in content


def test_stage5a3_world_exists() -> None:
    assert WORLD_PATH.is_file()
    assert WORLD_PATH.stat().st_size > 20_000


def test_stage5a3_world_uses_closed_loop_controllers() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    assert 'controller "closed_loop_inspection_robot"' in content
    assert 'controller "closed_loop_inspection_supervisor"' in content
    assert 'controller "live_camera_robot"' not in content
    assert "Stage 5A3 closed-loop route markers" in content


def test_route_configuration_is_complete() -> None:
    config = json.loads(ROUTE_CONFIG_PATH.read_text(encoding="utf-8"))

    waypoints = config["waypoints"]

    assert len(waypoints) >= 7
    assert waypoints[0]["name"] == "HOME_START"
    assert waypoints[-1]["name"] == "HOME_RETURN"
    assert waypoints[0]["x"] == waypoints[-1]["x"]
    assert waypoints[0]["z"] == waypoints[-1]["z"]
    assert sum(bool(item["scan"]) for item in waypoints) >= 3
    assert config["minimum_capture_count"] >= 200


def test_controller_performs_runtime_calibration() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "FORWARD_CALIBRATION_SECONDS" in content
    assert "TURN_CALIBRATION_SECONDS" in content
    assert "heading_offset" in content
    assert "turn_command_sign" in content
    assert "STAGE5A3_FORWARD_CALIBRATION_COMPLETE" in content
    assert "STAGE5A3_TURN_CALIBRATION_COMPLETE" in content


def test_controller_uses_closed_loop_waypoint_navigation() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "distance_to_target",
        "target_heading",
        "heading_error",
        "NAVIGATION_HEADING_GAIN",
        "NAVIGATION_DISTANCE_GAIN",
        "arrival_tolerance",
        "STATE_NAVIGATE",
        "STATE_SCAN",
        "STATE_RECOVERY",
        "closed_loop_waypoint_controller",
    )

    for token in required_tokens:
        assert token in content


def test_controller_uses_live_camera_and_sensor_fusion() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "camera.enable(time_step)" in content
    assert "camera.getImage()" in content
    assert "gps.enable(time_step)" in content
    assert "inertial_unit.enable(time_step)" in content
    assert "compass.enable(time_step)" in content
    assert "discover_distance_sensors" in content
    assert "Node.DISTANCE_SENSOR" in content
    assert "maximum_proximity_ratio" in content


def test_controller_records_zone_evidence() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "save_evidence_frame" in content
    assert "SCAN_VIEW_CAPTURED" in content
    assert "evidence_frames" in content
    assert "arrival_records" in content


def test_stage5a3_does_not_claim_cv_or_policy_control() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    prohibited_tokens = (
        "YOLO(",
        "MaskablePPO(",
        "PolicyDryRunEngine(",
        "stable_baselines3",
        "sb3_contrib",
        "ultralytics",
    )

    for token in prohibited_tokens:
        assert token not in content

    assert '"cv_model_connected": False' in content
    assert '"policy_controls_motors": False' in content
    assert '"collision_free_claimed": False' in content


def test_supervisor_terminates_marker_based_runtime() -> None:
    content = SUPERVISOR_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "stage5a3_complete.marker" in content
    assert "stage5a3_failure.marker" in content
    assert "stage5a3_timeout.marker" in content
    assert "simulationQuit(0)" in content
    assert "simulationQuit(1)" in content
    assert "simulationQuit(2)" in content


def test_validator_enforces_mission_quality() -> None:
    content = VALIDATOR_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "MINIMUM_CAPTURE_COUNT",
        "MINIMUM_PATH_LENGTH_METERS",
        "MAXIMUM_RETURN_DISTANCE_METERS",
        "MINIMUM_HEADING_CHANGE_DEGREES",
        "MINIMUM_EVIDENCE_FRAME_COUNT",
        "MAXIMUM_ARRIVAL_DISTANCE_METERS",
        "closed_loop_waypoint_controller",
        "collision_free_claimed",
    )

    for token in required_tokens:
        assert token in content


def test_launcher_uses_automated_runtime_validation() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")
    runner_content = LAUNCHER_RUNNER_PATH.read_text(encoding="utf-8")

    assert "run_stage5a3_closed_loop_mission.py" in content
    assert "stage5a3_complete.marker" in runner_content
    assert "stage5a3_failure.marker" in runner_content
    assert "WEBOTS_PYTHON_COMMAND" in runner_content
    assert "RISK_AWARE_PROJECT_ROOT" in runner_content
    assert '"--mode=fast"' in runner_content
    assert "validate_stage5a3_closed_loop_mission.py" in runner_content
