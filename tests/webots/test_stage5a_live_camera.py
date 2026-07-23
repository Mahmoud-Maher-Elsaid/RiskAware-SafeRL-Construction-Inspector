from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4d_showcase.wbt"

WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"

ROBOT_CONTROLLER_PATH = (
    PROJECT_ROOT / "webots" / "controllers" / "live_camera_robot" / "live_camera_robot.py"
)

SUPERVISOR_CONTROLLER_PATH = (
    PROJECT_ROOT / "webots" / "controllers" / "live_camera_supervisor" / "live_camera_supervisor.py"
)

VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "validate_stage5a_live_camera.py"

LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_stage5a_live_camera.ps1"


def test_stage4d_world_is_preserved() -> None:
    assert SOURCE_WORLD_PATH.is_file()

    content = SOURCE_WORLD_PATH.read_text(encoding="utf-8")

    assert 'controller "showcase_robot"' in content

    assert 'controller "showcase_supervisor"' in content


def test_stage5a_world_exists() -> None:
    assert WORLD_PATH.is_file()
    assert WORLD_PATH.stat().st_size > 20_000


def test_stage5a_world_uses_live_camera_controllers() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    assert 'controller "live_camera_robot"' in content

    assert 'controller "live_camera_supervisor"' in content

    assert 'controller "showcase_robot"' not in content


def test_stage5a_camera_contract() -> None:
    content = WORLD_PATH.read_text(encoding="utf-8")

    assert 'name "inspection camera"' in content
    assert "width 640" in content
    assert "height 360" in content
    assert 'name "gps"' in content
    assert 'name "inertial unit"' in content
    assert 'name "compass"' in content


def test_live_camera_controller_captures_bgra_frames() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "camera.enable(time_step)" in content
    assert "camera.getImage()" in content
    assert "expected_byte_length" in content
    assert "pixel_format" in content
    assert '"BGRA"' in content
    assert "hashlib.sha256" in content
    assert "cv2.Laplacian" in content
    assert "image_entropy" in content


def test_live_camera_controller_synchronizes_sensors() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "gps.enable(time_step)" in content
    assert "inertial_unit.enable(time_step)" in content
    assert "compass.enable(time_step)" in content
    assert "gps.getValues()" in content
    assert "inertial_unit.getRollPitchYaw()" in content
    assert "compass.getValues()" in content


def test_stage5a_does_not_claim_cv_or_policy_control() -> None:
    content = ROBOT_CONTROLLER_PATH.read_text(encoding="utf-8")

    prohibited_tokens = (
        "YOLO(",
        "MaskablePPO(",
        "PolicyDryRunEngine(",
        "stable_baselines3",
        "sb3_contrib",
        "ultralytics",
    )

    for prohibited_token in prohibited_tokens:
        assert prohibited_token not in content

    assert '"cv_model_connected": False' in content

    assert '"policy_controls_motors": False' in content


def test_supervisor_terminates_automated_runtime() -> None:
    content = SUPERVISOR_CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "stage5a_complete.marker" in content
    assert "stage5a_failure.marker" in content
    assert "simulationQuit(0)" in content
    assert "simulationQuit(1)" in content
    assert "simulationQuit(2)" in content


def test_validator_enforces_camera_quality() -> None:
    content = VALIDATOR_PATH.read_text(encoding="utf-8")

    required_tokens = (
        "MINIMUM_UNIQUE_CHECKSUMS",
        "MINIMUM_MEAN_CONTRAST",
        "MINIMUM_MEAN_ENTROPY_BITS",
        "MINIMUM_MEAN_SHARPNESS",
        "MINIMUM_NON_BLACK_RATIO",
        "MINIMUM_MEAN_FRAME_DIFFERENCE",
        "MINIMUM_ROBOT_PATH_LENGTH_METERS",
    )

    for required_token in required_tokens:
        assert required_token in content


def test_launcher_uses_marker_based_completion() -> None:
    content = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "stage5a_complete.marker" in content
    assert "stage5a_failure.marker" in content
    assert "stage5a_timeout.marker" in content
    assert "WEBOTS_PYTHON_COMMAND" in content
    assert "RISK_AWARE_PROJECT_ROOT" in content
    assert "--mode=realtime" in content
    assert "validate_stage5a_live_camera.py" in content
