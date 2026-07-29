from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"
DESTINATION = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5c_rl_motor_runtime.wbt"


def main() -> None:
    content = SOURCE.read_text(encoding="utf-8")
    replacements = {
        'title "RiskAware SafeRL Stage 5A Live Camera Acquisition"': (
            'title "RiskAware SafeRL Stage 5C RL Motor Runtime"'
        ),
        'controller "live_camera_robot"': ('controller "rl_autonomous_inspection_robot"'),
        'controller "live_camera_supervisor"': ('controller "rl_autonomous_inspection_supervisor"'),
    }
    for old, new in replacements.items():
        if old not in content:
            raise RuntimeError(f"Stage 5C source token is missing: {old}")
        content = content.replace(old, new, 1)
    required = (
        "DEF HUMAN_VIEWPOINT Viewpoint {",
        "DEF SHOWCASE_ROBOT Robot {",
        'name "inspection camera"',
        'controller "rl_autonomous_inspection_robot"',
        'controller "rl_autonomous_inspection_supervisor"',
    )
    for token in required:
        if token not in content:
            raise RuntimeError(f"Generated Stage 5C world token is missing: {token}")
    DESTINATION.write_text(content, encoding="utf-8", newline="\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
