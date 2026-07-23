from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(sys.argv[1]).resolve()

SOURCE_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"

ROUTE_CONFIG_PATH = PROJECT_ROOT / "configs" / "webots" / "stage5a3_closed_loop_route.json"

TARGET_WORLD_PATH = (
    PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a3_closed_loop_mission.wbt"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(
    path: Path,
) -> dict[str, Any]:
    require(
        path.is_file(),
        f"Required JSON file is missing: {path}",
    )

    return json.loads(path.read_text(encoding="utf-8"))


def marker_node(
    *,
    name: str,
    x: float,
    z: float,
    color: tuple[float, float, float],
) -> str:
    del name

    red, green, blue = color

    return f"""
Transform {{
  translation {x:.4f} 0.035 {z:.4f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {red:.3f} {green:.3f} {blue:.3f}
        roughness 0.8
        metalness 0
        transparency 0.18
      }}
      geometry Cylinder {{
        radius 0.24
        height 0.025
      }}
    }}
  ]
}}
""".strip()


def main() -> None:
    require(
        SOURCE_WORLD_PATH.is_file(),
        f"Stage 5A source world is missing: {SOURCE_WORLD_PATH}",
    )

    source = SOURCE_WORLD_PATH.read_text(encoding="utf-8")

    route_config = load_json(ROUTE_CONFIG_PATH)

    require(
        source.count('controller "live_camera_robot"') == 1,
        "Expected one live camera robot controller.",
    )

    require(
        source.count('controller "live_camera_supervisor"') == 1,
        "Expected one live camera supervisor controller.",
    )

    world = source.replace(
        'title "RiskAware SafeRL Stage 5A Live Camera Acquisition"',
        'title "RiskAware SafeRL Stage 5A3 Closed-Loop Inspection Mission"',
        1,
    )

    world = world.replace(
        'controller "live_camera_robot"',
        'controller "closed_loop_inspection_robot"',
        1,
    )

    world = world.replace(
        'controller "live_camera_supervisor"',
        'controller "closed_loop_inspection_supervisor"',
        1,
    )

    required_world_tokens = (
        'follow "professional construction inspection robot"',
        "followSmoothness 0.25",
        "followOrientation FALSE",
        "translation 0.38 0.46 0",
        "width 640",
        "height 360",
        "BallJoint {",
        'name "rear caster ball"',
        'name "front stabilizer ball"',
        "anchor -0.34 -0.055 0",
        "anchor 0 0 0.36",
        "anchor 0 0 -0.36",
        'contactMaterial "rearCaster"',
        'contactMaterial "driveWheel"',
        "damping Damping {",
        "maxTorque 2.5",
        "centerOfMass [",
        "-0.12 -0.055 0",
        "DEF SHOWCASE_ROBOT Robot {",
        'name "inspection camera"',
        'name "gps"',
        'name "inertial unit"',
        'name "compass"',
        'name "left wheel motor"',
        'name "right wheel motor"',
        'controller "closed_loop_inspection_robot"',
        'controller "closed_loop_inspection_supervisor"',
    )

    for token in required_world_tokens:
        require(
            token in world,
            f"Generated world token is missing: {token}",
        )

    require(
        "translation -0.25 -0.015 0" not in world,
        "The floating fixed caster is still present.",
    )

    waypoints = route_config["waypoints"]

    require(
        len(waypoints) >= 7,
        "The Stage 5A3 route is too short.",
    )

    palette = (
        (0.10, 0.72, 0.28),
        (0.10, 0.48, 0.92),
        (0.98, 0.66, 0.08),
        (0.88, 0.18, 0.18),
    )

    marker_nodes = [
        marker_node(
            name=str(waypoint["name"]),
            x=float(waypoint["x"]),
            z=float(waypoint["z"]),
            color=palette[index % len(palette)],
        )
        for index, waypoint in enumerate(waypoints)
    ]

    world = (
        world.rstrip()
        + "\n\n# Stage 5A3 closed-loop route markers\n"
        + "\n\n".join(marker_nodes)
        + "\n"
    )

    TARGET_WORLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TARGET_WORLD_PATH.write_text(
        world,
        encoding="utf-8",
        newline="\n",
    )

    print("Created Stage 5A3 closed-loop mission world:")
    print(TARGET_WORLD_PATH)
    print(f"World size: {TARGET_WORLD_PATH.stat().st_size} bytes")
    print(f"Route waypoints: {len(waypoints)}")


if __name__ == "__main__":
    main()
