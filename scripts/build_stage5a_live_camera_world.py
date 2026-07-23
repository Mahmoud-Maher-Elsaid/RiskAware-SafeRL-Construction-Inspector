from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(sys.argv[1]).resolve()

SOURCE_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4d_showcase.wbt"

TARGET_WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage5a_live_camera.wbt"

ROBOT_NAME = "professional construction inspection robot"

FIXED_CASTER_VISUAL_BLOCKS = (
    """    Pose {
      translation -0.25 -0.015 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.08 0.08 0.09
            roughness 0.5
            metalness 0.4
          }
          geometry Sphere {
            radius 0.065
            subdivision 3
          }
        }
      ]
    }
""",
    """    Pose {
      translation -0.25 -0.055 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.08 0.08 0.09
            roughness 0.5
            metalness 0.4
          }
          geometry Sphere {
            radius 0.065
            subdivision 3
          }
        }
      ]
    }
""",
)

FIXED_CASTER_BOUNDING_BLOCKS = (
    """      Pose {
        translation -0.25 -0.015 0
        children [
          Sphere {
            radius 0.065
            subdivision 2
          }
        ]
      }
""",
    """      Pose {
        translation -0.25 -0.055 0
        children [
          Sphere {
            radius 0.065
            subdivision 2
          }
        ]
      }
""",
)

BALL_CASTER_BLOCK = """    BallJoint {
      jointParameters BallJointParameters {
        anchor -0.25 -0.055 0
      }
      endPoint Solid {
        translation -0.25 -0.055 0
        name "rear caster ball"
        contactMaterial "rearCaster"
        children [
          Shape {
            appearance PBRAppearance {
              baseColor 0.08 0.08 0.09
              roughness 0.5
              metalness 0.4
            }
            geometry Sphere {
              radius 0.065
              subdivision 3
            }
          }
        ]
        boundingObject Sphere {
          radius 0.065
          subdivision 2
        }
        physics Physics {
          density -1
          mass 0.04
        }
      }
    }
    BallJoint {
      jointParameters BallJointParameters {
        anchor 0.34 -0.055 0
      }
      endPoint Solid {
        translation 0.34 -0.055 0
        name "front stabilizer ball"
        contactMaterial "rearCaster"
        children [
          Shape {
            appearance PBRAppearance {
              baseColor 0.08 0.08 0.09
              roughness 0.5
              metalness 0.4
            }
            geometry Sphere {
              radius 0.065
              subdivision 3
            }
          }
        ]
        boundingObject Sphere {
          radius 0.065
          subdivision 2
        }
        physics Physics {
          density -1
          mass 0.04
        }
      }
    }
"""


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def replace_one_of(
    content: str,
    candidates: tuple[str, ...],
    replacement: str,
    *,
    name: str,
) -> str:
    matches = [candidate for candidate in candidates if candidate in content]

    require(
        len(matches) == 1,
        (f"Expected exactly one {name} representation. Found {len(matches)}."),
    )

    matched = matches[0]

    require(
        content.count(matched) == 1,
        f"The {name} block is not unique.",
    )

    return content.replace(
        matched,
        replacement,
        1,
    )


def configure_camera(
    world: str,
) -> str:
    camera_matches = list(
        re.finditer(
            r"Camera\s*\{.*?\n\s{4}\}",
            world,
            flags=re.DOTALL,
        )
    )

    inspection_matches = [
        match for match in camera_matches if 'name "inspection camera"' in match.group(0)
    ]

    require(
        len(inspection_matches) == 1,
        (f"Expected exactly one inspection camera block. Found {len(inspection_matches)}."),
    )

    camera_match = inspection_matches[0]
    camera_block = camera_match.group(0)

    camera_block = re.sub(
        r"\btranslation\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+",
        "translation 0.38 0.46 0",
        camera_block,
        count=1,
    )

    camera_block = re.sub(
        r"\bwidth\s+\d+",
        "width 640",
        camera_block,
        count=1,
    )

    camera_block = re.sub(
        r"\bheight\s+\d+",
        "height 360",
        camera_block,
        count=1,
    )

    camera_block = re.sub(
        r"\bfieldOfView\s+[0-9.]+",
        "fieldOfView 1.05",
        camera_block,
        count=1,
    )

    for token in (
        "translation 0.38 0.46 0",
        "width 640",
        "height 360",
        "fieldOfView 1.05",
    ):
        require(
            token in camera_block,
            f"Camera token is missing: {token}",
        )

    return world[: camera_match.start()] + camera_block + world[camera_match.end() :]


def configure_viewpoint(
    world: str,
) -> str:
    viewpoint_replacement = f"""Viewpoint {{
  orientation -0.57735 0.57735 0.57735 2.0944
  position -3.8 4.1 -0.9
  fieldOfView 0.88
  follow "{ROBOT_NAME}"
  followSmoothness 0.25
  followOrientation FALSE
}}"""

    updated, count = re.subn(
        r"Viewpoint\s*\{.*?\n\}",
        viewpoint_replacement,
        world,
        count=1,
        flags=re.DOTALL,
    )

    require(
        count == 1,
        "Could not configure the tracking Viewpoint.",
    )

    return updated


def configure_body_physics(
    world: str,
) -> str:
    replacement = """physics Physics {
    density -1
    mass 4.5
    centerOfMass [
      -0.08 0.035 0
    ]
  }"""

    updated, count = re.subn(
        (
            r"physics Physics\s*\{\s*"
            r"density -1\s*"
            r"mass 4\.5\s*"
            r"(?:centerOfMass\s*\[[^\]]*\]\s*)?"
            r"\}"
        ),
        replacement,
        world,
        count=1,
        flags=re.DOTALL,
    )

    require(
        count == 1,
        "Could not configure the main robot physics.",
    )

    return updated


def apply_final_stability_model(
    world: str,
) -> str:
    world_info = """WorldInfo {
  coordinateSystem "EUN"
  basicTimeStep 32
  title "RiskAware SafeRL Stage 5A Live Camera Acquisition"
  contactProperties [
    ContactProperties {
      material1 "driveWheel"
      material2 "default"
      coulombFriction [ 0.6 ]
      rollingFriction 0 0 0
      bounce 0
    }
    ContactProperties {
      material1 "rearCaster"
      material2 "default"
      coulombFriction [ 0 ]
      rollingFriction 0 0 0
      bounce 0
    }
  ]
}"""

    world, count = re.subn(
        r"WorldInfo\s*\{.*?\n\}",
        world_info,
        world,
        count=1,
        flags=re.DOTALL,
    )

    require(
        count == 1,
        "Could not configure the Stage 5A contact model.",
    )

    world = world.replace(
        "anchor -0.25 -0.055 0",
        "anchor -0.34 -0.055 0",
        1,
    )
    world = world.replace(
        "translation -0.25 -0.055 0",
        "translation -0.34 -0.055 0",
        1,
    )
    world = world.replace("0 0 0.245", "0 0 0.36")
    world = world.replace("0 0 -0.245", "0 0 -0.36")

    for wheel_name in (
        "left showcase wheel",
        "right showcase wheel",
    ):
        original = f'name "{wheel_name}"\n        children ['
        corrected = f'name "{wheel_name}"\n        contactMaterial "driveWheel"\n        children ['

        require(
            world.count(original) == 1,
            f"Could not locate {wheel_name}.",
        )
        world = world.replace(
            original,
            corrected,
            1,
        )

    require(
        world.count("maxTorque 2.5") == 2,
        "Expected two original wheel torque limits.",
    )
    body_physics = """physics Physics {
    density -1
    mass 4.5
    centerOfMass [
      -0.12 -0.055 0
    ]
    damping Damping {
      linear 0.08
      angular 0.85
    }
  }"""

    world, count = re.subn(
        (
            r"physics Physics\s*\{\s*"
            r"density -1\s*"
            r"mass 4\.5\s*"
            r"(?:centerOfMass\s*\[[^\]]*\]\s*)?"
            r"(?:damping\s+Damping\s*\{[^}]*\}\s*)?"
            r"\}"
        ),
        body_physics,
        world,
        count=1,
        flags=re.DOTALL,
    )

    require(
        count == 1,
        "Could not configure the main robot damping.",
    )

    return world


def main() -> None:
    require(
        SOURCE_WORLD_PATH.is_file(),
        f"Source world is missing: {SOURCE_WORLD_PATH}",
    )

    source = SOURCE_WORLD_PATH.read_text(encoding="utf-8")

    require(
        source.count('controller "showcase_robot"') == 1,
        "Expected one showcase robot controller.",
    )

    require(
        source.count('controller "showcase_supervisor"') == 1,
        "Expected one showcase supervisor controller.",
    )

    world = source.replace(
        'title "RiskAware SafeRL Professional Construction Showcase"',
        'title "RiskAware SafeRL Stage 5A Live Camera Acquisition"',
        1,
    )

    world = world.replace(
        'controller "showcase_robot"',
        'controller "live_camera_robot"',
        1,
    )

    world = world.replace(
        'controller "showcase_supervisor"',
        'controller "live_camera_supervisor"',
        1,
    )

    world = configure_camera(world)
    world = configure_viewpoint(world)

    world = replace_one_of(
        world,
        FIXED_CASTER_VISUAL_BLOCKS,
        BALL_CASTER_BLOCK,
        name="rear caster visual",
    )

    world = replace_one_of(
        world,
        FIXED_CASTER_BOUNDING_BLOCKS,
        "",
        name="rear caster parent bounding object",
    )

    world = configure_body_physics(world)

    world = apply_final_stability_model(world)

    required_tokens = (
        "DEF SHOWCASE_ROBOT Robot {",
        f'name "{ROBOT_NAME}"',
        'name "inspection camera"',
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
        'controller "live_camera_robot"',
        'controller "live_camera_supervisor"',
        'name "gps"',
        'name "inertial unit"',
        'name "compass"',
        'name "left wheel motor"',
        'name "right wheel motor"',
    )

    for token in required_tokens:
        require(
            token in world,
            f"Generated world token is missing: {token}",
        )

    prohibited_tokens = (
        'controller "showcase_robot"',
        'controller "showcase_supervisor"',
        "translation -0.25 -0.015 0",
    )

    for token in prohibited_tokens:
        require(
            token not in world,
            f"Deprecated world token remains: {token}",
        )

    TARGET_WORLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TARGET_WORLD_PATH.write_text(
        world.rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("Created Stage 5A live camera world:")
    print(TARGET_WORLD_PATH)
    print(f"World size: {TARGET_WORLD_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
