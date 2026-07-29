from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "webots/worlds/construction_site_stage5c_rl_motor_runtime.wbt"
CONFIG_DIRECTORY = ROOT / "configs/webots"
WORLD_DIRECTORY = ROOT / "webots/worlds"

EXTRA_STATIC_ASSETS = """
DEF FINAL_RESTRICTED_AREA Pose {
  translation 8.5 0.035 6.5
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.75 0.08 0.06 roughness 0.8 }
      geometry Box { size 3.2 0.03 2.4 }
    }
  ]
}
DEF FINAL_FALL_HAZARD Solid {
  translation 9.0 0.12 4.2
  name "guarded fall hazard"
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.05 0.05 0.05 roughness 1 }
      geometry Cylinder { radius 0.7 height 0.12 }
    }
  ]
  boundingObject Cylinder { radius 0.7 height 0.12 }
  locked TRUE
}
"""

CONTEXT_HOARDING = """
DEF FINAL_PERIMETER_HOARDING Solid {
  translation 11.7 1.5 0
  name "construction perimeter hoarding"
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.28 0.34 0.40 roughness 0.9 }
      geometry Box { size 0.18 3 18 }
    }
  ]
  boundingObject Box { size 0.18 3 18 }
  locked TRUE
}
"""

DYNAMIC_WORKER = """
DEF FINAL_DYNAMIC_WORKER Robot {
  translation 0 0.85 7.4
  supervisor TRUE
  name "deterministic moving worker"
  controller "final_dynamic_worker"
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.95 0.52 0.08 roughness 0.7 }
      geometry Capsule { radius 0.22 height 1.25 }
    }
  ]
  boundingObject Capsule { radius 0.22 height 1.25 }
}
"""


def build(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    content = SOURCE.read_text(encoding="utf-8")
    content = content.replace(
        'title "RiskAware SafeRL Stage 5C RL Motor Runtime"',
        f'title "{config["title"]}"',
        1,
    )
    width, depth = config["environment_size_m"]
    content = content.replace(
        "size 24 0.24 18",
        f"size {width:g} 0.24 {depth:g}",
        2,
    )
    metadata = (
        f'  info ["final_world={config["name"]}", '
        f'"hazard_density={config["hazard_density"]}", '
        f'"worker_density={config["worker_density"]}", '
        f'"dynamic_obstacles={str(config["dynamic_obstacles"]).lower()}", '
        f'"expected_inspection_targets={config["expected_inspection_targets"]}"]\n'
    )
    content = content.replace("WorldInfo {\n", "WorldInfo {\n" + metadata, 1)
    content = content.replace(
        "DEF SHOWCASE_ROBOT Robot {",
        CONTEXT_HOARDING + "\nDEF SHOWCASE_ROBOT Robot {",
        1,
    )
    if config["name"] != "site_small":
        content = content.replace(
            "DEF SHOWCASE_ROBOT Robot {", EXTRA_STATIC_ASSETS + "\nDEF SHOWCASE_ROBOT Robot {", 1
        )
    if config["dynamic_obstacles"]:
        content = content.replace(
            "DEF SHOWCASE_ROBOT Robot {", DYNAMIC_WORKER + "\nDEF SHOWCASE_ROBOT Robot {", 1
        )
    required = (
        "DEF HUMAN_VIEWPOINT Viewpoint {",
        'controller "rl_autonomous_inspection_robot"',
        'controller "rl_autonomous_inspection_supervisor"',
        f"final_world={config['name']}",
    )
    for token in required:
        if token not in content:
            raise RuntimeError(f"Final world token is missing: {token}")
    destination = WORLD_DIRECTORY / f"{config['name']}.wbt"
    destination.write_text(content, encoding="utf-8", newline="\n")
    return destination


def main() -> None:
    outputs = [
        build(CONFIG_DIRECTORY / f"{name}.yaml")
        for name in ("site_small", "site_medium", "site_dynamic")
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
