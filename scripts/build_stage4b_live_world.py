from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_WORLD = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4a.wbt"

LIVE_WORLD = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4b_live_bridge.wbt"


def split_top_level_blocks(
    body: str,
) -> list[str]:
    blocks: list[str] = []
    current_lines: list[str] = []
    brace_depth = 0

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not current_lines and not stripped:
            continue

        if not current_lines and stripped.startswith("#"):
            continue

        current_lines.append(line)

        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        if brace_depth < 0:
            raise RuntimeError("The source world contains unbalanced braces.")

        if brace_depth == 0:
            block = "\n".join(current_lines).strip()

            if block:
                blocks.append(block)

            current_lines = []

    if current_lines:
        raise RuntimeError("The source world ended with an incomplete node.")

    return blocks


def configure_world_info(
    block: str,
) -> str:
    lines = block.splitlines()
    coordinate_system_found = False

    for index, line in enumerate(lines):
        if line.strip().startswith("coordinateSystem "):
            lines[index] = '  coordinateSystem "EUN"'
            coordinate_system_found = True
            break

    if not coordinate_system_found:
        lines.insert(
            1,
            '  coordinateSystem "EUN"',
        )

    return "\n".join(lines)


source_content = SOURCE_WORLD.read_text(encoding="utf-8")

source_lines = source_content.splitlines()

if not source_lines:
    raise RuntimeError("The source world is empty.")

header = source_lines[0].strip()

if not header.startswith("#VRML_SIM"):
    raise RuntimeError("The source world header is invalid.")

blocks = split_top_level_blocks("\n".join(source_lines[1:]))

world_info_matches = [block for block in blocks if block.lstrip().startswith("WorldInfo {")]

robot_matches = [block for block in blocks if ("DEF CONSTRUCTION_ROBOT Robot" in block)]

supervisor_matches = [block for block in blocks if ('controller "scenario_supervisor"' in block)]

if len(world_info_matches) != 1:
    raise RuntimeError("Expected exactly one WorldInfo node.")

if len(robot_matches) != 1:
    raise RuntimeError("Expected exactly one construction robot.")

if len(supervisor_matches) != 1:
    raise RuntimeError("Expected exactly one Stage 4A Supervisor.")

original_world_info_block = world_info_matches[0]

configured_world_info_block = configure_world_info(original_world_info_block)

original_robot_block = robot_matches[0]
robot_block = original_robot_block

old_supervisor_block = supervisor_matches[0]

if 'controller "manual_robot_test"' not in robot_block:
    raise RuntimeError("The Stage 4A robot controller was not found.")

robot_block = robot_block.replace(
    'controller "manual_robot_test"',
    'controller "live_bridge_robot"',
    1,
)

device_nodes = """
    GPS {
      name "gps"
    }
    InertialUnit {
      name "inertial unit"
    }
    Compass {
      name "compass"
    }
    Emitter {
      name "bridge emitter"
      channel 7
    }
    Receiver {
      name "bridge receiver"
      channel 8
    }
""".rstrip()

if "children [" not in robot_block:
    raise RuntimeError("The construction robot has no children field.")

robot_block = robot_block.replace(
    "children [",
    "children [\n" + device_nodes,
    1,
)

new_supervisor_block = """
Robot {
  name "live bridge supervisor"
  supervisor TRUE
  controller "live_bridge_supervisor"
  children [
    Receiver {
      name "bridge receiver"
      channel 7
    }
    Emitter {
      name "bridge emitter"
      channel 8
    }
  ]
}
""".strip()

output_blocks: list[str] = []

for block in blocks:
    if block == original_world_info_block:
        output_blocks.append(configured_world_info_block)
        continue

    if block == original_robot_block:
        output_blocks.append(robot_block)
        continue

    if block == old_supervisor_block:
        continue

    output_blocks.append(block)

output_blocks.append(new_supervisor_block)

live_content = header + "\n\n" + "\n\n".join(output_blocks) + "\n"

required_fragments = (
    'coordinateSystem "EUN"',
    'controller "live_bridge_robot"',
    'controller "live_bridge_supervisor"',
    'name "gps"',
    'name "inertial unit"',
    'name "compass"',
    'name "bridge emitter"',
    'name "bridge receiver"',
)

for fragment in required_fragments:
    if fragment not in live_content:
        raise RuntimeError(f"Missing live world fragment: {fragment}")

if live_content.count('coordinateSystem "EUN"') != 1:
    raise RuntimeError("The Stage 4B2 world must contain exactly one EUN coordinate system.")

if 'controller "scenario_supervisor"' in live_content:
    raise RuntimeError("The old Supervisor is still present.")

LIVE_WORLD.write_text(
    live_content,
    encoding="utf-8",
    newline="\n",
)

print("Created Stage 4B2 live world with EUN coordinates:")
print(LIVE_WORLD)
