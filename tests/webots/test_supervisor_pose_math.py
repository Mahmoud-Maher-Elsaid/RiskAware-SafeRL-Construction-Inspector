from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPERVISOR_PATH = (
    PROJECT_ROOT / "webots" / "controllers" / "scenario_supervisor" / "scenario_supervisor.py"
)


def load_supervisor_module() -> ModuleType:
    source = SUPERVISOR_PATH.read_text(encoding="utf-8")

    if "def main() -> None:" not in source:
        raise RuntimeError("The Supervisor controller does not contain main().")

    helper_source = source.split(
        "def main() -> None:",
        maxsplit=1,
    )[0]

    helper_source = helper_source.replace(
        "from controller import Supervisor\n",
        "",
    )

    namespace: dict[str, object] = {
        "__file__": str(SUPERVISOR_PATH),
        "__name__": "scenario_supervisor_helpers",
    }

    exec(
        compile(
            helper_source,
            str(SUPERVISOR_PATH),
            "exec",
        ),
        namespace,
    )

    module = ModuleType("scenario_supervisor_helpers")

    for name, value in namespace.items():
        setattr(module, name, value)

    return module


def test_zero_rotation_axes_are_equivalent() -> None:
    module = load_supervisor_module()

    assert module.rotations_equivalent(
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        tolerance=1e-6,
    )


def test_equivalent_axis_angle_signs_match() -> None:
    module = load_supervisor_module()

    assert module.rotations_equivalent(
        [0.0, 1.0, 0.0, 0.5],
        [0.0, -1.0, 0.0, -0.5],
        tolerance=1e-6,
    )


def test_different_rotations_do_not_match() -> None:
    module = load_supervisor_module()

    assert not module.rotations_equivalent(
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        tolerance=1e-6,
    )


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (
            [0.0, 0.04, 0.0],
            [0.002, 0.041, -0.002],
            True,
        ),
        (
            [0.0, 0.04, 0.0],
            [0.02, 0.04, 0.0],
            False,
        ),
    ],
)
def test_position_tolerance(
    first: list[float],
    second: list[float],
    expected: bool,
) -> None:
    module = load_supervisor_module()

    result = module.positions_equivalent(
        first,
        second,
        tolerance=0.005,
    )

    assert result is expected
