from __future__ import annotations

import math
import sys
from pathlib import Path

SUPERVISOR_DIR = (
    Path(__file__).resolve().parents[2]
    / "webots"
    / "controllers"
    / "hierarchical_experimental_supervisor"
)
sys.path.insert(0, str(SUPERVISOR_DIR))

from camera_math import camera_matrix_from_forward, determinant, matvec  # noqa: E402


def _yaw_forward(degrees: float) -> tuple[float, float, float]:
    angle = math.radians(degrees)
    # Physical body-forward is local -X; this is its world horizontal vector.
    return (-math.cos(angle), 0.0, -math.sin(angle))


def test_camera_basis_maps_view_axes_and_is_level() -> None:
    for degrees in (0.0, 90.0, -90.0, 180.0):
        matrix = camera_matrix_from_forward(_yaw_forward(degrees))
        forward = matvec(matrix, (0.0, 0.0, -1.0))
        up = matvec(matrix, (0.0, 1.0, 0.0))
        base = _yaw_forward(degrees)
        pitch = math.radians(10.0)
        expected = (base[0] * math.cos(pitch), -math.sin(pitch), base[2] * math.cos(pitch))
        assert max(abs(forward[i] - expected[i]) for i in range(3)) < 1.0e-6
        assert up[1] > 0.98
        assert abs(determinant(matrix) - 1.0) < 1.0e-6


def test_camera_basis_is_orthonormal() -> None:
    matrix = camera_matrix_from_forward(_yaw_forward(37.0))
    right = matvec(matrix, (1.0, 0.0, 0.0))
    up = matvec(matrix, (0.0, 1.0, 0.0))
    forward = matvec(matrix, (0.0, 0.0, -1.0))

    def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return sum(a[i] * b[i] for i in range(3))

    assert abs(dot(right, up)) < 1.0e-6
    assert abs(dot(right, forward)) < 1.0e-6
    assert abs(dot(up, forward)) < 1.0e-6
