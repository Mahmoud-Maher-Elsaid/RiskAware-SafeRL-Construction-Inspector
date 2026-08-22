"""Pure geometry for the robot-mounted human-view camera.

The camera is visualization-only.  Webots Viewpoint looks along local -Z;
the returned row-major rotation therefore has columns (right, up, -forward).
"""

from __future__ import annotations

import math

Vector = tuple[float, float, float]
Matrix = tuple[float, ...]

# Derived from the visible-demo robot body: body X extent is +/-0.31 m and
# the top is about 0.41 m above the robot root.  Keep the rider camera just
# forward of the center and comfortably above the body.
CAMERA_FORWARD_OFFSET_M = -0.08
CAMERA_HEIGHT_OFFSET_M = 0.82
CAMERA_LATERAL_OFFSET_M = 0.0
CAMERA_DOWNWARD_PITCH_RAD = math.radians(10.0)


def normalize(vector: Vector) -> Vector:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1.0e-12:
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(sum(matrix[row * 3 + col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def horizontal(vector: Vector) -> Vector:
    return normalize((vector[0], 0.0, vector[2]))


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Vector, b: Vector) -> float:
    return sum(a[index] * b[index] for index in range(3))


def camera_matrix_from_forward(forward_world: Vector) -> Matrix:
    """Build a level, gently downward-looking camera frame."""
    forward = horizontal(forward_world)
    world_up: Vector = (0.0, 1.0, 0.0)
    look = normalize(
        tuple(
            forward[i] * math.cos(CAMERA_DOWNWARD_PITCH_RAD)
            - world_up[i] * math.sin(CAMERA_DOWNWARD_PITCH_RAD)
            for i in range(3)
        )
    )
    right = normalize(cross(look, world_up))
    up = normalize(cross(right, look))
    negative_look = tuple(-value for value in look)
    # Columns are camera-local +X, +Y, +Z in world coordinates.
    matrix: Matrix = (
        right[0],
        up[0],
        negative_look[0],
        right[1],
        up[1],
        negative_look[1],
        right[2],
        up[2],
        negative_look[2],
    )
    if abs(determinant(matrix) - 1.0) > 1.0e-6:
        raise ValueError("camera basis is not a proper rotation")
    return matrix


def determinant(matrix: Matrix) -> float:
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    )


def axis_angle_from_matrix(matrix: Matrix) -> list[float]:
    trace = max(-1.0, min(3.0, matrix[0] + matrix[4] + matrix[8]))
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) * 0.5)))
    if angle < 1.0e-8:
        return [0.0, 1.0, 0.0, 0.0]
    scale = 2.0 * math.sin(angle)
    axis = [
        (matrix[7] - matrix[5]) / scale,
        (matrix[2] - matrix[6]) / scale,
        (matrix[3] - matrix[1]) / scale,
    ]
    norm = math.sqrt(sum(value * value for value in axis))
    if norm < 1.0e-8:
        return [0.0, 1.0, 0.0, angle]
    return [axis[0] / norm, axis[1] / norm, axis[2] / norm, angle]
