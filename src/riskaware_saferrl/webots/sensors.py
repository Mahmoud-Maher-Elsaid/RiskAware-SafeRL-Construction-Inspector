from __future__ import annotations

import math


def clearance_meters(sensor: object) -> float:
    """Convert the visible-demo lookup-table response to metres."""
    value = float(sensor.getValue())
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
