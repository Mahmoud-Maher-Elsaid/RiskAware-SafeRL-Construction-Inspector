from __future__ import annotations

import pytest

from riskaware_saferrl.algorithms import (
    DualVariableController,
)


def test_dual_controller_warmup_and_bound() -> None:
    controller = DualVariableController(
        cost_limit=5.0,
        learning_rate=0.1,
        maximum=0.5,
        ema_beta=0.0,
        warmup_updates=2,
    )

    assert controller.update(20.0) == 0.0
    assert controller.update(20.0) == 0.0
    assert controller.update(20.0) == pytest.approx(0.3)
    assert controller.update(20.0) == pytest.approx(0.5)


def test_dual_controller_decreases_when_safe() -> None:
    controller = DualVariableController(
        cost_limit=5.0,
        learning_rate=0.1,
        maximum=2.0,
        ema_beta=0.0,
        warmup_updates=0,
        value=0.5,
    )

    assert controller.update(0.0) == pytest.approx(0.4)
