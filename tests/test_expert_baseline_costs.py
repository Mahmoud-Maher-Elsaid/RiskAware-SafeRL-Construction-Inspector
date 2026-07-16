import pytest

from riskaware_saferrl.evaluation.expert_baselines import (
    run_action_sequence,
)
from riskaware_saferrl.scenarios import Scenario


def test_action_sequence_accumulates_worker_cost() -> None:
    scenario = Scenario(
        scenario_id="cost_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((5, 5),),
        workers=((0, 1),),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )

    record = run_action_sequence(
        scenario,
        [4, 4, 4],
    )

    assert record["worker_cost"] == pytest.approx(3.0)
    assert record["safety_cost"] == pytest.approx(3.0)
    assert record["collision_cost"] == pytest.approx(0.0)
    assert record["restricted_cost"] == pytest.approx(0.0)
