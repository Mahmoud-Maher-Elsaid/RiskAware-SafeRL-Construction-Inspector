import pytest

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.evaluation.reward_audit import (
    RewardAuditWrapper,
)
from riskaware_saferrl.scenarios import Scenario


def test_reward_audit_handles_multiple_new_hazards() -> None:
    scenario = Scenario(
        scenario_id="viewpoint_reward_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((0, 1), (1, 0)),
        workers=(),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )

    environment = RewardAuditWrapper(
        ConstructionInspectionEnv(
            scenario=scenario,
            inspection_radius=2,
        )
    )
    environment.reset(seed=0)

    _, reward, terminated, _, info = environment.step(4)

    assert terminated
    assert reward == pytest.approx(10.99)
    assert info["reward_components"]["hazard_reward"] == 6.0
