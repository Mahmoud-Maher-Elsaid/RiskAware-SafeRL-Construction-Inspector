import pytest

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.evaluation.reward_audit import RewardAuditWrapper
from riskaware_saferrl.scenarios import Scenario


def test_reward_audit_reconstructs_valid_inspection_reward() -> None:
    scenario = Scenario(
        scenario_id="reward_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((0, 1),),
        workers=(),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )

    environment = RewardAuditWrapper(ConstructionInspectionEnv(scenario=scenario))
    environment.reset(seed=0)

    _, movement_reward, terminated, truncated, _ = environment.step(3)

    assert not terminated
    assert not truncated
    assert movement_reward == pytest.approx(0.04)

    _, inspection_reward, terminated, truncated, info = environment.step(4)

    assert terminated
    assert not truncated
    assert inspection_reward == pytest.approx(7.99)
    assert info["reward_components"]["hazard_reward"] == 3.0
    assert info["reward_components"]["completion_reward"] == 5.0

    environment.close()


def test_reward_audit_reconstructs_invalid_inspection() -> None:
    scenario = Scenario(
        scenario_id="reward_000002",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((5, 5),),
        workers=(),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )

    environment = RewardAuditWrapper(ConstructionInspectionEnv(scenario=scenario))
    environment.reset(seed=0)

    _, reward, _, _, info = environment.step(4)

    assert reward == pytest.approx(-0.11)
    assert info["reward_components"]["invalid_inspection_penalty"] == -0.1

    environment.close()
