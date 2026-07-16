import pytest

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.scenarios import Scenario


def test_inspection_detects_adjacent_hazard_from_viewpoint() -> None:
    scenario = Scenario(
        scenario_id="viewpoint_000001",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=((0, 2),),
        workers=(),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )

    environment = ConstructionInspectionEnv(
        scenario=scenario,
        inspection_radius=2,
    )
    environment.reset(seed=0)

    _, reward, terminated, truncated, info = environment.step(4)

    assert terminated
    assert not truncated
    assert reward == pytest.approx(7.99)
    assert info["new_hazards"] == 1
    assert info["hazard_recall"] == pytest.approx(1.0)


def test_single_inspection_can_detect_multiple_visible_hazards() -> None:
    scenario = Scenario(
        scenario_id="viewpoint_000002",
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

    environment = ConstructionInspectionEnv(
        scenario=scenario,
        inspection_radius=2,
    )
    environment.reset(seed=0)

    _, reward, terminated, _, info = environment.step(4)

    assert terminated
    assert reward == pytest.approx(10.99)
    assert info["new_hazards"] == 2


def test_obstacle_blocks_viewpoint_inspection() -> None:
    scenario = Scenario(
        scenario_id="viewpoint_000003",
        split="test",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=((0, 1),),
        hazards=((0, 2),),
        workers=(),
        restricted_zones=(),
        max_steps=20,
        vision_radius=3,
    )

    environment = ConstructionInspectionEnv(
        scenario=scenario,
        inspection_radius=2,
    )
    environment.reset(seed=0)

    _, reward, terminated, _, info = environment.step(4)

    assert not terminated
    assert reward == pytest.approx(-0.11)
    assert info["new_hazards"] == 0
