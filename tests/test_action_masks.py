import numpy as np

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.scenarios import Scenario


def create_scenario(
    *,
    scenario_id: str,
    agent_start: tuple[int, int] = (0, 0),
    obstacles: tuple[tuple[int, int], ...] = (),
    hazards: tuple[tuple[int, int], ...] = ((0, 2),),
    workers: tuple[tuple[int, int], ...] = (),
    restricted_zones: tuple[tuple[int, int], ...] = (),
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        split="test",
        grid_size=6,
        agent_start=agent_start,
        obstacles=obstacles,
        hazards=hazards,
        workers=workers,
        restricted_zones=restricted_zones,
        max_steps=20,
        vision_radius=3,
    )


def test_action_masks_remove_collisions_and_invalid_inspection() -> None:
    environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="mask_collision",
            obstacles=((1, 0),),
        ),
        inspection_radius=1,
    )
    environment.reset(seed=0)

    masks = environment.action_masks()

    assert masks.dtype == np.bool_
    assert masks.tolist() == [False, False, False, True, False]


def test_inspection_becomes_valid_at_a_viewpoint() -> None:
    environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="mask_inspection",
        ),
        inspection_radius=1,
    )
    environment.reset(seed=0)

    assert not bool(environment.action_masks()[4])

    environment.step(3)

    assert bool(environment.action_masks()[4])


def test_task_masks_keep_semantic_safety_risks_available() -> None:
    environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="mask_safety_separation",
            agent_start=(2, 2),
            hazards=((0, 0),),
            workers=((2, 3),),
            restricted_zones=((3, 2),),
        ),
        inspection_radius=1,
    )
    environment.reset(seed=0)

    masks = environment.action_masks()

    assert bool(masks[1])
    assert bool(masks[3])
    assert not environment.is_action_safe(1)
    assert not environment.is_action_safe(3)


def test_action_masks_keep_a_fallback_for_trapped_states() -> None:
    environment = ConstructionInspectionEnv(
        scenario=create_scenario(
            scenario_id="mask_fallback",
            hazards=((5, 5),),
        ),
        inspection_radius=1,
    )
    environment.reset(seed=0)

    environment.obstacles = {
        (0, 1),
        (1, 0),
    }

    masks = environment.action_masks()

    assert masks.tolist() == [False, False, False, False, True]
    assert environment.task_valid_actions() == [4]
