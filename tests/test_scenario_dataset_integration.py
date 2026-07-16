import numpy as np

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.scenario_dataset import ScenarioDataset, load_scenarios

DATASET_DIRECTORY = "data/scenarios"


def test_train_split_loads_all_scenarios() -> None:
    dataset = ScenarioDataset(DATASET_DIRECTORY, "train")

    assert len(dataset) == 1000
    assert dataset[0].split == "train"
    assert len(dataset.scenario_ids) == len(set(dataset.scenario_ids))


def test_fixed_scenario_is_loaded_exactly() -> None:
    scenario = load_scenarios(DATASET_DIRECTORY, "validation")[0]
    environment = ConstructionInspectionEnv(scenario=scenario)

    _, info = environment.reset(seed=999)

    assert environment.agent_position == scenario.agent_start
    assert environment.obstacles == set(scenario.obstacles)
    assert environment.hazards == set(scenario.hazards)
    assert environment.workers == set(scenario.workers)
    assert environment.restricted == set(scenario.restricted_zones)
    assert info["scenario_id"] == scenario.scenario_id
    assert info["scenario_split"] == "validation"


def test_scenario_pool_sampling_is_seed_reproducible() -> None:
    scenarios = load_scenarios(DATASET_DIRECTORY, "train")[:20]

    first = ConstructionInspectionEnv(scenarios=scenarios)
    second = ConstructionInspectionEnv(scenarios=scenarios)

    first_observation, first_info = first.reset(seed=1234)
    second_observation, second_info = second.reset(seed=1234)

    assert first_info["scenario_id"] == second_info["scenario_id"]
    np.testing.assert_array_equal(
        first_observation["map"],
        second_observation["map"],
    )
    np.testing.assert_array_equal(
        first_observation["state"],
        second_observation["state"],
    )


def test_scenario_index_option_replays_exact_layout() -> None:
    scenarios = load_scenarios(DATASET_DIRECTORY, "train")[:10]
    environment = ConstructionInspectionEnv(scenarios=scenarios)

    _, info = environment.reset(
        seed=1,
        options={"scenario_index": 7},
    )

    assert info["scenario_id"] == scenarios[7].scenario_id
    assert environment.agent_position == scenarios[7].agent_start
