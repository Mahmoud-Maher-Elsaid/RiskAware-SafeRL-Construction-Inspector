from pathlib import Path

import pytest

from riskaware_saferrl.scenarios import (
    Scenario,
    SplitConfig,
    generate_dataset,
    validate_dataset,
)


def test_dataset_generation_is_valid(tmp_path: Path) -> None:
    configs = {
        "train": SplitConfig(5, (4, 6), (2, 3), (1, 2), (2, 3), 8),
        "test": SplitConfig(3, (7, 9), (3, 4), (2, 3), (3, 5), 8),
    }

    generate_dataset(
        tmp_path / "dataset",
        seed=1234,
        split_configs=configs,
    )
    results = validate_dataset(tmp_path / "dataset")

    assert results["total_scenarios"] == 8
    assert results["unique_layouts"] == 8


def test_dataset_generation_is_deterministic(tmp_path: Path) -> None:
    configs = {
        "train": SplitConfig(5, (4, 6), (2, 3), (1, 2), (2, 3), 8),
    }

    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_dataset(first, seed=999, split_configs=configs)
    generate_dataset(second, seed=999, split_configs=configs)

    assert (first / "train" / "train.jsonl").read_text(encoding="utf-8") == (
        second / "train" / "train.jsonl"
    ).read_text(encoding="utf-8")


def test_scenario_rejects_overlap() -> None:
    scenario = Scenario(
        scenario_id="invalid_000001",
        split="invalid",
        grid_size=8,
        agent_start=(0, 0),
        obstacles=((1, 1),),
        hazards=((1, 1),),
        workers=((2, 2),),
        restricted_zones=((3, 3),),
    )

    with pytest.raises(ValueError, match="overlaps"):
        scenario.validate()
