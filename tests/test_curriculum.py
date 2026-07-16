from __future__ import annotations

import json

import pytest

from riskaware_saferrl.curriculum import (
    assign_training_tiers,
    load_curriculum_manifest,
    scenario_tiers_from_manifest,
    sha256_file,
)
from riskaware_saferrl.scenarios import Scenario


def create_scenario(identifier: str, hazard: tuple[int, int]) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        split="train",
        grid_size=6,
        agent_start=(0, 0),
        obstacles=(),
        hazards=(hazard,),
        workers=(),
        restricted_zones=(),
        max_steps=30,
        vision_radius=3,
    )


def test_assign_training_tiers_uses_balanced_rank_thirds() -> None:
    entries = [
        {
            "scenario_id": f"scenario_{index:02d}",
            "feasible": True,
            "tier": None,
            "difficulty_score": float(index),
            "plan_actions": index + 1,
        }
        for index in range(9)
    ]

    assigned = assign_training_tiers(entries)
    counts = {
        tier: sum(entry["tier"] == tier for entry in assigned)
        for tier in ("easy", "medium", "hard")
    }

    assert counts == {
        "easy": 3,
        "medium": 3,
        "hard": 3,
    }


def test_scenario_tiers_resolve_manifest_identifiers() -> None:
    scenarios = (
        create_scenario("easy_001", (0, 1)),
        create_scenario("medium_001", (2, 2)),
        create_scenario("hard_001", (5, 5)),
    )
    manifest = {
        "splits": {
            "train": {
                "entries": [
                    {
                        "scenario_id": "easy_001",
                        "feasible": True,
                        "tier": "easy",
                    },
                    {
                        "scenario_id": "medium_001",
                        "feasible": True,
                        "tier": "medium",
                    },
                    {
                        "scenario_id": "hard_001",
                        "feasible": True,
                        "tier": "hard",
                    },
                ]
            }
        }
    }

    tiers = scenario_tiers_from_manifest(scenarios, manifest)

    assert tiers["easy"][0].scenario_id == "easy_001"
    assert tiers["medium"][0].scenario_id == "medium_001"
    assert tiers["hard"][0].scenario_id == "hard_001"


def test_curriculum_manifest_verifies_source_hash(tmp_path) -> None:
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text('{"version":1}\n', encoding="utf-8")
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": sha256_file(source_manifest),
                },
                "splits": {},
            }
        ),
        encoding="utf-8",
    )

    load_curriculum_manifest(
        curriculum_path,
        source_manifest_path=source_manifest,
    )

    source_manifest.write_text('{"version":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_curriculum_manifest(
            curriculum_path,
            source_manifest_path=source_manifest,
        )
