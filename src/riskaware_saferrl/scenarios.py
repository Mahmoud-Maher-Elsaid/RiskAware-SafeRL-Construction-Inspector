from __future__ import annotations

import hashlib
import json
import shutil
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

Position = tuple[int, int]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    grid_size: int
    agent_start: Position
    obstacles: tuple[Position, ...]
    hazards: tuple[Position, ...]
    workers: tuple[Position, ...]
    restricted_zones: tuple[Position, ...]
    max_steps: int = 250
    vision_radius: int = 3

    def validate(self) -> None:
        if self.grid_size < 6:
            raise ValueError("grid_size must be at least 6")
        if not self.hazards:
            raise ValueError("Each scenario must contain at least one hazard")

        groups = {
            "agent_start": (self.agent_start,),
            "obstacles": self.obstacles,
            "hazards": self.hazards,
            "workers": self.workers,
            "restricted_zones": self.restricted_zones,
        }

        occupied: dict[Position, str] = {}
        for group_name, positions in groups.items():
            if len(set(positions)) != len(positions):
                raise ValueError(f"Duplicate positions found in {group_name}")

            for position in positions:
                row, column = position
                if not (0 <= row < self.grid_size and 0 <= column < self.grid_size):
                    raise ValueError(f"Position is outside the grid: {position}")

                if position in occupied:
                    raise ValueError(
                        f"Position {position} overlaps between "
                        f"{occupied[position]} and {group_name}"
                    )
                occupied[position] = group_name

        if not set(self.hazards).issubset(self.reachable_positions()):
            raise ValueError("One or more hazards are unreachable")

    def reachable_positions(self) -> set[Position]:
        obstacles = set(self.obstacles)
        frontier: deque[Position] = deque([self.agent_start])
        visited = {self.agent_start}

        while frontier:
            row, column = frontier.popleft()
            for row_delta, column_delta in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ):
                candidate = (row + row_delta, column + column_delta)
                if candidate in visited or candidate in obstacles:
                    continue
                if not (0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size):
                    continue
                visited.add(candidate)
                frontier.append(candidate)

        return visited

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "grid_size": self.grid_size,
            "agent_start": list(self.agent_start),
            "obstacles": [list(value) for value in self.obstacles],
            "hazards": [list(value) for value in self.hazards],
            "workers": [list(value) for value in self.workers],
            "restricted_zones": [list(value) for value in self.restricted_zones],
            "max_steps": self.max_steps,
            "vision_radius": self.vision_radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        def position(value: list[int]) -> Position:
            return int(value[0]), int(value[1])

        scenario = cls(
            scenario_id=str(data["scenario_id"]),
            split=str(data["split"]),
            grid_size=int(data["grid_size"]),
            agent_start=position(data["agent_start"]),
            obstacles=tuple(position(value) for value in data["obstacles"]),
            hazards=tuple(position(value) for value in data["hazards"]),
            workers=tuple(position(value) for value in data["workers"]),
            restricted_zones=tuple(position(value) for value in data["restricted_zones"]),
            max_steps=int(data["max_steps"]),
            vision_radius=int(data["vision_radius"]),
        )
        scenario.validate()
        return scenario

    def fingerprint(self) -> str:
        layout = self.to_dict()
        layout.pop("scenario_id")
        layout.pop("split")
        payload = json.dumps(
            layout,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SplitConfig:
    count: int
    obstacle_range: tuple[int, int]
    hazard_range: tuple[int, int]
    worker_range: tuple[int, int]
    restricted_range: tuple[int, int]
    grid_size: int = 12
    max_steps: int = 250
    vision_radius: int = 3


DEFAULT_SPLITS = {
    "train": SplitConfig(1000, (14, 20), (5, 7), (3, 5), (6, 10)),
    "validation": SplitConfig(200, (14, 20), (5, 7), (3, 5), (6, 10)),
    "test_seen": SplitConfig(200, (14, 20), (5, 7), (3, 5), (6, 10)),
    "test_unseen": SplitConfig(200, (21, 26), (7, 9), (5, 7), (11, 15)),
    "stress": SplitConfig(
        300,
        (27, 32),
        (8, 10),
        (6, 8),
        (14, 18),
        max_steps=300,
    ),
}


def _random_count(
    rng: np.random.Generator,
    value_range: tuple[int, int],
) -> int:
    return int(rng.integers(value_range[0], value_range[1] + 1))


def _generate_scenario(
    split: str,
    index: int,
    config: SplitConfig,
    rng: np.random.Generator,
    fingerprints: set[str],
) -> Scenario:
    for _ in range(10_000):
        counts = [
            _random_count(rng, config.obstacle_range),
            _random_count(rng, config.hazard_range),
            _random_count(rng, config.worker_range),
            _random_count(rng, config.restricted_range),
        ]
        total = 1 + sum(counts)
        cell_count = config.grid_size * config.grid_size

        selected = rng.choice(cell_count, size=total, replace=False)
        positions = [
            (int(value // config.grid_size), int(value % config.grid_size)) for value in selected
        ]

        cursor = 1
        obstacle_count, hazard_count, worker_count, restricted_count = counts

        scenario = Scenario(
            scenario_id=f"{split}_{index:06d}",
            split=split,
            grid_size=config.grid_size,
            agent_start=positions[0],
            obstacles=tuple(positions[cursor : cursor + obstacle_count]),
            hazards=tuple(
                positions[cursor + obstacle_count : cursor + obstacle_count + hazard_count]
            ),
            workers=tuple(
                positions[
                    cursor + obstacle_count + hazard_count : cursor
                    + obstacle_count
                    + hazard_count
                    + worker_count
                ]
            ),
            restricted_zones=tuple(positions[total - restricted_count :]),
            max_steps=config.max_steps,
            vision_radius=config.vision_radius,
        )

        try:
            scenario.validate()
        except ValueError:
            continue

        fingerprint = scenario.fingerprint()
        if fingerprint in fingerprints:
            continue

        fingerprints.add(fingerprint)
        return scenario

    raise RuntimeError(f"Could not generate scenario {split}_{index:06d}")


def generate_dataset(
    output_directory: str | Path,
    *,
    seed: int = 20260716,
    split_configs: dict[str, SplitConfig] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_directory)
    configs = split_configs or DEFAULT_SPLITS

    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_path}")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fingerprints: set[str] = set()
    manifest_splits: dict[str, Any] = {}

    for split, config in configs.items():
        scenarios = [
            _generate_scenario(split, index, config, rng, fingerprints)
            for index in range(config.count)
        ]

        split_path = output_path / split / f"{split}.jsonl"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()

        with split_path.open("w", encoding="utf-8", newline="\n") as file:
            for scenario in scenarios:
                line = json.dumps(
                    scenario.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                file.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))

        manifest_splits[split] = {
            "path": split_path.relative_to(output_path).as_posix(),
            "count": len(scenarios),
            "sha256": digest.hexdigest(),
            "config": asdict(config),
        }

    manifest = {
        "dataset_name": "RiskAware Construction Scenario Dataset",
        "dataset_version": "1.0.0",
        "generator_seed": seed,
        "total_scenarios": sum(value["count"] for value in manifest_splits.values()),
        "splits": manifest_splits,
    }

    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_dataset(dataset_directory: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_directory)
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))

    fingerprints: set[str] = set()
    split_counts: dict[str, int] = {}

    for split, split_data in manifest["splits"].items():
        split_path = dataset_path / split_data["path"]
        actual_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
        if actual_hash != split_data["sha256"]:
            raise ValueError(f"SHA256 mismatch for split: {split}")

        count = 0
        with split_path.open("r", encoding="utf-8") as file:
            for line in file:
                scenario = Scenario.from_dict(json.loads(line))
                if scenario.split != split:
                    raise ValueError(f"Incorrect split for {scenario.scenario_id}")
                fingerprint = scenario.fingerprint()
                if fingerprint in fingerprints:
                    raise ValueError(f"Duplicate layout: {scenario.scenario_id}")
                fingerprints.add(fingerprint)
                count += 1

        if count != split_data["count"]:
            raise ValueError(f"Count mismatch for split: {split}")
        split_counts[split] = count

    total = sum(split_counts.values())
    if total != manifest["total_scenarios"]:
        raise ValueError("Total scenario count does not match the manifest")

    return {
        "dataset_name": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "generator_seed": manifest["generator_seed"],
        "total_scenarios": total,
        "unique_layouts": len(fingerprints),
        "splits": split_counts,
    }
