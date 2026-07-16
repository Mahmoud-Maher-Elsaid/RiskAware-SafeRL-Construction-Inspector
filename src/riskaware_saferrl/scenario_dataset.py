from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from riskaware_saferrl.scenarios import Scenario


class ScenarioDataset(Sequence[Scenario]):
    """Load and verify one fixed scenario split."""

    def __init__(
        self,
        dataset_directory: str | Path,
        split: str,
        *,
        verify_hash: bool = True,
    ) -> None:
        self.dataset_directory = Path(dataset_directory)
        self.split = split

        manifest_path = self.dataset_directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest was not found: {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if split not in self.manifest["splits"]:
            available = ", ".join(sorted(self.manifest["splits"]))
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}")

        split_metadata = self.manifest["splits"][split]
        self.split_path = self.dataset_directory / split_metadata["path"]

        if not self.split_path.is_file():
            raise FileNotFoundError(f"Scenario split was not found: {self.split_path}")

        if verify_hash:
            actual_hash = hashlib.sha256(self.split_path.read_bytes()).hexdigest()
            expected_hash = str(split_metadata["sha256"])

            if actual_hash != expected_hash:
                raise ValueError(
                    f"SHA256 mismatch for split '{split}'. "
                    f"Expected {expected_hash}, received {actual_hash}."
                )

        self._scenarios = self._load_scenarios()

        expected_count = int(split_metadata["count"])
        if len(self._scenarios) != expected_count:
            raise ValueError(
                f"Scenario count mismatch for split '{split}'. "
                f"Expected {expected_count}, received {len(self._scenarios)}."
            )

        identifiers = [scenario.scenario_id for scenario in self._scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Duplicate scenario identifiers found in split '{split}'.")

    def _load_scenarios(self) -> tuple[Scenario, ...]:
        scenarios: list[Scenario] = []

        with self.split_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    scenario = Scenario.from_dict(json.loads(stripped))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid scenario at {self.split_path}:{line_number}"
                    ) from error

                if scenario.split != self.split:
                    raise ValueError(
                        f"Scenario {scenario.scenario_id} belongs to split "
                        f"'{scenario.split}', not '{self.split}'."
                    )

                scenarios.append(scenario)

        if not scenarios:
            raise ValueError(f"Split '{self.split}' contains no scenarios.")

        return tuple(scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, index: int) -> Scenario:
        return self._scenarios[index]

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self._scenarios)

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(scenario.scenario_id for scenario in self._scenarios)

    def get_by_id(self, scenario_id: str) -> Scenario:
        for scenario in self._scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario

        raise KeyError(f"Scenario was not found: {scenario_id}")


def load_scenarios(
    dataset_directory: str | Path,
    split: str,
    *,
    verify_hash: bool = True,
) -> tuple[Scenario, ...]:
    return tuple(
        ScenarioDataset(
            dataset_directory,
            split,
            verify_hash=verify_hash,
        )
    )
