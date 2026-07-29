from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import gymnasium as gym
import yaml
from gymnasium.utils.env_checker import check_env

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnv


def load_config(path: Path) -> GridEnvironmentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    return GridEnvironmentConfig(**{key: value for key, value in payload.items() if key in allowed})


def validate(path: Path, seed: int) -> dict[str, object]:
    config = load_config(path)
    environment = ResearchConstructionEnv(config)
    check_env(environment, skip_render_check=False)
    first_observation, first_info = environment.reset(seed=seed)
    second_environment = ResearchConstructionEnv(config)
    second_observation, second_info = second_environment.reset(seed=seed)
    deterministic_reset = all(
        (first_observation[key] == second_observation[key]).all() for key in first_observation
    )
    if not deterministic_reset or first_info != second_info:
        raise AssertionError("Seeded resets are not deterministic")
    action = int(next(index for index, valid in enumerate(environment.action_masks()) if valid))
    transition = environment.step(action)
    if not environment.observation_space.contains(transition[0]):
        raise AssertionError("Step observation does not match observation_space")
    state = environment.serialize_state()
    json.dumps(state)
    environment.close()
    second_environment.close()
    return {
        "status": "PASSED",
        "config": str(path),
        "seed": seed,
        "gymnasium_version": gym.__version__,
        "deterministic_reset": deterministic_reset,
        "observation_valid": True,
        "action_mask_valid": True,
        "serialization_valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=[
            Path("configs/grid/site_small.yaml"),
            Path("configs/grid/site_medium.yaml"),
            Path("configs/grid/site_dynamic.yaml"),
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/final_submission/stage1_grid_environment/validation.json"),
    )
    args = parser.parse_args()
    results = [validate(path, args.seed) for path in args.configs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASSED", "results": results}, indent=2) + "\n")
    print(f"GRID_ENVIRONMENT_VALIDATION=PASSED ({len(results)} configurations)")


if __name__ == "__main__":
    main()
