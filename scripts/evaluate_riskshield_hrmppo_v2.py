from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.policies import RecurrentMaskedPolicy
from riskaware_saferrl.safety import PredictiveSafetyShield

WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = {
    "low": (0.75, 0.75, 0.0),
    "medium": (1.0, 1.0, 0.1),
    "high": (1.5, 1.5, 0.2),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(world: str, profile: str) -> GridEnvironmentConfig:
    hazard_multiplier, worker_multiplier, noise = PROFILES[profile]
    payload = yaml.safe_load(Path(f"configs/grid/{world}.yaml").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    payload["hazard_density"] = min(0.5, payload["hazard_density"] * hazard_multiplier)
    payload["worker_density"] = min(0.5, payload["worker_density"] * worker_multiplier)
    payload["perception_false_negative_rate"] = noise
    return GridEnvironmentConfig(**payload)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = RecurrentMaskedPolicy().to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    rows: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for world in WORLDS:
            for profile in PROFILES:
                environment = ResearchConstructionEnvV2(load_config(world, profile))
                observation, _ = environment.reset(seed=seed)
                hidden = policy.initial_state(1, device)
                shield = PredictiveSafetyShield(
                    horizon=args.shield_horizon,
                    safety_budget=args.shield_step_budget,
                )
                invalid_actions = shield_interventions = 0
                while True:
                    maps = (
                        torch.as_tensor(observation["map"], device=device).unsqueeze(0).unsqueeze(0)
                    )
                    states = (
                        torch.as_tensor(observation["state"], device=device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    masks = (
                        torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    action, output = policy.predict(maps, states, masks, hidden)
                    hidden = output.recurrent_state
                    proposed = int(action.item())
                    invalid_actions += int(not bool(observation["action_mask"][proposed]))
                    decision = shield.decide(environment, proposed)
                    shield_interventions += int(decision.shield_decision != "accept")
                    observation, _, terminated, truncated, info = environment.step(
                        decision.final_action
                    )
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "seed": seed,
                        "world": world,
                        "profile": profile,
                        "success": bool(info["success"]),
                        "hazard_recall": float(info["hazard_recall"]),
                        "coverage": float(info["inspection_coverage"]),
                        "collision_rate": environment.collisions / max(1, environment.steps),
                        "collisions": environment.collisions,
                        "safety_cost": environment.cumulative_cost,
                        "constraint_violations": (
                            environment.collisions
                            + environment.near_misses
                            + environment.restricted_violations
                        ),
                        "steps": environment.steps,
                        "invalid_actions": invalid_actions,
                        "shield_interventions": shield_interventions,
                    }
                )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    hardest = [row for row in rows if row["world"] == "site_dynamic" and row["profile"] == "high"]
    target = [row for row in rows if row["world"] == "site_small"]
    overall_success = float(np.mean([row["success"] for row in rows]))
    target_success = float(np.mean([row["success"] for row in target]))
    hardest_success = float(np.mean([row["success"] for row in hardest]))
    hazard_recall = float(np.mean([row["hazard_recall"] for row in rows]))
    coverage = float(np.mean([row["coverage"] for row in rows]))
    collision_rate = sum(row["collisions"] for row in rows) / max(
        1, sum(row["steps"] for row in rows)
    )
    invalid_actions = sum(row["invalid_actions"] for row in rows)
    gate = (
        overall_success >= 0.60
        and target_success >= 0.80
        and hardest_success >= 0.30
        and hazard_recall >= 0.75
        and coverage >= 0.80
        and collision_rate <= 0.02
        and invalid_actions == 0
    )
    summary = {
        "status": "PASSED" if gate else "FAILED",
        "algorithm": "RiskShield-HRMPPO-v2",
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cuda_inference_verified": device.type == "cuda",
        "episodes": len(rows),
        "seeds": args.seeds,
        "overall_mission_success": overall_success,
        "target_mission_success": target_success,
        "hardest_condition_success": hardest_success,
        "hazard_recall": hazard_recall,
        "inspection_coverage": coverage,
        "collision_rate": collision_rate,
        "invalid_actions": invalid_actions,
        "mean_safety_cost": float(np.mean([row["safety_cost"] for row in rows])),
        "constraint_violations": int(sum(row["constraint_violations"] for row in rows)),
        "shield_interventions": int(sum(row["shield_interventions"] for row in rows)),
        "replacement_gate_partial": gate,
        "statistical_v1_comparison_complete": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/hrmppo_v2/evaluation"),
    )
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--shield-horizon", type=int, default=1)
    parser.add_argument("--shield-step-budget", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--require-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    print(f"RISKSHIELD_HRMPPO_V2_EVALUATION={summary['status']}")
    if args.require_gate and summary["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
