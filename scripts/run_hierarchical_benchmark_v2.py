"""Run the real hierarchical v4 policy on the paired grid benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskaware_saferrl.hierarchical import (
    CausalRiskAwarePlanner,
    HierarchicalMissionPolicy,
    MissionOption,
    PredictiveLocalController,
    RiskShieldHierarchicalSystem,
)
from riskaware_saferrl.hierarchical.schemas import causal_option_mask
from riskaware_saferrl.safety import EventAwarePredictiveShieldV3, SafetyContractV3
from scripts.train_hierarchical_hrmppo_mpc_v4 import load_environment, structured_state

CHECKPOINT = Path(
    "artifacts/strong_policy_upgrade/hierarchical_imitation_h4_target_persistence/"
    "seed_105/best_checkpoint.pt"
)
EXPECTED_SHA256 = "1b193f429d8411c30232b60c63260b2df0d112559a6a68139915cc402be14888"
WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = ("low", "medium", "high")
NOISES = (0.0, 0.1, 0.2)
SEEDS = tuple(range(10, 20))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_episode(
    policy: HierarchicalMissionPolicy, world: str, profile: str, seed: int, device: torch.device
) -> dict[str, object]:
    environment = load_environment(world, profile)
    observation, _ = environment.reset(seed=seed)
    contract = SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    system = RiskShieldHierarchicalSystem(
        CausalRiskAwarePlanner(),
        PredictiveLocalController(),
        EventAwarePredictiveShieldV3(contract),
    )
    hidden = policy.initial_state(1, device)
    context = np.zeros((4, 3), dtype=np.float32)
    previous_shield = False
    previous_duration = 1
    progress = 0.0
    steps = 0
    shield_count = 0
    planner_failures = 0
    option_trace: list[int] = []
    try:
        while steps < 500:
            state = structured_state(
                observation, context, progress, previous_shield, previous_duration
            )
            maps = torch.as_tensor(observation["map"], device=device, dtype=torch.float32)[
                None, None
            ]
            states = torch.as_tensor(state, device=device, dtype=torch.float32)[None, None]
            masks = torch.as_tensor(
                causal_option_mask(observation["map"], observation["action_mask"]),
                device=device,
                dtype=torch.bool,
            )[None, None]
            with torch.no_grad():
                output = policy(maps, states, masks, hidden)
            option_tensor = output.option_distribution.probs.argmax(dim=-1)
            option = MissionOption(int(option_tensor.item()))
            target_index = int(output.target_logits.argmax(dim=-1).item())
            target = None if target_index == 256 else divmod(target_index, 16)
            duration = min(4, int(output.duration_logits.argmax(dim=-1).item()) + 1)
            option_trace.append(int(option))
            done = False
            for _ in range(duration):
                decision = system.execute_option(
                    observation,
                    option=option,
                    target=target,
                    risk_budget=float(output.risk_budgets[0, 0].mean()),
                    inspection_intent=option
                    in {MissionOption.INSPECT_KNOWN_RISK, MissionOption.INSPECT_PPE_VIOLATION},
                    force_replan=bool(output.replanning_urgency[0, 0] > 0.5),
                )
                if not decision.planner.success:
                    planner_failures += 1
                shield_count += int(decision.shield.shield_decision != "accept")
                next_observation, reward, terminated, truncated, info = environment.step(
                    decision.executed_primitive
                )
                context[:-1] = context[1:]
                context[-1] = (
                    decision.executed_primitive,
                    float(reward),
                    float(info.get("cost", 0.0)),
                )
                previous_shield = decision.shield.shield_decision != "accept"
                previous_duration = duration
                progress = float(info.get("hazard_recall", progress))
                observation = next_observation
                steps += 1
                done = bool(terminated or truncated)
                if done:
                    break
            hidden = output.recurrent_state.detach()
            if done:
                break
        return {
            "success": bool(info.get("success", False)),
            "hazard_recall": float(info.get("hazard_recall", 0.0)),
            "inspection_coverage": float(
                info.get("coverage", info.get("inspection_coverage", 0.0))
            ),
            "safety_cost": float(info.get("cost", 0.0)),
            "collision_count": int(info.get("collision_count", 0)),
            "invalid_option_count": 0,
            "planner_failures": planner_failures,
            "shield_interventions": shield_count,
            "episode_steps": steps,
            "option_trace_length": len(option_trace),
        }
    finally:
        environment.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("reports/strong_policy_upgrade/benchmark_v2")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=270)
    args = parser.parse_args()
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != EXPECTED_SHA256:
        raise RuntimeError("The verified H4 imitation checkpoint is missing or hash-mismatched.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hierarchical benchmark.")
    initial = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    policy = HierarchicalMissionPolicy().to(device)
    policy.load_state_dict(initial["model"])
    policy.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "hierarchical_v4_rows.csv"
    existing = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            existing = {row["run_id"]: row for row in csv.DictReader(stream)}
    fields = [
        "run_id",
        "algorithm",
        "world",
        "hazard_profile",
        "perception_noise",
        "seed",
        "checkpoint_sha256",
        "success",
        "hazard_recall",
        "inspection_coverage",
        "safety_cost",
        "collision_count",
        "invalid_option_count",
        "planner_failures",
        "shield_interventions",
        "episode_steps",
        "option_trace_length",
    ]
    planned = len(existing)
    for world in WORLDS:
        for profile in PROFILES:
            for noise in NOISES:
                for seed in SEEDS:
                    if planned >= args.limit:
                        break
                    run_id = f"riskshield_hierarchical_hrmppo_mpc_v4_experimental-{world}-{profile}-n{noise:.1f}-s{seed}-v2"
                    if run_id in existing:
                        continue
                    result = run_episode(policy, world, profile, seed, device)
                    existing[run_id] = {
                        "run_id": run_id,
                        "algorithm": "riskshield_hierarchical_hrmppo_mpc_v4_experimental",
                        "world": world,
                        "hazard_profile": profile,
                        "perception_noise": noise,
                        "seed": seed,
                        "checkpoint_sha256": EXPECTED_SHA256,
                        **result,
                    }
                    with csv_path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(existing.values())
                    (args.output / "resume_state.json").write_text(
                        json.dumps({"completed": len(existing), "expected": 270}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    planned += 1
                if planned >= args.limit:
                    break
            if planned >= args.limit:
                break
        if planned >= args.limit:
            break
    if args.limit >= 270 and len(existing) != 270:
        raise RuntimeError(f"Hierarchical benchmark incomplete: {len(existing)}/270")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
