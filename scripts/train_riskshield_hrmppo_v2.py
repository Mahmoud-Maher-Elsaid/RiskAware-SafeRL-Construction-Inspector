from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from riskaware_saferrl.algorithms import RiskShieldHRMPPOV2
from riskaware_saferrl.buffers import RecurrentCostRolloutBuffer
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.policies import RecurrentMaskedPolicy
from riskaware_saferrl.safety import PredictiveSafetyShield

WORLDS = ("site_small", "site_medium", "site_dynamic")
CURRICULUM = (
    ("easy", ("site_small",), ("low",)),
    ("medium", ("site_small", "site_medium"), ("low", "medium")),
    ("full", WORLDS, ("low", "medium", "high")),
)
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


def curriculum_condition(step: int, total_steps: int, rng: random.Random) -> tuple[str, str, str]:
    fraction = step / max(1, total_steps)
    index = 0 if fraction < 0.25 else 1 if fraction < 0.60 else 2
    stage, worlds, profiles = CURRICULUM[index]
    return stage, rng.choice(worlds), rng.choice(profiles)


def make_environment(
    step: int, total_steps: int, rng: random.Random
) -> tuple[ResearchConstructionEnvV2, str, str, str]:
    stage, world, profile = curriculum_condition(step, total_steps, rng)
    return ResearchConstructionEnvV2(load_config(world, profile)), stage, world, profile


def tensor_observation(
    observation: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(observation["map"], device=device).unsqueeze(0).unsqueeze(0),
        torch.as_tensor(observation["state"], device=device).unsqueeze(0).unsqueeze(0),
        torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
        .unsqueeze(0)
        .unsqueeze(0),
    )


def save_checkpoint(
    path: Path,
    *,
    algorithm: RiskShieldHRMPPOV2,
    step: int,
    update: int,
    seed: int,
    curriculum_stage: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch.save(
        {
            **algorithm.state_dict(),
            "step": step,
            "update": update,
            "seed": seed,
            "curriculum_stage": curriculum_stage,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": {
                "name": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": numpy_state[2],
                "has_gaussian": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "python_rng_state": {
                "version": python_state[0],
                "state": torch.tensor(python_state[1], dtype=torch.int64),
                "gaussian": python_state[2],
            },
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    policy = RecurrentMaskedPolicy().to(device)
    initial_hash = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        policy.load_state_dict(initial["model_state_dict"])
        initial_hash = sha256_file(args.initial_checkpoint)
    algorithm = RiskShieldHRMPPOV2(
        policy,
        learning_rate=args.learning_rate,
        safety_budget=args.safety_budget,
        value_coefficient=args.value_coefficient,
        cost_value_coefficient=args.cost_value_coefficient,
        anchor_kl_coefficient=args.anchor_kl_coefficient,
    )
    step = update_index = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        algorithm.load_state_dict(checkpoint)
        step = int(checkpoint["step"])
        update_index = int(checkpoint["update"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        numpy_state = checkpoint["numpy_rng_state"]
        np.random.set_state(
            (
                numpy_state["name"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gaussian"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        python_state = checkpoint["python_rng_state"]
        random.setstate(
            (
                int(python_state["version"]),
                tuple(int(value) for value in python_state["state"].cpu().tolist()),
                python_state["gaussian"],
            )
        )
    output_dir = args.output_dir
    report_dir = args.report_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    metrics: list[dict[str, Any]] = []
    completed_episode_costs: list[float] = []
    episode_records: list[dict[str, Any]] = []
    environment, stage, world, profile = make_environment(step, args.total_steps, rng)
    episode_seed = args.seed * 100_000 + step
    observation, _ = environment.reset(seed=episode_seed)
    hidden = policy.initial_state(1, device)
    episode_start = True
    episode_reward = 0.0
    episode_shield_interventions = 0
    shield = PredictiveSafetyShield(
        horizon=args.shield_horizon,
        safety_budget=args.shield_step_budget,
    )
    started = time.perf_counter()
    while step < args.total_steps:
        buffer = RecurrentCostRolloutBuffer(gamma=args.gamma, gae_lambda=args.gae_lambda)
        rollout_steps = min(args.rollout_steps, args.total_steps - step)
        for _ in range(rollout_steps):
            maps, states, masks = tensor_observation(observation, device)
            with torch.no_grad():
                output = policy(maps, states, masks, hidden)
                proposed = output.distribution.sample()
                log_probability = output.distribution.log_prob(proposed)
            proposed_action = int(proposed.item())
            if not bool(observation["action_mask"][proposed_action]):
                raise RuntimeError("Masked policy proposed an invalid action")
            decision = shield.decide(environment, proposed_action)
            executed_action = int(decision.final_action)
            next_observation, reward, terminated, truncated, info = environment.step(
                executed_action
            )
            done = terminated or truncated
            cost = float(info.get("cost", 0.0))
            buffer.add(
                map=maps[0, 0].detach(),
                state=states[0, 0].detach(),
                recurrent_state=hidden.detach(),
                episode_start=torch.tensor(episode_start, device=device),
                action=proposed[0, 0].detach(),
                action_mask=masks[0, 0].detach(),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                cost=torch.tensor(cost, dtype=torch.float32, device=device),
                reward_value=output.reward_value[0, 0].detach(),
                cost_value=output.cost_value[0, 0].detach(),
                log_probability=log_probability[0, 0].detach(),
                done=torch.tensor(done, device=device),
            )
            hidden = output.recurrent_state.detach()
            observation = next_observation
            episode_start = False
            episode_reward += reward
            episode_shield_interventions += int(decision.shield_decision != "accept")
            step += 1
            if done:
                completed_episode_costs.append(float(environment.cumulative_cost))
                episode_records.append(
                    {
                        "step": step,
                        "seed": episode_seed,
                        "curriculum_stage": stage,
                        "world": world,
                        "profile": profile,
                        "success": bool(info["success"]),
                        "hazard_recall": float(info["hazard_recall"]),
                        "coverage": float(info["inspection_coverage"]),
                        "reward": episode_reward,
                        "safety_cost": float(environment.cumulative_cost),
                        "collisions": int(environment.collisions),
                        "constraint_violations": int(
                            environment.collisions
                            + environment.near_misses
                            + environment.restricted_violations
                        ),
                        "shield_interventions": episode_shield_interventions,
                    }
                )
                environment, stage, world, profile = make_environment(step, args.total_steps, rng)
                episode_seed = args.seed * 100_000 + step
                observation, _ = environment.reset(seed=episode_seed)
                hidden = policy.initial_state(1, device)
                episode_start = True
                episode_reward = 0.0
                episode_shield_interventions = 0
            if step >= args.total_steps:
                break
        maps, states, masks = tensor_observation(observation, device)
        with torch.no_grad():
            bootstrap = policy(maps, states, masks, hidden)
        batch = buffer.finalize(
            bootstrap.reward_value[0, 0].detach(),
            bootstrap.cost_value[0, 0].detach(),
        )
        recent_cost = (
            float(np.mean(completed_episode_costs[-10:]))
            if completed_episode_costs
            else float(batch.costs.sum().detach().cpu())
        )
        update = algorithm.update(
            batch,
            epochs=args.ppo_epochs,
            observed_episode_cost=recent_cost,
        )
        update_index += 1
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        row = {
            "step": step,
            "update": update_index,
            "curriculum_stage": stage,
            **asdict(update),
            "recent_episode_cost": recent_cost,
            "process_rss_bytes": process.memory_info().rss,
            "available_physical_bytes": vm.available,
            "committed_memory_bytes": vm.total - vm.available + swap.used,
            "gpu_allocated_bytes": (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            ),
            "gpu_reserved_bytes": (
                torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
            ),
            "steps_per_second": step / max(1e-9, time.perf_counter() - started),
        }
        metrics.append(row)
        if (
            row["process_rss_bytes"] > args.max_memory_gb * 1024**3
            or row["available_physical_bytes"] < args.minimum_available_gb * 1024**3
        ):
            emergency = output_dir / "memory_guard_checkpoint.pt"
            checkpoint_hash = save_checkpoint(
                emergency,
                algorithm=algorithm,
                step=step,
                update=update_index,
                seed=args.seed,
                curriculum_stage=stage,
            )
            failure = {
                "status": "FAILED",
                "reason": "memory_guard",
                "checkpoint": emergency.as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "last_memory": row,
            }
            (report_dir / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            raise MemoryError("HRMPPO memory guard stopped training cleanly")
        if step % args.checkpoint_interval < rollout_steps or step >= args.total_steps:
            checkpoint_path = output_dir / f"checkpoint_{step:09d}.pt"
            save_checkpoint(
                checkpoint_path,
                algorithm=algorithm,
                step=step,
                update=update_index,
                seed=args.seed,
                curriculum_stage=stage,
            )
    final_checkpoint = output_dir / "final_checkpoint.pt"
    checkpoint_hash = save_checkpoint(
        final_checkpoint,
        algorithm=algorithm,
        step=step,
        update=update_index,
        seed=args.seed,
        curriculum_stage=stage,
    )
    summary = {
        "status": "PASSED",
        "algorithm": "RiskShield-HRMPPO-v2",
        "seed": args.seed,
        "device": str(device),
        "cuda_verified": device.type == "cuda",
        "total_environment_steps": step,
        "updates": update_index,
        "initial_checkpoint": args.initial_checkpoint.as_posix()
        if args.initial_checkpoint
        else None,
        "initial_checkpoint_sha256": initial_hash,
        "final_checkpoint": final_checkpoint.as_posix(),
        "final_checkpoint_sha256": checkpoint_hash,
        "safety_budget": args.safety_budget,
        "pid_lagrange": algorithm.pid_lagrange.state_dict(),
        "predictive_shield": {
            "active": True,
            "horizon": args.shield_horizon,
            "step_budget": args.shield_step_budget,
        },
        "curriculum": [stage[0] for stage in CURRICULUM],
        "episodes": len(episode_records),
        "episode_metrics": episode_records,
        "update_metrics": metrics,
    }
    (report_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hrmppo_v2"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/hrmppo_v2/training"),
    )
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--value-coefficient", type=float, default=0.1)
    parser.add_argument("--cost-value-coefficient", type=float, default=0.1)
    parser.add_argument("--anchor-kl-coefficient", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--safety-budget", type=float, default=20.0)
    parser.add_argument("--shield-horizon", type=int, default=1)
    parser.add_argument("--shield-step-budget", type=float, default=1.0)
    parser.add_argument("--max-memory-gb", type=float, default=8.0)
    parser.add_argument("--minimum-available-gb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train(args)
    print(
        f"RISKSHIELD_HRMPPO_V2_TRAINING={summary['status']} "
        f"(steps={summary['total_environment_steps']})"
    )


if __name__ == "__main__":
    main()
