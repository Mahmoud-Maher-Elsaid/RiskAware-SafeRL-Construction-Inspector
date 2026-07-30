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

from riskaware_saferrl.algorithms import (
    RecurrentMaskedSafePolicyV3,
    RiskShieldHRMPPOSafeV3,
)
from riskaware_saferrl.buffers import RecurrentVectorCostBuffer
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.safety import (
    ControlledInspectionState,
    EventAwarePredictiveShieldV3,
    SafetyContractV3,
    SafetyContractV3Wrapper,
    SafetyStepContext,
)
from riskaware_saferrl.safety.safety_contract_v3 import VECTOR_COST_NAMES

WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = {
    "low": (0.75, 0.75, 0.0),
    "medium": (1.0, 1.0, 0.1),
    "high": (1.5, 1.5, 0.2),
}
CURRICULUM = (
    ("easy", ("site_small",), ("low",)),
    ("medium", ("site_small", "site_medium"), ("low", "medium")),
    ("full", WORLDS, tuple(PROFILES)),
)


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


def make_environment(
    *,
    step: int,
    total_steps: int,
    rng: random.Random,
    safety_contract_path: Path,
    shield_config: dict[str, Any],
) -> tuple[SafetyContractV3Wrapper, EventAwarePredictiveShieldV3, str, str, str]:
    fraction = step / max(1, total_steps)
    stage_index = 0 if fraction < 0.25 else 1 if fraction < 0.60 else 2
    stage, worlds, profiles = CURRICULUM[stage_index]
    world = rng.choice(worlds)
    profile = rng.choice(profiles)
    contract = SafetyContractV3.from_yaml(safety_contract_path)
    environment = SafetyContractV3Wrapper(
        ResearchConstructionEnvV2(load_config(world, profile)),
        contract,
    )
    shield = EventAwarePredictiveShieldV3(
        contract,
        minimum_horizon=int(shield_config["minimum_horizon"]),
        maximum_horizon=int(shield_config["maximum_horizon"]),
    )
    return environment, shield, stage, world, profile


def inspection_state(
    observation: dict[str, np.ndarray],
    *,
    proposed_action: int,
    shield_active: bool,
    dwell_steps: int,
) -> ControlledInspectionState:
    robot = np.argwhere(observation["map"][5] > 0)[0]
    hazards = np.argwhere(observation["map"][1] > 0)
    target_recognized = any(
        abs(int(robot[0]) - int(target[0])) + abs(int(robot[1]) - int(target[1])) <= 2
        for target in hazards
    )
    workers = np.argwhere(observation["map"][2] > 0)
    clearance = (
        min(
            abs(int(robot[0]) - int(worker[0])) + abs(int(robot[1]) - int(worker[1]))
            for worker in workers
        )
        if len(workers)
        else float("inf")
    )
    action_mask = np.asarray(observation["action_mask"], dtype=np.bool_)
    retreat = bool(action_mask[:4].any())
    return ControlledInspectionState(
        target_recognized=target_recognized,
        inspection_intent=proposed_action == 4,
        safe_approach=clearance >= 2,
        speed=0.0 if proposed_action == 4 else 1.0,
        human_clearance_cells=float(clearance),
        dwell_steps=dwell_steps,
        retreat_route_available=retreat,
        shield_active=shield_active,
    )


def tensors(
    observation: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(observation["map"], device=device).unsqueeze(0).unsqueeze(0),
        torch.as_tensor(observation["state"], device=device).unsqueeze(0).unsqueeze(0),
        torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
        .unsqueeze(0)
        .unsqueeze(0),
    )


def safe_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return {
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
    }


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
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


def save_checkpoint(
    path: Path,
    *,
    algorithm: RiskShieldHRMPPOSafeV3,
    step: int,
    update: int,
    seed: int,
    evaluation_history: list[dict[str, Any]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            **algorithm.state_dict(),
            **safe_rng_state(),
            "step": step,
            "update": update,
            "seed": seed,
            "observation_schema": {
                "map": [11, 16, 16],
                "state": [22],
                "action_mask": [5],
            },
            "vector_cost_schema": list(VECTOR_COST_NAMES),
            "recurrent_state_schema": [1, 1, 256],
            "curriculum_state": {
                "fraction": step,
                "schedule": [stage[0] for stage in CURRICULUM],
            },
            "evaluation_history": evaluation_history,
        },
        temporary,
    )
    temporary.replace(path)
    return sha256_file(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    baseline = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
    policy = RecurrentMaskedSafePolicyV3.from_v2_checkpoint(baseline, device)
    algorithm = RiskShieldHRMPPOSafeV3(
        policy,
        budgets=config["budgets"],
        cost_scales=config["cost_scales"],
        learning_rate=float(config["learning_rate"]),
        cost_critic_learning_rate=float(config["cost_critic_learning_rate"]),
        clip_range=float(config["clip_range"]),
        entropy_coefficient=float(config["entropy_coefficient"]),
        anchor_kl_coefficient=float(config["anchor_kl_coefficient"]),
        maximum_kl=float(config["maximum_kl"]),
        multiplier_warmup_updates=int(config["multiplier_warmup_updates"]),
    )
    step = update_index = 0
    evaluation_history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        algorithm.load_state_dict(checkpoint)
        step = int(checkpoint["step"])
        update_index = int(checkpoint["update"])
        evaluation_history = list(checkpoint["evaluation_history"])
        restore_rng_state(checkpoint)
    environment, shield, stage, world, profile = make_environment(
        step=step,
        total_steps=args.total_steps,
        rng=rng,
        safety_contract_path=args.safety_contract,
        shield_config=config["shield"],
    )
    observation, _ = environment.reset(seed=args.seed * 100_000 + step)
    hidden = policy.initial_state(1, device)
    episode_start = True
    inspection_dwell = 0
    completed_vector_costs: list[dict[str, float]] = []
    episode_vector = {name: 0.0 for name in VECTOR_COST_NAMES}
    episode_steps = 0
    metrics = []
    process = psutil.Process()
    started = time.perf_counter()
    while step < args.total_steps:
        buffer = RecurrentVectorCostBuffer(
            reward_gamma=float(config["reward_gamma"]),
            reward_gae_lambda=float(config["reward_gae_lambda"]),
            cost_gamma=float(config["cost_gamma"]),
            cost_gae_lambda=float(config["cost_gae_lambda"]),
        )
        rollout_steps = min(int(config["rollout_steps"]), args.total_steps - step)
        for _ in range(rollout_steps):
            maps, states, masks = tensors(observation, device)
            with torch.no_grad():
                output = policy(maps, states, masks, hidden)
                proposed = output.distribution.sample()
            proposed_action = int(proposed.item())
            inspection_dwell = inspection_dwell + 1 if proposed_action == 4 else 0
            inspection = inspection_state(
                observation,
                proposed_action=proposed_action,
                shield_active=True,
                dwell_steps=inspection_dwell,
            )
            decision = shield.decide(observation, proposed_action, inspection)
            executed = torch.tensor([[decision.final_action]], dtype=torch.long, device=device)
            with torch.no_grad():
                log_probability = output.distribution.log_prob(executed)
            environment.prepare_step(
                SafetyStepContext(
                    proposed_action=proposed_action,
                    predicted_trajectory=decision.predicted_trajectory,
                    predicted_vector_cost=decision.predicted_vector_cost,
                    shield_result=decision.to_dict(),
                    inspection=inspection,
                )
            )
            next_observation, reward, terminated, truncated, info = environment.step(
                decision.final_action
            )
            vector = torch.tensor(
                list(info["safety_vector_v3"].values()),
                dtype=torch.float32,
                device=device,
            )
            done = terminated or truncated
            buffer.add(
                map=maps[0, 0].detach(),
                state=states[0, 0].detach(),
                recurrent_state=hidden.detach(),
                episode_start=torch.tensor(episode_start, device=device),
                # The environment reward and vector costs result from the
                # shield-executed action. Crediting the rejected proposal is
                # an off-transition assignment that prevents the actor from
                # learning the safe replacement.
                action=executed[0, 0].detach(),
                action_mask=masks[0, 0].detach(),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                vector_cost=vector,
                reward_value=output.reward_value[0, 0].detach(),
                vector_cost_value=output.vector_cost_values[0, 0].detach(),
                log_probability=log_probability[0, 0].detach(),
                done=torch.tensor(done, device=device),
            )
            for index, name in enumerate(VECTOR_COST_NAMES):
                episode_vector[name] += float(vector[index].detach().cpu())
            episode_steps += 1
            observation = next_observation
            hidden = output.recurrent_state.detach()
            episode_start = False
            step += 1
            if done:
                completed_vector_costs.append({**episode_vector, "_steps": float(episode_steps)})
                episode_vector = {name: 0.0 for name in VECTOR_COST_NAMES}
                episode_steps = 0
                environment, shield, stage, world, profile = make_environment(
                    step=step,
                    total_steps=args.total_steps,
                    rng=rng,
                    safety_contract_path=args.safety_contract,
                    shield_config=config["shield"],
                )
                observation, _ = environment.reset(seed=args.seed * 100_000 + step)
                hidden = policy.initial_state(1, device)
                episode_start = True
                inspection_dwell = 0
            if step >= args.total_steps:
                break
        maps, states, masks = tensors(observation, device)
        with torch.no_grad():
            bootstrap = policy(maps, states, masks, hidden)
        batch = buffer.finalize(
            bootstrap.reward_value[0, 0].detach(),
            bootstrap.vector_cost_values[0, 0].detach(),
        )
        observed = {
            name: (
                float(
                    np.mean(
                        [
                            (
                                100.0 * row[name] / max(1.0, row["_steps"])
                                if name
                                in {
                                    "uncontrolled_semantic_risk",
                                    "uncertainty",
                                }
                                else row[name]
                            )
                            for row in completed_vector_costs[-10:]
                        ]
                    )
                )
                if completed_vector_costs
                else float(
                    batch.vector_costs[:, index].sum().detach().cpu()
                    * (
                        100.0 / max(1, len(buffer.records))
                        if name
                        in {
                            "uncontrolled_semantic_risk",
                            "uncertainty",
                        }
                        else 1.0
                    )
                )
            )
            for index, name in enumerate(VECTOR_COST_NAMES)
        }
        update = algorithm.update(
            batch,
            observed_episode_costs=observed,
            epochs=int(config["ppo_epochs"]),
        )
        update_index += 1
        memory = process.memory_info().rss
        row = {
            "step": step,
            "update": update_index,
            "curriculum_stage": stage,
            "world": world,
            "profile": profile,
            **asdict(update),
            "observed_episode_costs": observed,
            "process_rss_bytes": memory,
            "gpu_allocated_bytes": (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            ),
            "steps_per_second": step / max(1e-9, time.perf_counter() - started),
        }
        metrics.append(row)
        if memory > args.max_memory_gb * 1024**3:
            emergency = args.output_dir / "memory_guard_checkpoint.pt"
            save_checkpoint(
                emergency,
                algorithm=algorithm,
                step=step,
                update=update_index,
                seed=args.seed,
                evaluation_history=evaluation_history,
            )
            raise MemoryError("Safe v3 memory guard stopped training")
        if step % args.checkpoint_interval < rollout_steps or step >= args.total_steps:
            save_checkpoint(
                args.output_dir / f"checkpoint_{step:09d}.pt",
                algorithm=algorithm,
                step=step,
                update=update_index,
                seed=args.seed,
                evaluation_history=evaluation_history,
            )
    final = args.output_dir / "final_checkpoint.pt"
    checkpoint_hash = save_checkpoint(
        final,
        algorithm=algorithm,
        step=step,
        update=update_index,
        seed=args.seed,
        evaluation_history=evaluation_history,
    )
    summary = {
        "status": "PASSED",
        "algorithm": "RiskShield-HRMPPO-Safe-v3",
        "strategy": args.strategy,
        "steps": step,
        "updates": update_index,
        "seed": args.seed,
        "cuda_verified": device.type == "cuda",
        "initial_checkpoint": args.initial_checkpoint.as_posix(),
        "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
        "final_checkpoint": final.as_posix(),
        "final_checkpoint_sha256": checkpoint_hash,
        "vector_cost_schema": list(VECTOR_COST_NAMES),
        "update_metrics": metrics,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("pid_lagrange", "safety_dagger_pid", "shield_aware_kl"),
        default="pid_lagrange",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/riskshield_hrmppo_safe_v3.yaml"),
    )
    parser.add_argument(
        "--safety-contract",
        type=Path,
        default=Path("configs/safety/safety_contract_v3.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    parser.add_argument("--max-memory-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    summary = train(parse_args())
    print(f"RISKSHIELD_HRMPPO_SAFE_V3_TRAINING={summary['status']} (steps={summary['steps']})")


if __name__ == "__main__":
    main()
