from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from riskaware_saferrl.algorithms.hierarchical_v4 import (
    RiskShieldHierarchicalHRMPPOV4,
)
from riskaware_saferrl.buffers import RecurrentVectorCostBuffer
from riskaware_saferrl.envs import GridEnvironmentConfig, ResearchConstructionEnvV2
from riskaware_saferrl.hierarchical import (
    CausalRiskAwarePlanner,
    HierarchicalMissionPolicy,
    MissionOption,
    PredictiveLocalController,
    RiskShieldHierarchicalSystem,
)
from riskaware_saferrl.hierarchical.schemas import VECTOR_COST_NAMES, causal_option_mask
from riskaware_saferrl.safety import EventAwarePredictiveShieldV3, SafetyContractV3
from riskaware_saferrl.training.hierarchical_demonstrations import _vector_cost

WORLDS = ("site_small", "site_medium", "site_dynamic")
PROFILES = {
    "low": (0.75, 0.75, 0.0),
    "medium": (1.0, 1.0, 0.1),
    "high": (1.5, 1.5, 0.2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train option-level constrained HRMPPO v4.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/riskshield_hierarchical_hrmppo_mpc_v4.yaml"),
    )
    parser.add_argument(
        "--initialize",
        type=Path,
        default=Path(
            "artifacts/strong_policy_upgrade/hierarchical_imitation_systematic_final/"
            "seed_102/best_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hierarchical_hrmppo_v4"),
    )
    parser.add_argument("--option-steps", type=int, default=5_000)
    parser.add_argument("--rollout-options", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-memory-gb", type=float)
    return parser.parse_args()


def load_environment(world: str, profile: str) -> ResearchConstructionEnvV2:
    multiplier, worker_multiplier, noise = PROFILES[profile]
    payload = yaml.safe_load(Path(f"configs/grid/{world}.yaml").read_text("utf-8"))
    allowed = {field.name for field in fields(GridEnvironmentConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    payload["hazard_density"] = min(0.5, payload["hazard_density"] * multiplier)
    payload["worker_density"] = min(0.5, payload["worker_density"] * worker_multiplier)
    payload["perception_false_negative_rate"] = noise
    return ResearchConstructionEnvV2(GridEnvironmentConfig(**payload))


def structured_state(
    observation: dict[str, np.ndarray],
    context: np.ndarray,
    progress: float,
    shield_intervened: bool,
    duration: int,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(observation["state"], dtype=np.float32),
            context.reshape(-1),
            np.asarray(
                (progress, float(shield_intervened), duration / 8.0),
                dtype=np.float32,
            ),
        )
    )


def save_checkpoint(
    path: Path,
    algorithm: RiskShieldHierarchicalHRMPPOV4,
    *,
    option_steps: int,
    episode_index: int,
    history: list[dict[str, Any]],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = algorithm.state_dict()
    torch.save(
        {
            **state,
            "model": algorithm.policy.state_dict(),
            "option_steps": option_steps,
            "episode_index": episode_index,
            "history": history,
            "seed": seed,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "recurrent_state_schema": [1, 1, algorithm.policy.recurrent_hidden_size],
            "architecture": "RiskShield-Hierarchical-HRMPPO-MPC-v4",
        },
        path,
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text("utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    rollout_options = int(args.rollout_options or config["rollout_options"])
    memory_limit = float(args.max_memory_gb or config["maximum_memory_gb"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    initial = torch.load(args.initialize, map_location=device, weights_only=False)
    policy = HierarchicalMissionPolicy().to(device)
    policy.load_state_dict(initial["model"])
    algorithm = RiskShieldHierarchicalHRMPPOV4(
        policy,
        budgets=config["budgets"],
        cost_scales=config["cost_scales"],
        learning_rate=float(config["learning_rate"]),
        cost_critic_learning_rate=float(config["cost_critic_learning_rate"]),
        clip_range=float(config["clip_range"]),
        maximum_kl=float(config["maximum_kl"]),
        anchor_kl_coefficient=float(config["anchor_kl_coefficient"]),
        entropy_coefficient=float(config["entropy_coefficient"]),
        multiplier_warmup_updates=int(config["multiplier_warmup_updates"]),
    )
    checkpoint = args.output / "resume_checkpoint.pt"
    option_steps = episode_index = 0
    history: list[dict[str, Any]] = []
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        algorithm.load_state_dict(saved)
        option_steps = int(saved["option_steps"])
        episode_index = int(saved["episode_index"])
        history = list(saved["history"])
        random.setstate(saved["python_random_state"])
        np.random.set_state(saved["numpy_random_state"])
        torch.random.set_rng_state(saved["torch_random_state"])
        if device.type == "cuda" and saved["cuda_random_state"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_random_state"])
    process = psutil.Process()
    contract = SafetyContractV3.from_yaml(Path("configs/safety/safety_contract_v3.yaml"))
    context = np.zeros((4, 3), dtype=np.float32)
    hidden = policy.initial_state(1, device)
    observation: dict[str, np.ndarray] | None = None
    environment: ResearchConstructionEnvV2 | None = None
    system: RiskShieldHierarchicalSystem | None = None
    progress = 0.0
    previous_shield = False
    previous_duration = 1
    episode_start = True
    episode_cost_totals = np.zeros(len(VECTOR_COST_NAMES), dtype=np.float64)
    current_episode_costs = np.zeros(len(VECTOR_COST_NAMES), dtype=np.float64)
    completed_episodes = 0
    completed_primitive_steps = 0
    current_episode_primitive_steps = 0
    while option_steps < args.option_steps:
        buffer = RecurrentVectorCostBuffer(
            cost_count=len(VECTOR_COST_NAMES),
            reward_gamma=float(config["reward_gamma"]),
            reward_gae_lambda=float(config["reward_gae_lambda"]),
            cost_gamma=float(config["cost_gamma"]),
            cost_gae_lambda=float(config["cost_gae_lambda"]),
        )
        for _ in range(min(rollout_options, args.option_steps - option_steps)):
            if observation is None:
                world = WORLDS[episode_index % len(WORLDS)]
                profile = tuple(PROFILES)[(episode_index // len(WORLDS)) % len(PROFILES)]
                environment = load_environment(world, profile)
                observation, _ = environment.reset(seed=seed + episode_index)
                system = RiskShieldHierarchicalSystem(
                    CausalRiskAwarePlanner(),
                    PredictiveLocalController(),
                    EventAwarePredictiveShieldV3(contract),
                )
                context.fill(0)
                hidden = policy.initial_state(1, device)
                progress = 0.0
                previous_shield = False
                previous_duration = 1
                episode_start = True
            assert environment is not None and system is not None
            state = structured_state(
                observation,
                context,
                progress,
                previous_shield,
                previous_duration,
            )
            maps = torch.as_tensor(observation["map"], device=device, dtype=torch.float32)[
                None, None
            ]
            states = torch.as_tensor(state, device=device)[None, None]
            option_masks = torch.as_tensor(
                causal_option_mask(observation["map"], observation["action_mask"]),
                device=device,
                dtype=torch.bool,
            )[None, None]
            with torch.no_grad():
                output = policy(maps, states, option_masks, hidden)
                option_tensor = output.option_distribution.sample()
                log_probability = output.option_distribution.log_prob(option_tensor)[0, 0]
            option = MissionOption(int(option_tensor.item()))
            target_index = int(output.target_logits.argmax(dim=-1).item())
            target = None if target_index == 256 else divmod(target_index, 16)
            duration = min(
                int(config["option_decision_interval"]),
                int(output.duration_logits.argmax(dim=-1).item()) + 1,
            )
            reward_total = 0.0
            vector_total = np.zeros(len(VECTOR_COST_NAMES), dtype=np.float32)
            done = False
            for _primitive_step in range(duration):
                decision = system.execute_option(
                    observation,
                    option=option,
                    target=target,
                    risk_budget=float(output.risk_budgets[0, 0].mean()),
                    inspection_intent=option
                    in {
                        MissionOption.INSPECT_KNOWN_RISK,
                        MissionOption.INSPECT_PPE_VIOLATION,
                    },
                    force_replan=bool(output.replanning_urgency[0, 0] > 0.5),
                )
                next_observation, reward, terminated, truncated, info = environment.step(
                    decision.executed_primitive
                )
                current_vector = _vector_cost(
                    np.asarray(observation["map"]),
                    float(info["cost"]),
                    np.asarray(observation["state"]),
                )
                current_vector[5] = float(decision.shield.shield_decision != "accept")
                current_vector[6] = float(
                    decision.executed_primitive == 4 and option == MissionOption.EXPLORE_FRONTIER
                )
                reward_total += float(reward)
                vector_total += current_vector
                current_episode_primitive_steps += 1
                context[:-1] = context[1:]
                context[-1] = (
                    decision.executed_primitive,
                    float(reward),
                    float(info["cost"]),
                )
                progress = float(info["hazard_recall"])
                previous_shield = decision.shield.shield_decision != "accept"
                observation = next_observation
                done = bool(terminated or truncated)
                if done:
                    completed_episodes += 1
                    episode_index += 1
                    break
            buffer.add(
                map=maps[0, 0],
                state=states[0, 0],
                recurrent_state=hidden.detach(),
                episode_start=torch.tensor(episode_start, device=device),
                action=option_tensor[0, 0].detach(),
                action_mask=option_masks[0, 0],
                reward=torch.tensor(reward_total, device=device),
                vector_cost=torch.as_tensor(vector_total, device=device),
                reward_value=output.reward_value[0, 0].detach(),
                vector_cost_value=output.vector_cost_values[0, 0].detach(),
                log_probability=log_probability.detach(),
                done=torch.tensor(done, device=device),
            )
            current_episode_costs += vector_total
            hidden = output.recurrent_state.detach()
            previous_duration = duration
            episode_start = done
            option_steps += 1
            if done:
                episode_cost_totals += current_episode_costs
                current_episode_costs.fill(0)
                completed_primitive_steps += current_episode_primitive_steps
                current_episode_primitive_steps = 0
                observation = None
            rss_gb = process.memory_info().rss / (1024**3)
            if rss_gb > memory_limit:
                save_checkpoint(
                    checkpoint,
                    algorithm,
                    option_steps=option_steps,
                    episode_index=episode_index,
                    history=history,
                    seed=seed,
                )
                raise MemoryError(f"RSS memory guard exceeded at {rss_gb:.2f} GiB")
        last_reward = torch.zeros((), device=device)
        last_cost = torch.zeros(len(VECTOR_COST_NAMES), device=device)
        if observation is not None:
            with torch.no_grad():
                state = structured_state(
                    observation,
                    context,
                    progress,
                    previous_shield,
                    previous_duration,
                )
                bootstrap = policy(
                    torch.as_tensor(observation["map"], device=device)[None, None],
                    torch.as_tensor(state, device=device)[None, None],
                    torch.as_tensor(
                        causal_option_mask(observation["map"], observation["action_mask"]),
                        device=device,
                    )[None, None],
                    hidden,
                )
                last_reward = bootstrap.reward_value[0, 0]
                last_cost = bootstrap.vector_cost_values[0, 0]
        batch = buffer.finalize(last_reward, last_cost)
        observed_costs = {}
        for index, name in enumerate(VECTOR_COST_NAMES):
            if index < 3:
                denominator = max(1, completed_episodes)
                scale = 1.0
            else:
                denominator = max(1, completed_primitive_steps)
                scale = 100.0
            observed_costs[name] = float(
                episode_cost_totals[index] * scale / denominator
            )
        started = time.perf_counter()
        update = algorithm.update(
            batch,
            observed_episode_costs=observed_costs,
            epochs=int(config["ppo_epochs"]),
        )
        telemetry = {
            "option_steps": option_steps,
            "episodes": completed_episodes,
            "seconds": time.perf_counter() - started,
            "rss_gb": process.memory_info().rss / (1024**3),
            "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3)
            if device.type == "cuda"
            else 0.0,
            **update.__dict__,
        }
        history.append(telemetry)
        print(json.dumps(telemetry))
        save_checkpoint(
            checkpoint,
            algorithm,
            option_steps=option_steps,
            episode_index=episode_index,
            history=history,
            seed=seed,
        )
    final = args.output / "final_checkpoint.pt"
    save_checkpoint(
        final,
        algorithm,
        option_steps=option_steps,
        episode_index=episode_index,
        history=history,
        seed=seed,
    )
    (args.output / "training_summary.json").write_text(
        json.dumps(
            {
                "status": "PASSED",
                "architecture": "RiskShield-Hierarchical-HRMPPO-MPC-v4",
                "option_steps": option_steps,
                "episodes": completed_episodes,
                "cuda_verified": device.type == "cuda",
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("HIERARCHICAL_HRMPPO_V4_TRAINING=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
