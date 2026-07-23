import argparse

import gymnasium as gym
import numpy as np

import riskaware_saferrl
from riskaware_saferrl.evaluation import EpisodeMetrics
from riskaware_saferrl.safety import SafetyShield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shield", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(riskaware_saferrl.ENV_ID)
    if args.shield:
        env = SafetyShield(env)

    records: list[EpisodeMetrics] = []

    for episode in range(args.episodes):
        _, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        total_cost = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_cost += info["cost"]
            steps += 1

            if terminated or truncated:
                records.append(
                    EpisodeMetrics(
                        reward=total_reward,
                        safety_cost=total_cost,
                        hazard_recall=info["hazard_recall"],
                        coverage=info["coverage"],
                        success=info["success"],
                        steps=steps,
                    )
                )
                break

    print(f"Episodes: {len(records)}")
    print(f"Mean reward: {np.mean([record.reward for record in records]):.3f}")
    print(f"Mean safety cost: {np.mean([record.safety_cost for record in records]):.3f}")
    print(f"Mean hazard recall: {np.mean([record.hazard_recall for record in records]):.3f}")
    print(f"Mean coverage: {np.mean([record.coverage for record in records]):.3f}")
    print(f"Success rate: {np.mean([record.success for record in records]):.3f}")

    env.close()


if __name__ == "__main__":
    main()
