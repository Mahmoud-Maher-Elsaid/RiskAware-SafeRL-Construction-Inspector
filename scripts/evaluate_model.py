import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import riskaware_saferrl
from riskaware_saferrl.safety import SafetyShield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--shield", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model)

    rewards = []
    costs = []
    recalls = []
    success = []

    for episode in range(args.episodes):
        env = gym.make(riskaware_saferrl.ENV_ID)
        if args.shield:
            env = SafetyShield(env)

        observation, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        total_cost = 0.0

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            total_cost += info["cost"]

            if terminated or truncated:
                rewards.append(total_reward)
                costs.append(total_cost)
                recalls.append(info["hazard_recall"])
                success.append(info["success"])
                break

        env.close()

    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Mean safety cost: {np.mean(costs):.3f}")
    print(f"Mean hazard recall: {np.mean(recalls):.3f}")
    print(f"Success rate: {np.mean(success):.3f}")


if __name__ == "__main__":
    main()
