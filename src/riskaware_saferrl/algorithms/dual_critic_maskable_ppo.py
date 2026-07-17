from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import (
    get_action_masks,
    is_masking_supported,
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import (
    GymEnv,
    Schedule,
)
from stable_baselines3.common.utils import (
    explained_variance,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as functional

from riskaware_saferrl.algorithms.cost_value import (
    CostValueNetwork,
)
from riskaware_saferrl.algorithms.dual_controller import (
    DualVariableController,
)
from riskaware_saferrl.algorithms.dual_critic_buffer import (
    DualCriticMaskableDictRolloutBuffer,
)


class DualCriticMaskablePPO(MaskablePPO):
    """Maskable PPO with independent reward and cost critics."""

    def __init__(
        self,
        policy: str,
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int | None = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: float | Schedule | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = 0.03,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        *,
        cost_limit: float = 5.0,
        cost_gamma: float = 1.0,
        cost_gae_lambda: float = 0.95,
        cost_learning_rate: float = 1e-4,
        cost_features_dim: int = 128,
        initial_lagrange_multiplier: float = 0.0,
        dual_learning_rate: float = 0.01,
        dual_maximum: float = 2.0,
        dual_ema_beta: float = 0.9,
        dual_warmup_updates: int = 5,
        _init_setup_model: bool = True,
    ) -> None:
        if cost_learning_rate <= 0.0:
            raise ValueError("cost_learning_rate must be positive.")
        if cost_features_dim < 1:
            raise ValueError("cost_features_dim must be positive.")

        self.cost_limit = float(cost_limit)
        self.cost_gamma = float(cost_gamma)
        self.cost_gae_lambda = float(cost_gae_lambda)
        self.cost_learning_rate = float(cost_learning_rate)
        self.cost_features_dim = int(cost_features_dim)
        self.dual_controller = DualVariableController(
            cost_limit=cost_limit,
            learning_rate=dual_learning_rate,
            maximum=dual_maximum,
            ema_beta=dual_ema_beta,
            warmup_updates=dual_warmup_updates,
            value=initial_lagrange_multiplier,
        )
        self._episode_cost_accumulators: np.ndarray | None = None
        self._latest_dual_diagnostics: dict[
            str,
            Any,
        ] = {}

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=(normalize_advantage),
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=(DualCriticMaskableDictRolloutBuffer),
            rollout_buffer_kwargs={
                "cost_gamma": cost_gamma,
                "cost_gae_lambda": (cost_gae_lambda),
            },
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        if _init_setup_model:
            self._setup_model()

    @property
    def lagrange_multiplier(self) -> float:
        return float(self.dual_controller.value)

    def _setup_model(self) -> None:
        super()._setup_model()

        if not isinstance(
            self.observation_space,
            spaces.Dict,
        ):
            raise TypeError("DualCriticMaskablePPO requires a Dict observation space.")
        if not isinstance(
            self.rollout_buffer,
            DualCriticMaskableDictRolloutBuffer,
        ):
            raise TypeError("Unexpected rollout buffer type.")

        self.cost_value_network = CostValueNetwork(
            self.observation_space,
            features_dim=self.cost_features_dim,
        ).to(self.device)
        self.cost_optimizer = torch.optim.Adam(
            self.cost_value_network.parameters(),
            lr=self.cost_learning_rate,
        )
        self._episode_cost_accumulators = np.zeros(
            self.n_envs,
            dtype=np.float64,
        )

    def _get_torch_save_params(
        self,
    ) -> tuple[list[str], list[str]]:
        state_dicts, torch_variables = super()._get_torch_save_params()
        return (
            [
                *state_dicts,
                "cost_value_network",
                "cost_optimizer",
            ],
            torch_variables,
        )

    def get_dual_diagnostics(
        self,
    ) -> dict[str, Any]:
        return dict(self._latest_dual_diagnostics)

    def _update_dual_controller(
        self,
        *,
        completed_episode_costs: list[float],
        mean_step_cost: float,
    ) -> None:
        multiplier_before = self.dual_controller.value
        observed_episode_cost: float | None

        if completed_episode_costs:
            observed_episode_cost = float(np.mean(completed_episode_costs))
            multiplier_after = self.dual_controller.update(observed_episode_cost)
            updated = True
        else:
            observed_episode_cost = None
            multiplier_after = multiplier_before
            updated = False

        controller_diagnostics = self.dual_controller.diagnostics()
        self._latest_dual_diagnostics = {
            "timesteps": int(self.num_timesteps),
            "rollout_updates": int(controller_diagnostics["update_count"]),
            "completed_episodes": len(completed_episode_costs),
            "mean_episode_cost": (observed_episode_cost),
            "mean_step_cost": float(mean_step_cost),
            "cost_limit": self.cost_limit,
            "multiplier_before": float(multiplier_before),
            "multiplier_after": float(multiplier_after),
            "ema_episode_cost": (controller_diagnostics["ema_cost"]),
            "dual_updated": updated,
            "dual_warmup_updates": int(self.dual_controller.warmup_updates),
        }

        self.logger.record(
            "dual_critic/mean_step_cost",
            mean_step_cost,
        )
        self.logger.record(
            "dual_critic/lagrange_multiplier",
            float(multiplier_after),
        )
        if observed_episode_cost is not None:
            self.logger.record(
                "dual_critic/mean_episode_cost",
                observed_episode_cost,
            )
        if controller_diagnostics["ema_cost"] is not None:
            self.logger.record(
                "dual_critic/ema_episode_cost",
                float(controller_diagnostics["ema_cost"]),
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        if not isinstance(
            rollout_buffer,
            DualCriticMaskableDictRolloutBuffer,
        ):
            raise TypeError("The rollout buffer does not support dual critics.")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided.")
        if self._episode_cost_accumulators is None:
            self._episode_cost_accumulators = np.zeros(
                env.num_envs,
                dtype=np.float64,
            )

        self.policy.set_training_mode(False)
        self.cost_value_network.eval()

        step_count = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking.")

        completed_episode_costs: list[float] = []
        rollout_step_cost_sum = 0.0
        rollout_transition_count = 0

        callback.on_rollout_start()

        while step_count < n_rollout_steps:
            with torch.no_grad():
                observation_tensor = obs_as_tensor(
                    self._last_obs,
                    self.device,
                )
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probabilities = self.policy(
                    observation_tensor,
                    action_masks=action_masks,
                )
                cost_values = self.cost_value_network(observation_tensor)

            actions = actions.cpu().numpy()
            (
                new_observations,
                rewards,
                dones,
                infos,
            ) = env.step(actions)

            raw_costs = np.asarray(
                [float(info.get("cost", 0.0)) for info in infos],
                dtype=np.float32,
            )
            critic_costs = raw_costs.copy()

            self.num_timesteps += env.num_envs
            rollout_step_cost_sum += float(np.sum(raw_costs))
            rollout_transition_count += env.num_envs

            self._episode_cost_accumulators += raw_costs
            for index, done in enumerate(dones):
                if bool(done):
                    completed_episode_costs.append(float(self._episode_cost_accumulators[index]))
                    self._episode_cost_accumulators[index] = 0.0

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            step_count += 1

            if isinstance(
                self.action_space,
                spaces.Discrete,
            ):
                actions = actions.reshape(-1, 1)

            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get(
                        "TimeLimit.truncated",
                        False,
                    )
                ):
                    terminal_observation = self.policy.obs_to_tensor(
                        infos[index]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_reward_value = self.policy.predict_values(terminal_observation)[0]
                        terminal_cost_value = self.cost_value_network(terminal_observation)[0]

                    rewards[index] += self.gamma * terminal_reward_value
                    critic_costs[index] += self.cost_gamma * float(terminal_cost_value.item())

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probabilities,
                costs=critic_costs,
                cost_values=cost_values,
                action_masks=action_masks,
            )
            self._last_obs = new_observations
            self._last_episode_starts = dones

        with torch.no_grad():
            final_observation_tensor = obs_as_tensor(
                new_observations,
                self.device,
            )
            final_reward_values = self.policy.predict_values(final_observation_tensor)
            final_cost_values = self.cost_value_network(final_observation_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=final_reward_values,
            dones=dones,
        )
        rollout_buffer.compute_cost_returns_and_advantage(
            last_cost_values=final_cost_values,
            dones=dones,
        )

        mean_step_cost = rollout_step_cost_sum / max(1, rollout_transition_count)
        self._update_dual_controller(
            completed_episode_costs=(completed_episode_costs),
            mean_step_cost=mean_step_cost,
        )

        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self.cost_value_network.train()
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_value = None
        if self.clip_range_vf is not None:
            clip_range_value = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        policy_losses: list[float] = []
        reward_value_losses: list[float] = []
        cost_value_losses: list[float] = []
        clip_fractions: list[float] = []
        approximate_kls: list[float] = []
        reward_objectives: list[float] = []
        cost_objectives: list[float] = []

        continue_training = True
        final_total_loss = torch.zeros(
            (),
            device=self.device,
        )

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(
                    self.action_space,
                    spaces.Discrete,
                ):
                    actions = rollout_data.actions.long().flatten()

                (
                    reward_values,
                    log_probabilities,
                    entropy,
                ) = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=(rollout_data.action_masks),
                )
                reward_values = reward_values.flatten()

                reward_advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages

                if self.normalize_advantage:
                    reward_advantages = (reward_advantages - reward_advantages.mean()) / (
                        reward_advantages.std() + 1e-8
                    )
                    cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                        cost_advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_probabilities - rollout_data.old_log_prob)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )

                reward_surrogate = torch.minimum(
                    reward_advantages * ratio,
                    reward_advantages * clipped_ratio,
                )
                cost_surrogate = torch.maximum(
                    cost_advantages * ratio,
                    cost_advantages * clipped_ratio,
                )

                multiplier = self.lagrange_multiplier
                policy_loss = -(reward_surrogate - multiplier * cost_surrogate).mean() / (
                    1.0 + multiplier
                )

                if clip_range_value is None:
                    predicted_reward_values = reward_values
                else:
                    predicted_reward_values = rollout_data.old_values + torch.clamp(
                        reward_values - rollout_data.old_values,
                        -clip_range_value,
                        clip_range_value,
                    )

                reward_value_loss = functional.mse_loss(
                    rollout_data.returns,
                    predicted_reward_values,
                )

                if entropy is None:
                    entropy_loss = -torch.mean(-log_probabilities)
                else:
                    entropy_loss = -torch.mean(entropy)

                total_policy_loss = (
                    policy_loss + self.ent_coef * entropy_loss + self.vf_coef * reward_value_loss
                )

                with torch.no_grad():
                    log_ratio = log_probabilities - rollout_data.old_log_prob
                    approximate_kl = (
                        torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).cpu().item()
                    )

                approximate_kls.append(float(approximate_kl))

                if self.target_kl is not None and approximate_kl > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping due to KL={approximate_kl:.6f}")
                    break

                self.policy.optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

                predicted_cost_values = self.cost_value_network(rollout_data.observations)
                cost_value_loss = functional.smooth_l1_loss(
                    predicted_cost_values,
                    rollout_data.cost_returns,
                )

                self.cost_optimizer.zero_grad()
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.cost_value_network.parameters(),
                    self.max_grad_norm,
                )
                self.cost_optimizer.step()

                final_total_loss = total_policy_loss.detach()
                policy_losses.append(float(policy_loss.item()))
                reward_value_losses.append(float(reward_value_loss.item()))
                cost_value_losses.append(float(cost_value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                reward_objectives.append(float(reward_surrogate.mean().item()))
                cost_objectives.append(float(cost_surrogate.mean().item()))
                clip_fractions.append(
                    float(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).item())
                )

            self._n_updates += 1
            if not continue_training:
                break

        reward_explained_variance = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        cost_explained_variance = explained_variance(
            self.rollout_buffer.cost_values.flatten(),
            self.rollout_buffer.cost_returns.flatten(),
        )

        self.logger.record(
            "train/entropy_loss",
            float(np.mean(entropy_losses)),
        )
        self.logger.record(
            "train/policy_gradient_loss",
            float(np.mean(policy_losses)),
        )
        self.logger.record(
            "train/reward_value_loss",
            float(np.mean(reward_value_losses)),
        )
        self.logger.record(
            "train/cost_value_loss",
            float(np.mean(cost_value_losses)),
        )
        self.logger.record(
            "train/reward_objective",
            float(np.mean(reward_objectives)),
        )
        self.logger.record(
            "train/cost_objective",
            float(np.mean(cost_objectives)),
        )
        self.logger.record(
            "train/approx_kl",
            float(np.mean(approximate_kls)),
        )
        self.logger.record(
            "train/clip_fraction",
            float(np.mean(clip_fractions)),
        )
        self.logger.record(
            "train/loss",
            float(final_total_loss.item()),
        )
        self.logger.record(
            "train/reward_explained_variance",
            reward_explained_variance,
        )
        self.logger.record(
            "train/cost_explained_variance",
            cost_explained_variance,
        )
        self.logger.record(
            "train/lagrange_multiplier",
            self.lagrange_multiplier,
        )
        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record(
            "train/clip_range",
            clip_range,
        )
        if clip_range_value is not None:
            self.logger.record(
                "train/clip_range_vf",
                clip_range_value,
            )
