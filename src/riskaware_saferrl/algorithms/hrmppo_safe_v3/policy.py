from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from riskaware_saferrl.policies.recurrent_masked_policy import SemanticMapEncoder
from riskaware_saferrl.safety.safety_contract_v3 import VECTOR_COST_NAMES


@dataclass
class SafePolicyOutputV3:
    distribution: Categorical
    reward_value: Tensor
    vector_cost_values: Tensor
    subgoal_logits: Tensor
    recurrent_state: Tensor


class RecurrentMaskedSafePolicyV3(nn.Module):
    """Recurrent masked actor with five independent safety-value critics."""

    def __init__(
        self,
        *,
        map_channels: int = 11,
        state_features: int = 22,
        action_count: int = 5,
        subgoal_count: int = 6,
        recurrent_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.map_channels = map_channels
        self.state_features = state_features
        self.action_count = action_count
        self.subgoal_count = subgoal_count
        self.recurrent_hidden_size = recurrent_hidden_size
        self.map_encoder = SemanticMapEncoder(map_channels)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_features, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.recurrent = nn.GRU(192, recurrent_hidden_size, batch_first=True)
        self.actor = nn.Linear(recurrent_hidden_size, action_count)
        self.subgoal_head = nn.Linear(recurrent_hidden_size, subgoal_count)
        self.reward_critic = nn.Linear(recurrent_hidden_size, 1)
        self.cost_critics = nn.ModuleDict(
            {name: nn.Linear(recurrent_hidden_size, 1) for name in VECTOR_COST_NAMES}
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(1, batch_size, self.recurrent_hidden_size, device=device)

    @classmethod
    def from_v2_checkpoint(
        cls, checkpoint: dict[str, object], device: torch.device | str
    ) -> RecurrentMaskedSafePolicyV3:
        policy = cls().to(device)
        source = checkpoint["model_state_dict"]
        if not isinstance(source, dict):
            raise TypeError("v2 checkpoint model state must be a dictionary")
        target = policy.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        source_state_weight = source["state_encoder.0.weight"]
        target_state_weight = target["state_encoder.0.weight"]
        target_state_weight[:, : source_state_weight.shape[1]] = source_state_weight
        compatible["state_encoder.0.weight"] = target_state_weight
        policy.load_state_dict(compatible, strict=False)
        for critic in policy.cost_critics.values():
            if "cost_critic.weight" in source:
                critic.weight.data.copy_(source["cost_critic.weight"])
                critic.bias.data.copy_(source["cost_critic.bias"])
        return policy

    def encode(self, maps: Tensor, states: Tensor) -> Tensor:
        if maps.ndim != 5 or states.ndim != 3:
            raise ValueError("Expected maps [B,T,C,H,W] and states [B,T,F]")
        if states.shape[-1] != self.state_features:
            raise ValueError(
                f"Expected {self.state_features} state features, got {states.shape[-1]}"
            )
        batch, sequence = maps.shape[:2]
        map_features = self.map_encoder(maps.reshape(batch * sequence, *maps.shape[2:]))
        state_features = self.state_encoder(states.reshape(batch * sequence, states.shape[-1]))
        return torch.cat((map_features, state_features), dim=-1).reshape(batch, sequence, -1)

    @staticmethod
    def masked_logits(logits: Tensor, masks: Tensor) -> Tensor:
        if logits.shape != masks.shape:
            raise ValueError("Action logits and masks must have identical shapes")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("Every action mask must permit at least one action")
        return logits.masked_fill(~masks.bool(), torch.finfo(logits.dtype).min)

    def forward(
        self,
        maps: Tensor,
        states: Tensor,
        action_masks: Tensor,
        recurrent_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> SafePolicyOutputV3:
        batch, sequence = maps.shape[:2]
        hidden = (
            self.initial_state(batch, maps.device) if recurrent_state is None else recurrent_state
        )
        encoded = self.encode(maps.float(), states.float())
        if episode_starts is None:
            features, hidden = self.recurrent(encoded, hidden)
        else:
            outputs = []
            for index in range(sequence):
                reset = episode_starts[:, index].view(1, batch, 1).bool()
                hidden = torch.where(reset, torch.zeros_like(hidden), hidden)
                output, hidden = self.recurrent(encoded[:, index : index + 1], hidden)
                outputs.append(output)
            features = torch.cat(outputs, dim=1)
        vector_values = torch.stack(
            [self.cost_critics[name](features).squeeze(-1) for name in VECTOR_COST_NAMES],
            dim=-1,
        )
        return SafePolicyOutputV3(
            distribution=Categorical(logits=self.masked_logits(self.actor(features), action_masks)),
            reward_value=self.reward_critic(features).squeeze(-1),
            vector_cost_values=vector_values,
            subgoal_logits=self.subgoal_head(features),
            recurrent_state=hidden,
        )
