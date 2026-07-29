from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


@dataclass
class PolicyOutput:
    distribution: Categorical
    reward_value: Tensor
    cost_value: Tensor
    subgoal_logits: Tensor
    recurrent_state: Tensor


class SemanticMapEncoder(nn.Module):
    """CNN spatial tokens with global positional attention."""

    def __init__(self, map_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, 64))
        self.position_embedding = nn.Parameter(torch.zeros(1, 17, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=192,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.projection = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, maps: Tensor) -> Tensor:
        tokens = self.convolution(maps).flatten(2).transpose(1, 2)
        if tokens.shape[1] != 16:
            raise ValueError(f"Expected 4x4 spatial tokens, got {tokens.shape[1]}")
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        attended = self.attention(torch.cat((class_token, tokens), dim=1) + self.position_embedding)
        return self.projection(attended[:, 0])


class RecurrentMaskedPolicy(nn.Module):
    """CNN/MLP/GRU actor with independent reward and cost value heads."""

    def __init__(
        self,
        *,
        map_channels: int = 11,
        state_features: int = 17,
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
        self.cost_critic = nn.Linear(recurrent_hidden_size, 1)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(1, batch_size, self.recurrent_hidden_size, device=device)

    def encode(self, maps: Tensor, states: Tensor) -> Tensor:
        if maps.ndim != 5 or states.ndim != 3:
            raise ValueError("Expected maps [B,T,C,H,W] and states [B,T,F]")
        batch, sequence = maps.shape[:2]
        flat_maps = maps.reshape(batch * sequence, *maps.shape[2:])
        map_features = self.map_encoder(flat_maps)
        state_features = self.state_encoder(states.reshape(batch * sequence, states.shape[-1]))
        return torch.cat((map_features, state_features), dim=-1).reshape(batch, sequence, -1)

    @staticmethod
    def masked_logits(logits: Tensor, masks: Tensor) -> Tensor:
        if logits.shape != masks.shape:
            raise ValueError(f"Logit and mask shapes differ: {logits.shape} != {masks.shape}")
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
    ) -> PolicyOutput:
        batch, sequence = maps.shape[:2]
        if recurrent_state is None:
            recurrent_state = self.initial_state(batch, maps.device)
        encoded = self.encode(maps.float(), states.float())
        if episode_starts is None:
            features, recurrent_state = self.recurrent(encoded, recurrent_state)
        else:
            outputs = []
            hidden = recurrent_state
            for index in range(sequence):
                reset = episode_starts[:, index].view(1, batch, 1).bool()
                hidden = torch.where(reset, torch.zeros_like(hidden), hidden)
                output, hidden = self.recurrent(encoded[:, index : index + 1], hidden)
                outputs.append(output)
            features = torch.cat(outputs, dim=1)
            recurrent_state = hidden
        logits = self.masked_logits(self.actor(features), action_masks)
        return PolicyOutput(
            distribution=Categorical(logits=logits),
            reward_value=self.reward_critic(features).squeeze(-1),
            cost_value=self.cost_critic(features).squeeze(-1),
            subgoal_logits=self.subgoal_head(features),
            recurrent_state=recurrent_state,
        )

    @torch.no_grad()
    def predict(
        self,
        maps: Tensor,
        states: Tensor,
        action_masks: Tensor,
        recurrent_state: Tensor | None = None,
        *,
        deterministic: bool = True,
        episode_starts: Tensor | None = None,
    ) -> tuple[Tensor, PolicyOutput]:
        output = self(maps, states, action_masks, recurrent_state, episode_starts)
        action = (
            output.distribution.probs.argmax(dim=-1)
            if deterministic
            else output.distribution.sample()
        )
        return action, output
