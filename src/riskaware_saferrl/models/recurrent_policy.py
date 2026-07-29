from __future__ import annotations

import torch
from torch import nn


class RecurrentSemanticPolicy(nn.Module):
    """Partial-observation policy using only observation histories and recurrent state."""

    def __init__(
        self, observation_size: int = 1012, hidden_size: int = 128, actions: int = 5
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_size, 256), nn.ReLU())
        self.recurrent = nn.GRU(256, hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(
        self, observation_history: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation_history.ndim != 3:
            raise ValueError("observation_history must have shape [batch, time, features].")
        encoded = self.encoder(observation_history)
        if state is None:
            state = self.initial_state(observation_history.shape[0], observation_history.device)
        recurrent_features, next_state = self.recurrent(encoded, state)
        return (
            self.policy_head(recurrent_features),
            self.value_head(recurrent_features),
            next_state,
        )
