from __future__ import annotations

import torch
from gymnasium import spaces
from torch import nn

from riskaware_saferrl.models import SemanticMapExtractor


class CostValueNetwork(nn.Module):
    """Independent semantic-map safety-cost critic."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        features_dim: int = 128,
    ) -> None:
        super().__init__()

        self.features_extractor = SemanticMapExtractor(
            observation_space,
            features_dim=features_dim,
        )
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        features = self.features_extractor(observations)
        return self.value_head(features).flatten()
