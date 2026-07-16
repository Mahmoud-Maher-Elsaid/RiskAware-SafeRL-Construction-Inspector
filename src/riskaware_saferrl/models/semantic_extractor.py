from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SemanticMapExtractor(BaseFeaturesExtractor):
    """Encode semantic-map and robot-state observations before policy learning."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        semantic_dim = int(np.prod(observation_space["map"].shape))
        state_dim = int(np.prod(observation_space["state"].shape))

        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(160, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        semantic_features = self.semantic_encoder(observations["map"].float())

        state_features = self.state_encoder(observations["state"].float())

        combined_features = torch.cat(
            (semantic_features, state_features),
            dim=1,
        )

        return self.fusion(combined_features)
