from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from riskaware_saferrl.hierarchical.schemas import OPTION_COUNT, VECTOR_COST_NAMES
from riskaware_saferrl.policies.recurrent_masked_policy import SemanticMapEncoder


@dataclass
class HierarchicalPolicyOutput:
    option_distribution: Categorical
    target_logits: Tensor
    duration_logits: Tensor
    risk_budgets: Tensor
    replanning_urgency: Tensor
    inspection_intent: Tensor
    confidence: Tensor
    reward_value: Tensor
    vector_cost_values: Tensor
    recurrent_state: Tensor


class HierarchicalMissionPolicy(nn.Module):
    """Recurrent option policy operating above causal motion planning."""

    def __init__(
        self,
        *,
        map_channels: int = 11,
        state_features: int = 32,
        map_size: int = 16,
        option_count: int = OPTION_COUNT,
        duration_bins: int = 8,
        recurrent_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.map_channels = map_channels
        self.state_features = state_features
        self.map_size = map_size
        self.option_count = option_count
        self.duration_bins = duration_bins
        self.recurrent_hidden_size = recurrent_hidden_size
        self.map_encoder = SemanticMapEncoder(map_channels)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_features, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(224, recurrent_hidden_size, batch_first=True)
        self.option_actor = nn.Linear(recurrent_hidden_size, option_count)
        self.target_selection_head = nn.Linear(recurrent_hidden_size, map_size * map_size + 1)
        self.option_duration_head = nn.Linear(recurrent_hidden_size, duration_bins)
        self.risk_budget_head = nn.Linear(recurrent_hidden_size, len(VECTOR_COST_NAMES))
        self.replanning_head = nn.Linear(recurrent_hidden_size, 1)
        self.inspection_intent_head = nn.Linear(recurrent_hidden_size, 1)
        self.confidence_head = nn.Linear(recurrent_hidden_size, 1)
        self.reward_critic = nn.Linear(recurrent_hidden_size, 1)
        self.cost_critics = nn.ModuleList(
            nn.Linear(recurrent_hidden_size, 1) for _ in VECTOR_COST_NAMES
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(1, batch_size, self.recurrent_hidden_size, device=device)

    @staticmethod
    def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
        if logits.shape != mask.shape:
            raise ValueError("Option logits and mask must have identical shapes")
        valid = mask.bool()
        if not bool(torch.all(valid.any(dim=-1))):
            raise ValueError("Every option mask must permit at least one option")
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def forward(
        self,
        maps: Tensor,
        states: Tensor,
        option_masks: Tensor,
        recurrent_state: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> HierarchicalPolicyOutput:
        if maps.ndim != 5 or states.ndim != 3:
            raise ValueError("Expected maps [B,T,C,H,W] and states [B,T,F]")
        batch, sequence = maps.shape[:2]
        if states.shape[-1] != self.state_features:
            raise ValueError(f"Expected {self.state_features} structured state features")
        if recurrent_state is None:
            recurrent_state = self.initial_state(batch, maps.device)
        flat_maps = maps.float().reshape(batch * sequence, *maps.shape[2:])
        spatial = self.map_encoder(flat_maps).reshape(batch, sequence, -1)
        structured = self.state_encoder(states.float())
        encoded = torch.cat((spatial, structured), dim=-1)
        if episode_starts is None:
            features, hidden = self.recurrent(encoded, recurrent_state)
        else:
            outputs: list[Tensor] = []
            hidden = recurrent_state
            for index in range(sequence):
                reset = episode_starts[:, index].reshape(1, batch, 1).bool()
                hidden = torch.where(reset, torch.zeros_like(hidden), hidden)
                current, hidden = self.recurrent(encoded[:, index : index + 1], hidden)
                outputs.append(current)
            features = torch.cat(outputs, dim=1)
        option_logits = self.masked_logits(self.option_actor(features), option_masks)
        return HierarchicalPolicyOutput(
            option_distribution=Categorical(logits=option_logits),
            target_logits=self.target_selection_head(features),
            duration_logits=self.option_duration_head(features),
            risk_budgets=torch.sigmoid(self.risk_budget_head(features)),
            replanning_urgency=torch.sigmoid(self.replanning_head(features)).squeeze(-1),
            inspection_intent=torch.sigmoid(self.inspection_intent_head(features)).squeeze(-1),
            confidence=torch.sigmoid(self.confidence_head(features)).squeeze(-1),
            reward_value=self.reward_critic(features).squeeze(-1),
            vector_cost_values=torch.cat(
                [critic(features) for critic in self.cost_critics], dim=-1
            ),
            recurrent_state=hidden,
        )

    @torch.no_grad()
    def predict(
        self,
        maps: Tensor,
        states: Tensor,
        option_masks: Tensor,
        recurrent_state: Tensor | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[Tensor, HierarchicalPolicyOutput]:
        output = self(maps, states, option_masks, recurrent_state)
        option = (
            output.option_distribution.probs.argmax(dim=-1)
            if deterministic
            else output.option_distribution.sample()
        )
        return option, output
