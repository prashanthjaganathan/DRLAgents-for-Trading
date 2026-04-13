"""Neural network architectures for policy gradient agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Maps observations to action probabilities.

    Used by all three agents: REINFORCE, REINFORCE+Baseline, PPO.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (unnormalized log-probabilities)."""
        return self.net(x)

    def get_distribution(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> torch.distributions.Categorical:
        """
        Returns a Categorical distribution over actions.

        Args:
            obs:         Observation tensor.
            action_mask: Boolean tensor of valid actions. Invalid actions get -inf logits.
        """
        logits = self.forward(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """
    Maps observations to a scalar state-value estimate V(s).

    Used by REINFORCE+Baseline and PPO (the critic).
    """

    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate."""
        return self.net(x).squeeze(-1)
