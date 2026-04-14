"""Vanilla REINFORCE agent — simplest policy gradient method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from agents.base import BaseAgent
from agents.policy_gradient.network import PolicyNetwork


class ReinforceAgent(BaseAgent):
    """
    REINFORCE (Monte Carlo Policy Gradient).

    Collects a full episode, then updates the policy using:
        loss = -sum( log_prob(a_t) * G_t )
    where G_t is the discounted return from step t onward.

    Pros:  Simple, unbiased gradient estimate.
    Cons:  High variance, slow convergence.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        super().__init__(obs_dim, act_dim, config)

        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 3e-4)

        self.policy = PolicyNetwork(obs_dim, act_dim, config.get("hidden", 128))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # episode buffers (cleared after each learn() call)
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    def select_action(
        self, obs: np.ndarray, *, explore: bool = True, action_mask: np.ndarray | None = None
    ) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0) if action_mask is not None else None

        dist = self.policy.get_distribution(obs_t, mask_t)

        action = dist.sample() if explore else dist.probs.argmax(dim=-1)

        self._log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        """Call after each env.step() to buffer the reward."""
        self._rewards.append(reward)

    def learn(self, **kwargs) -> dict[str, float]:
        """
        Update policy after a full episode.

        Computes discounted returns G_t for each step, then does a single
        gradient update on the full episode.
        """
        if not self._rewards:
            return {"loss": 0.0}

        # --- compute discounted returns ---
        returns = self._compute_returns()

        # --- normalize returns (reduces variance) ---
        returns_t = torch.FloatTensor(returns)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / returns_t.std()

        # --- policy gradient loss ---
        loss = torch.tensor(0.0)
        for log_prob, g in zip(self._log_probs, returns_t):
            loss -= log_prob * g  # negative because we maximize

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- clear buffers ---
        avg_reward = np.mean(self._rewards)
        self._log_probs.clear()
        self._rewards.clear()

        return {"loss": loss.item(), "avg_reward": avg_reward}

    def _compute_returns(self) -> list[float]:
        """Compute discounted return G_t for each timestep."""
        returns = []
        g = 0.0
        for r in reversed(self._rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        return returns

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()},
            path / "reinforce.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "reinforce.pt", weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
