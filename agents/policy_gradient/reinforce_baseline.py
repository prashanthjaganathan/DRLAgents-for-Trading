"""REINFORCE with learned value baseline for variance reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from agents.policy_gradient.network import PolicyNetwork, ValueNetwork


class ReinforceBaselineAgent(BaseAgent):
    """
    REINFORCE with a learned baseline V(s).

    Policy loss:  -sum( log_prob(a_t) * A_t )   where A_t = G_t - V(s_t)
    Value loss:   MSE( V(s_t), G_t )

    The advantage A_t centers the return around the expected value,
    dramatically reducing gradient variance compared to vanilla REINFORCE.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        super().__init__(obs_dim, act_dim, config)

        self.gamma = config.get("gamma", 0.99)
        self.lr_policy = config.get("lr_policy", 3e-4)
        self.lr_value = config.get("lr_value", 1e-3)

        hidden = config.get("hidden", 128)
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden)
        self.value = ValueNetwork(obs_dim, hidden)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.lr_value)

        # episode buffers
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._obs: list[torch.Tensor] = []

    def select_action(
        self, obs: np.ndarray, *, explore: bool = True, action_mask: np.ndarray | None = None
    ) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0) if action_mask is not None else None

        dist = self.policy.get_distribution(obs_t, mask_t)
        v = self.value(obs_t)

        action = dist.sample() if explore else dist.probs.argmax(dim=-1)

        self._log_probs.append(dist.log_prob(action))
        self._values.append(v)
        self._obs.append(obs_t)

        return action.item()

    def store_reward(self, reward: float) -> None:
        """Call after each env.step() to buffer the reward."""
        self._rewards.append(reward)

    def learn(self, **kwargs) -> dict[str, float]:
        """Update policy and value network after a full episode."""
        if not self._rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        # --- discounted returns ---
        returns = self._compute_returns()
        returns_t = torch.FloatTensor(returns)

        # --- advantages: A_t = G_t - V(s_t) ---
        values_t = torch.cat(self._values).detach()
        advantages = returns_t - values_t

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # --- policy loss ---
        policy_loss = torch.tensor(0.0)
        for log_prob, adv in zip(self._log_probs, advantages):
            policy_loss -= log_prob * adv

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --- value loss (MSE) ---
        all_obs = torch.cat(self._obs)
        predicted_values = self.value(all_obs)
        value_loss = nn.functional.mse_loss(predicted_values, returns_t)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # --- clear buffers ---
        avg_reward = np.mean(self._rewards)
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._obs.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "avg_reward": avg_reward,
        }

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
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "policy_opt": self.policy_optimizer.state_dict(),
                "value_opt": self.value_optimizer.state_dict(),
            },
            path / "reinforce_baseline.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "reinforce_baseline.pt", weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
        self.policy_optimizer.load_state_dict(ckpt["policy_opt"])
        self.value_optimizer.load_state_dict(ckpt["value_opt"])
