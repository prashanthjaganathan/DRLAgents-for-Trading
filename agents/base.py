"""Abstract base class that all agents must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAgent(ABC):
    """
    Shared interface for DQN, PolicyGradient, ModelBased, and HRL agents.

    This lets scripts/train.py and scripts/evaluate.py work with ANY agent
    polymorphically — just pass the agent instance.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config

    @abstractmethod
    def select_action(self, obs: np.ndarray, *, explore: bool = True) -> int:
        """
        Choose an action given an observation.

        Args:
            obs:     Current observation from the environment.
            explore: If True, use exploration strategy (e.g. epsilon-greedy, stochastic policy).
                     If False, act greedily (for evaluation).
        """
        ...

    @abstractmethod
    def learn(self, **kwargs) -> dict[str, float]:
        """
        Run one learning update.

        Returns a dict of training metrics (e.g. {"loss": 0.05, "entropy": 1.2}).
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model weights and optimizer state."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model weights and optimizer state."""
        ...

    def on_episode_end(self, episode: int, info: dict) -> None:
        """Optional hook for end-of-episode bookkeeping (e.g. decay epsilon)."""
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(obs={self.obs_dim}, act={self.act_dim})"
