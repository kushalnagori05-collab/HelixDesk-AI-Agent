"""AbstractAgent — base class for all HelixDesk agents."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractAgent(ABC):
    """Base class for HelixDesk agents.

    Subclasses must implement act(). learn() and reset() are optional.
    """

    def __init__(self, observation_space, action_space, config=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Given an observation, return an action array of shape (4,).

        Args:
            observation: The 42-dim observation vector.

        Returns:
            Action array: [classify, priority, assign, secondary].
        """

    def learn(self, obs, action, reward, next_obs, terminated, info):
        """Optional: update internal policy. No-op by default."""

    def reset(self):
        """Called at episode start. No-op by default."""
