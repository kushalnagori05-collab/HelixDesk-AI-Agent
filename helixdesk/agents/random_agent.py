"""RandomAgent — uniform-random baseline agent."""

import numpy as np

from helixdesk.agents.base_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """Agent that takes uniformly random actions. Lower-bound baseline."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Sample a random action from the action space."""
        return self.action_space.sample()
