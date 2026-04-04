"""RuleAgent — deterministic rule-based HelixDesk policy."""

import numpy as np

from helixdesk.agents.base_agent import AbstractAgent


class RuleAgent(AbstractAgent):
    """Hand-crafted rule-based agent implementing full business logic.

    Serves as the upper-bound baseline before RL training. Implements
    deterministic rules in priority order:

    1. Keyword flag → complaint, critical, least-loaded employee
    2. High sentiment (>0.85) → complaint, high priority
    3. Enterprise tier → complaint, high priority
    4. All employees at capacity → flag for human review
    5. Query-like pattern → query, auto-reply from KB
    6. Default → complaint, medium, least-loaded employee
    """

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Apply deterministic rules to produce an action.

        Args:
            observation: The 42-dim observation vector.

        Returns:
            Action array: [classify, priority, assign, secondary].
        """
        sentiment = observation[0]
        has_keyword_flag = observation[1] >= 0.5
        is_enterprise = observation[2] >= 0.5
        # is_standard = observation[3] >= 0.5
        # is_free = observation[4] >= 0.5

        # Employee loads (dims 15, 17, 19, 21, 23)
        employee_loads = [observation[15 + i * 2] for i in range(5)]

        # Find least-loaded employee
        least_loaded_idx = int(np.argmin(employee_loads))
        min_load = employee_loads[least_loaded_idx]

        # Check if all employees are at capacity (load_norm >= 1.0)
        all_at_capacity = all(load >= 1.0 for load in employee_loads)

        # --- Rule 1: Keyword flag ---
        if has_keyword_flag:
            if all_at_capacity:
                return np.array([2, 3, 5, 2], dtype=np.int64)  # flag for review
            return np.array([1, 0, least_loaded_idx, 1], dtype=np.int64)
            # complaint, critical, least-loaded, alert GM

        # --- Rule 2: High sentiment ---
        if sentiment > 0.85:
            if all_at_capacity:
                return np.array([2, 3, 5, 2], dtype=np.int64)
            return np.array([1, 1, least_loaded_idx, 2], dtype=np.int64)
            # complaint, high, least-loaded, no secondary

        # --- Rule 3: Enterprise tier ---
        if is_enterprise:
            if all_at_capacity:
                return np.array([2, 3, 5, 2], dtype=np.int64)
            return np.array([1, 1, least_loaded_idx, 2], dtype=np.int64)
            # complaint, high, least-loaded, no secondary

        # --- Rule 4: All at capacity ---
        if all_at_capacity:
            return np.array([2, 3, 5, 2], dtype=np.int64)  # flag for review

        # --- Rule 5: Low sentiment suggests query ---
        if sentiment < 0.4:
            return np.array([0, 3, 5, 0], dtype=np.int64)
            # query, normal (ignored), no assignment, auto-reply from KB

        # --- Rule 6: Default — complaint, medium ---
        return np.array([1, 2, least_loaded_idx, 2], dtype=np.int64)
        # complaint, medium, least-loaded, no secondary
