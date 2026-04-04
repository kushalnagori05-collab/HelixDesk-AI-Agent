"""
HelixDesk OpenEnv — Observation and Action Space definitions.

Observation: Box(low=-1, high=1, shape=(42,), dtype=float32)
Action:      MultiDiscrete([3, 4, 6, 3])

Category encoding (dims 5–9) for 8 categories using 5 slots:
  Cat 0 (login_failure):   [1, 0, 0, 0, 0]
  Cat 1 (billing_dispute):  [0, 1, 0, 0, 0]
  Cat 2 (refund_request):   [0, 0, 1, 0, 0]
  Cat 3 (product_defect):   [0, 0, 0, 1, 0]
  Cat 4 (shipping_delay):   [0, 0, 0, 0, 1]
  Cat 5 (account_locked):   [1, 0, 0, 0, 1]  ← overflow: reuse dim5 + flag
  Cat 6 (data_privacy):     [0, 1, 0, 0, 1]  ← overflow: reuse dim6 + flag
  Cat 7 (general_query):    [0, 0, 1, 0, 1]  ← overflow: reuse dim7 + flag
"""

import gymnasium
import numpy as np

# ---------------------------------------------------------------------------
# Observation dimension layout (42 total)
# ---------------------------------------------------------------------------
# fmt: off
OBS_DIMS = {
    # Current email features (dims 0–9)
    "sentiment_intensity":        0,
    "has_keyword_flag":           1,
    "customer_tier_enterprise":   2,
    "customer_tier_standard":     3,
    "customer_tier_free":         4,
    "category_0":                 5,
    "category_1":                 6,
    "category_2":                 7,
    "category_3":                 8,
    "category_4_overflow":        9,   # also overflow flag for cats 5-7

    # Queue state (dims 10–14)
    "critical_count_norm":       10,
    "high_count_norm":           11,
    "medium_count_norm":         12,
    "normal_count_norm":         13,
    "review_queue_count_norm":   14,

    # Team state (dims 15–24) — 5 employees × 2 features
    "employee_0_load_norm":       15,
    "employee_0_avg_resolve_norm":16,
    "employee_1_load_norm":       17,
    "employee_1_avg_resolve_norm":18,
    "employee_2_load_norm":       19,
    "employee_2_avg_resolve_norm":20,
    "employee_3_load_norm":       21,
    "employee_3_avg_resolve_norm":22,
    "employee_4_load_norm":       23,
    "employee_4_avg_resolve_norm":24,

    # SLA state (dims 25–28)
    "overdue_count_norm":        25,
    "near_deadline_count_norm":  26,
    "sla_pressure":              27,
    "critical_overdue_flag":     28,

    # Trend state (dims 29–36) — 8 categories growth rates
    "trend_growth_cat_0":        29,
    "trend_growth_cat_1":        30,
    "trend_growth_cat_2":        31,
    "trend_growth_cat_3":        32,
    "trend_growth_cat_4":        33,
    "trend_growth_cat_5":        34,
    "trend_growth_cat_6":        35,
    "trend_growth_cat_7":        36,

    # Time state (dims 37–38)
    "hour_of_day_norm":          37,
    "day_of_week_norm":          38,

    # Episode progress (dims 39–41)
    "steps_remaining_norm":      39,
    "episode_reward_norm":       40,
    "agent_confidence":          41,
}
# fmt: on

OBS_SIZE = 42

# ---------------------------------------------------------------------------
# Category encoding lookup
# ---------------------------------------------------------------------------
# Maps category index (0-7) → values for dims 5-9
CATEGORY_ENCODING = {
    0: [1.0, 0.0, 0.0, 0.0, 0.0],   # login_failure
    1: [0.0, 1.0, 0.0, 0.0, 0.0],   # billing_dispute
    2: [0.0, 0.0, 1.0, 0.0, 0.0],   # refund_request
    3: [0.0, 0.0, 0.0, 1.0, 0.0],   # product_defect
    4: [0.0, 0.0, 0.0, 0.0, 1.0],   # shipping_delay
    5: [1.0, 0.0, 0.0, 0.0, 1.0],   # account_locked  (overflow)
    6: [0.0, 1.0, 0.0, 0.0, 1.0],   # data_privacy    (overflow)
    7: [0.0, 0.0, 1.0, 0.0, 1.0],   # general_query   (overflow)
}

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
# Dimension 0 — Classification
ACTION_CLASSIFY_QUERY = 0
ACTION_CLASSIFY_COMPLAINT = 1
ACTION_FLAG_FOR_REVIEW = 2

# Dimension 1 — Priority
ACTION_PRIORITY_CRITICAL = 0
ACTION_PRIORITY_HIGH = 1
ACTION_PRIORITY_MEDIUM = 2
ACTION_PRIORITY_NORMAL = 3

# Dimension 2 — Assignment target
ACTION_ASSIGN_EMP_0 = 0
ACTION_ASSIGN_EMP_1 = 1
ACTION_ASSIGN_EMP_2 = 2
ACTION_ASSIGN_EMP_3 = 3
ACTION_ASSIGN_EMP_4 = 4
ACTION_ASSIGN_NO_ASSIGNMENT = 5

# Dimension 3 — Secondary action
ACTION_SECONDARY_AUTO_REPLY = 0
ACTION_SECONDARY_ALERT_GM = 1
ACTION_SECONDARY_NONE = 2

# Forced action when flagging for review
REVIEW_FORCED_ACTION = (ACTION_PRIORITY_NORMAL, ACTION_ASSIGN_NO_ASSIGNMENT, ACTION_SECONDARY_NONE)


def build_observation_space() -> gymnasium.spaces.Box:
    """Build the 42-dimensional observation space. All values normalised to [-1, 1]."""
    return gymnasium.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(OBS_SIZE,),
        dtype=np.float32,
    )


def build_action_space() -> gymnasium.spaces.MultiDiscrete:
    """Build the MultiDiscrete action space: [classify(3), priority(4), assign(6), secondary(3)]."""
    return gymnasium.spaces.MultiDiscrete([3, 4, 6, 3])


def encode_category(category_name: str, categories_list: list[str]) -> list[float]:
    """
    Encode a category name into its 5-float representation for dims 5-9.

    Args:
        category_name: The category string (e.g. "login_failure")
        categories_list: Ordered list of all category names from config

    Returns:
        List of 5 floats representing the one-hot + overflow encoding.
    """
    idx = categories_list.index(category_name)
    return CATEGORY_ENCODING[idx]


def encode_customer_tier(tier: str) -> tuple[float, float, float]:
    """
    Encode customer tier into 3-float one-hot for dims 2-4.

    Returns:
        (enterprise, standard, free) — exactly one is 1.0
    """
    if tier == "enterprise":
        return (1.0, 0.0, 0.0)
    elif tier == "standard":
        return (0.0, 1.0, 0.0)
    else:  # "free"
        return (0.0, 0.0, 1.0)
