"""Hard task: trend detection + CSAT quality + overdue control.

Run one full episode (100 emails).

Score = product of 3 sub-scores, each in [0.0, 1.0]:
  trend_score   = fraction of surge-active steps where agent chose alert_gm
  csat_score    = min(avg_csat / 4.7, 1.0)   — very demanding threshold
  overdue_score = 1.0 if peak_overdue == 0, else max(0, 1 - peak_overdue / 3)

Product scoring ensures ALL 3 dimensions must be strong simultaneously.
A single weak axis drags the whole score down hard.

Target: rule agent <= 0.40, random agent <= 0.20
"""

from __future__ import annotations

import numpy as np

from helixdesk import HelixDeskEnv


def grade(env: HelixDeskEnv | None = None, agent=None) -> float:
    """Grade the agent on trend alerting, CSAT quality, and overdue control.

    Agent must simultaneously:
      1. Catch trend surge alerts (alert_gm when category surging)
      2. Maintain avg CSAT >= 4.7 (very demanding)
      3. Keep peak overdue count at 0 (zero tolerance)

    Score = product of 3 sub-scores — a single weak axis collapses the score.

    Args:
        env: Optional pre-built env. If None, creates one with default config.
        agent: Must implement .act(obs) -> action array.

    Returns:
        Score in [0.0, 1.0].
    """
    if env is None:
        env = HelixDeskEnv()

    obs, info = env.reset(seed=42)
    env.action_space.seed(42)  # seed action space for deterministic random agent

    total_surge_steps = 0
    trend_alerts_caught = 0
    csat_scores: list[float] = []
    peak_overdue = 0

    done = False
    while not done:
        action = agent.act(obs)
        action_arr = np.asarray(action, dtype=np.int64)

        obs, reward, terminated, truncated, info = env.step(action_arr)
        done = terminated or truncated

        # Track surge detection — count steps with active trend alerts
        active_alerts = info.get("trend_alerts_active", 0)
        if active_alerts > 0:
            total_surge_steps += 1
            secondary = int(action_arr[3])
            if secondary == 1:  # alert_gm
                trend_alerts_caught += 1

        # Track CSAT
        csat = info.get("csat_score")
        if csat is not None:
            csat_scores.append(float(csat))

        # Track peak overdue
        overdue = info.get("overdue_count", 0)
        if overdue > peak_overdue:
            peak_overdue = overdue

    env.close()

    # Sub-score 1: Trend catch rate
    if total_surge_steps > 0:
        trend_score = trend_alerts_caught / total_surge_steps
    else:
        trend_score = 0.0  # No surges = nothing to score

    # Sub-score 2: CSAT quality — threshold 4.7 (very demanding)
    if csat_scores:
        avg_csat = float(np.mean(csat_scores))
        csat_component = min(avg_csat / 4.7, 1.0)
    else:
        csat_component = 0.0

    # Sub-score 3: Overdue control — zero tolerance, penalty per overdue ticket
    if peak_overdue == 0:
        overdue_component = 1.0
    else:
        overdue_component = max(0.0, 1.0 - (peak_overdue / 3.0))

    # Product scoring — all 3 must be strong; one weak axis collapses everything
    final_score = trend_score * csat_component * overdue_component

    return max(0.0, min(1.0, final_score))
