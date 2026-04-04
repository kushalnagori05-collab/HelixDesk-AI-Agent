"""TrendWatchdog — 72-hour rolling volume monitor for complaint surges."""

from __future__ import annotations


class TrendWatchdog:
    """Monitors complaint volume per category over a configurable time window.

    Detects surges by comparing the current window's volume to the prior
    window's volume. Alerts when growth exceeds the configured threshold.
    """

    def __init__(self, config: dict):
        env_cfg = config["env"]
        self._window_minutes: float = env_cfg["trend_window_hours"] * 60.0
        self._threshold_pct: float = env_cfg["trend_alert_threshold"]
        self._categories: list[str] = config["email_gen"]["categories"]
        self._history: dict[str, list[tuple[float, str]]] = {}
        # maps category → [(sim_time_minutes, ticket_id), ...]

    def record(self, category: str, sim_time: float) -> None:
        """Record a new complaint in this category at the given simulation time.

        Args:
            category: The complaint category.
            sim_time: Current simulation time in minutes.
        """
        if category not in self._history:
            self._history[category] = []
        self._history[category].append((sim_time, f"t_{sim_time:.0f}"))

    def tick(self, current_time: float) -> list[str]:
        """Check for surge alerts in all categories.

        Prunes history older than 2× the window, then computes growth rate
        for each category:
          prior_window  = events in [t - 2*window, t - window]
          current_window = events in [t - window, t]
          growth = (current - prior) / max(prior, 1) × 100

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            List of category names where growth >= threshold.
        """
        alerts: list[str] = []
        cutoff_old = current_time - 2 * self._window_minutes

        for category in self._categories:
            entries = self._history.get(category, [])

            # Prune entries older than 2 windows
            entries = [(t, tid) for t, tid in entries if t >= cutoff_old]
            self._history[category] = entries

            # Split into prior and current windows
            prior_cutoff = current_time - self._window_minutes
            prior_count = sum(1 for t, _ in entries if t < prior_cutoff)
            current_count = sum(1 for t, _ in entries if t >= prior_cutoff)

            # Compute growth percentage
            growth_pct = (current_count - prior_count) / max(prior_count, 1) * 100.0

            if growth_pct >= self._threshold_pct:
                alerts.append(category)

        return alerts

    def get_growth_rates(self, current_time: float) -> dict[str, float]:
        """Return growth rate as a fraction (not percent) for each category.

        The returned value is clipped to [-1.0, 1.0] for use in the
        observation vector.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            Dict mapping category name → growth fraction in [-1.0, 1.0].
        """
        rates: dict[str, float] = {}
        for category in self._categories:
            entries = self._history.get(category, [])
            prior_cutoff = current_time - self._window_minutes
            prior_count = sum(1 for t, _ in entries if t < prior_cutoff)
            current_count = sum(1 for t, _ in entries if t >= prior_cutoff)

            growth = (current_count - prior_count) / max(prior_count, 1)
            rates[category] = max(-1.0, min(1.0, growth))

        return rates

    def reset(self) -> None:
        """Clear all volume history."""
        self._history = {}
