"""SimClock — simulated time tracker for HelixDesk episodes."""

import numpy as np


class SimClock:
    """Tracks simulated time in minutes since episode start.

    Each tick advances by 8–15 minutes (uniform random), modelling the
    inter-arrival time between customer emails.
    """

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.minutes: float = 0.0
        self.tick_minutes: float = 0.0

    def tick(self) -> float:
        """Advance time by 8–15 simulated minutes. Return new cumulative time."""
        delta = float(self.rng.integers(8, 16))
        self.minutes += delta
        self.tick_minutes = delta
        return self.minutes

    def reset(self) -> None:
        """Reset clock to episode start (t = 0)."""
        self.minutes = 0.0
        self.tick_minutes = 0.0

    @property
    def hour_of_day(self) -> float:
        """Simulated hour of day (0–23.99), assuming episode starts at 09:00."""
        total_hours = 9.0 + (self.minutes / 60.0)
        return total_hours % 24.0

    @property
    def day_of_week(self) -> float:
        """Simulated weekday (0=Mon .. 6=Sun), assuming episode starts on Monday."""
        total_hours = 9.0 + (self.minutes / 60.0)
        return (total_hours // 24.0) % 7.0
