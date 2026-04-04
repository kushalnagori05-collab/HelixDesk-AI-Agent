"""Full environment API tests for HelixDeskEnv."""

import numpy as np
import pytest
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helixdesk import HelixDeskEnv


@pytest.fixture
def env():
    """Create a fresh HelixDeskEnv for each test."""
    e = HelixDeskEnv()
    yield e
    e.close()


def test_reset_returns_correct_shape(env):
    """reset() must return obs of shape (42,) with float32 dtype."""
    obs, info = env.reset()
    assert obs.shape == (42,), f"Expected shape (42,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert "step" in info
    assert info["step"] == 0


def test_step_returns_correct_structure(env):
    """step() must return the 5-tuple with correct types and required info keys."""
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (42,), f"Expected shape (42,), got {obs.shape}"
    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
    assert isinstance(terminated, bool), f"Expected bool terminated, got {type(terminated)}"
    assert isinstance(truncated, bool), f"Expected bool truncated, got {type(truncated)}"
    assert "reward_breakdown" in info
    assert "step" in info
    assert "email_id" in info
    assert "queue_depth" in info
    assert "overdue_count" in info
    assert "trend_alerts_active" in info
    assert "episode_reward_so_far" in info


def test_episode_terminates_at_correct_step(env):
    """Episode must terminate at exactly step == episode_emails (100)."""
    env.reset()
    for i in range(99):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert not terminated, f"Episode terminated early at step {i + 1}"

    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert terminated, "Episode did not terminate at step 100"


def test_state_is_idempotent(env):
    """state() called multiple times must return the same observation."""
    env.reset()
    s1 = env.state()
    s2 = env.state()
    assert np.array_equal(s1, s2), "state() is not idempotent"


def test_reset_resets_sim_clock(env):
    """After stepping and resetting, sim clock should return to t=0."""
    env.reset(seed=42)
    for _ in range(20):
        env.step(env.action_space.sample())

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    # hour_of_day_norm (dim 37) must match after two resets with same seed
    assert np.isclose(obs1[37], obs2[37]), (
        f"hour_of_day_norm differs after reset: {obs1[37]} vs {obs2[37]}"
    )


def test_reward_is_clipped(env):
    """Reward must always be a float in [-1.0, 1.0]."""
    env.reset()
    for _ in range(50):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert -1.0 <= reward <= 1.0, f"Reward {reward} out of [-1, 1] range"


def test_info_has_all_required_keys(env):
    """Every step's info dict must contain all spec-required keys."""
    required_keys = [
        "step", "sim_time_minutes", "email_id", "ticket_type",
        "priority", "assigned_to", "reward_breakdown", "queue_depth",
        "overdue_count", "trend_alerts_active", "csat_score",
        "episode_reward_so_far",
    ]
    env.reset()
    for _ in range(10):
        _, _, _, _, info = env.step(env.action_space.sample())
        for key in required_keys:
            assert key in info, f"Missing required key '{key}' in info dict"


def test_observation_values_in_range(env):
    """All observation values must be in [-1, 1] after reset and steps."""
    obs, _ = env.reset()
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Obs out of range after reset"

    for _ in range(20):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Obs out of range after step"
