"""Gymnasium compatibility test — verifies env passes check_env."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gymnasium.utils.env_checker import check_env
from helixdesk import HelixDeskEnv


def test_gymnasium_compatibility():
    """HelixDeskEnv must pass gymnasium's env_checker with 0 errors."""
    env = HelixDeskEnv()
    check_env(env.unwrapped, warn=True)
    env.close()
