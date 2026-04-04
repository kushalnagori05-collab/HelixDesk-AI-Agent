"""Reward function unit tests."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from helixdesk.rewards import RewardFunction
from helixdesk.simulator.email_gen import EmailEvent
from helixdesk.simulator.employee_sim import TickResolutionEvent


@pytest.fixture
def config():
    """Load config from yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def reward_fn(config):
    return RewardFunction(config)


def _make_email(
    ticket_type="complaint",
    has_keyword_flag=False,
    sentiment=0.7,
    category="billing_dispute",
    customer_tier="standard",
) -> EmailEvent:
    return EmailEvent(
        email_id="test-001",
        sender_email="test@example.com",
        category=category,
        ticket_type=ticket_type,
        body_text="Test email body",
        sentiment_intensity=sentiment,
        has_keyword_flag=has_keyword_flag,
        customer_tier=customer_tier,
        true_priority="medium",
        created_at_minutes=0.0,
    )


def test_keyword_flag_non_complaint_gives_negative(reward_fn):
    """Keyword-flagged email classified as non-complaint should yield negative reward."""
    email = _make_email(ticket_type="complaint", has_keyword_flag=True)
    action = np.array([0, 3, 5, 2])  # classify_as_query

    total, events = reward_fn.compute(
        action=action, email=email, resolution_events=[],
        trend_alerts=[], queue_state={}, kb_updated=False,
        employee_loads=[1, 1, 1, 1, 1], prev_employee_loads=[1, 1, 1, 1, 1],
    )

    event_types = [e.event_type for e in events]
    assert "keyword_flag_missed" in event_types
    assert total < 0, f"Expected negative reward, got {total}"


def test_on_time_resolution_gives_positive(reward_fn):
    """On-time ticket resolution should produce +1.0 resolve_on_time component."""
    email = _make_email()
    action = np.array([1, 2, 0, 2])  # classify_as_complaint
    resolution = [TickResolutionEvent(ticket_id="t-001", resolved=True, csat_score=4)]

    total, events = reward_fn.compute(
        action=action, email=email, resolution_events=resolution,
        trend_alerts=[], queue_state={}, kb_updated=False,
        employee_loads=[1, 1, 1, 1, 1], prev_employee_loads=[1, 1, 1, 1, 1],
    )

    event_types = [e.event_type for e in events]
    assert "resolve_on_time" in event_types
    resolve_event = next(e for e in events if e.event_type == "resolve_on_time")
    assert resolve_event.value == 1.0


def test_total_reward_always_clipped(reward_fn):
    """Total reward must always be in [-1.0, 1.0]."""
    email = _make_email(has_keyword_flag=True)

    # Action that triggers many penalties
    action = np.array([0, 3, 5, 2])
    resolutions = [
        TickResolutionEvent(ticket_id="t-001", resolved=False, csat_score=None),
        TickResolutionEvent(ticket_id="t-002", resolved=False, csat_score=None),
        TickResolutionEvent(ticket_id="t-003", resolved=False, csat_score=None),
    ]

    total, events = reward_fn.compute(
        action=action, email=email, resolution_events=resolutions,
        trend_alerts=[], queue_state={}, kb_updated=False,
        employee_loads=[1, 1, 1, 1, 1], prev_employee_loads=[1, 1, 1, 1, 1],
    )

    assert -1.0 <= total <= 1.0, f"Total reward {total} not clipped to [-1, 1]"


def test_resolve_and_csat_dont_double_count(reward_fn):
    """resolve_on_time and csat_high should be separate events for the same ticket."""
    email = _make_email()
    action = np.array([1, 2, 0, 2])
    resolution = [TickResolutionEvent(ticket_id="t-001", resolved=True, csat_score=5)]

    _, events = reward_fn.compute(
        action=action, email=email, resolution_events=resolution,
        trend_alerts=[], queue_state={}, kb_updated=False,
        employee_loads=[1, 1, 1, 1, 1], prev_employee_loads=[1, 1, 1, 1, 1],
    )

    event_types = [e.event_type for e in events]
    assert "resolve_on_time" in event_types
    assert "csat_high" in event_types
    # They should be different events
    resolve_events = [e for e in events if e.event_type == "resolve_on_time"]
    csat_events = [e for e in events if e.event_type == "csat_high"]
    assert len(resolve_events) == 1
    assert len(csat_events) == 1


def test_reward_breakdown_keys_match_event_types(reward_fn):
    """All event_type strings should be valid reward signal names."""
    valid_types = {
        "correct_classification", "misclassification", "keyword_flag_missed",
        "keyword_not_critical", "resolve_on_time", "csat_high", "bad_autoreply",
        "missed_deadline", "trend_prevented", "balanced_assignment",
        "kb_updated", "unnecessary_escalation",
    }

    email = _make_email()
    action = np.array([1, 2, 0, 2])
    resolution = [TickResolutionEvent(ticket_id="t-001", resolved=True, csat_score=4)]

    _, events = reward_fn.compute(
        action=action, email=email, resolution_events=resolution,
        trend_alerts=[], queue_state={}, kb_updated=True,
        employee_loads=[1, 2, 3, 1, 1], prev_employee_loads=[1, 1, 1, 4, 4],
    )

    for event in events:
        assert event.event_type in valid_types, (
            f"Unknown event_type '{event.event_type}'. Valid: {valid_types}"
        )
