"""Simulator unit tests for all simulation components."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from helixdesk.simulator.clock import SimClock
from helixdesk.simulator.email_gen import EmailGenerator, EmailEvent
from helixdesk.simulator.employee_sim import EmployeeSimulator
from helixdesk.simulator.knowledge_base import KnowledgeBase
from helixdesk.simulator.trend_watchdog import TrendWatchdog


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- SimClock tests ---

def test_simclock_tick_advances_monotonically():
    """SimClock.tick() must always increase the time."""
    clock = SimClock(seed=42)
    times = [clock.tick() for _ in range(100)]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], f"Time did not advance at tick {i}"


def test_simclock_tick_within_range():
    """Each tick should advance by 8-15 minutes."""
    clock = SimClock(seed=123)
    for _ in range(100):
        prev = clock.minutes
        clock.tick()
        delta = clock.minutes - prev
        assert 8 <= delta <= 15, f"Tick delta {delta} out of [8, 15] range"


def test_simclock_reset():
    """SimClock.reset() must set time back to 0."""
    clock = SimClock(seed=42)
    clock.tick()
    clock.tick()
    assert clock.minutes > 0
    clock.reset()
    assert clock.minutes == 0.0


# --- EmailGenerator tests ---

def test_email_generator_produces_valid_event(config):
    """EmailGenerator.next() must return a valid EmailEvent."""
    gen = EmailGenerator(config, seed=42)
    email = gen.next(0.0)

    assert isinstance(email, EmailEvent)
    assert email.email_id  # non-empty
    assert email.category in config["email_gen"]["categories"]
    assert email.ticket_type in ("query", "complaint")
    assert 0.0 <= email.sentiment_intensity <= 1.0
    assert email.customer_tier in ("enterprise", "standard", "free")
    assert email.true_priority in ("critical", "high", "medium", "normal")


def test_email_generator_keyword_flag_sentiment(config):
    """Keyword-flagged emails must have sentiment >= 0.85."""
    gen = EmailGenerator(config, seed=42)
    flagged_emails = []
    for _ in range(1000):
        email = gen.next(0.0)
        if email.has_keyword_flag:
            flagged_emails.append(email)

    if flagged_emails:  # May not generate any with this seed
        for email in flagged_emails:
            assert email.sentiment_intensity >= 0.85, (
                f"Keyword-flagged email has sentiment {email.sentiment_intensity} < 0.85"
            )


# --- EmployeeSimulator tests ---

def test_employee_assign_raises_at_max_load(config):
    """EmployeeSimulator.assign() must raise when employee is at max load."""
    emp_sim = EmployeeSimulator(config, seed=42)
    max_load = config["sla"]["max_employee_load"]

    # Fill employee 0 to max
    for i in range(max_load):
        emp_sim.assign(0, f"ticket-{i}", 9999.0)

    with pytest.raises(ValueError):
        emp_sim.assign(0, "ticket-overflow", 9999.0)


def test_employee_loads_after_assignment(config):
    """get_loads() must reflect current assignment counts."""
    emp_sim = EmployeeSimulator(config, seed=42)
    emp_sim.assign(0, "t1", 9999.0)
    emp_sim.assign(0, "t2", 9999.0)
    emp_sim.assign(2, "t3", 9999.0)

    loads = emp_sim.get_loads()
    assert loads[0] == 2
    assert loads[1] == 0
    assert loads[2] == 1


# --- TrendWatchdog tests ---

def test_trend_watchdog_detects_surge(config):
    """TrendWatchdog.tick() must return alert when growth exceeds threshold."""
    watchdog = TrendWatchdog(config)
    window_mins = config["env"]["trend_window_hours"] * 60

    # Create prior window: 2 events
    watchdog.record("billing_dispute", 0.0)
    watchdog.record("billing_dispute", 10.0)

    # Create current window: 10 events (5x growth = 400%)
    current_base = window_mins + 1
    for i in range(10):
        watchdog.record("billing_dispute", current_base + i * 5)

    current_time = current_base + 60
    alerts = watchdog.tick(current_time)
    assert "billing_dispute" in alerts, f"Expected surge alert, got {alerts}"


# --- KnowledgeBase tests ---

def test_kb_lookup_exact_match():
    """KnowledgeBase.lookup() must return similarity 1.0 for exact category match."""
    kb = KnowledgeBase()
    entry, similarity = kb.lookup("login_failure", 0.5)

    assert entry is not None
    assert similarity == 1.0
    assert entry.category == "login_failure"


def test_kb_lookup_no_match():
    """KnowledgeBase.lookup() must return None for unknown categories."""
    kb = KnowledgeBase()
    entry, similarity = kb.lookup("totally_unknown_category", 0.5)

    assert entry is None or similarity < 1.0


def test_kb_add_entry():
    """KnowledgeBase.add_entry() must increase the entry count."""
    kb = KnowledgeBase()
    initial_count = len(kb._entries)
    kb.add_entry("login_failure", ["new keyword"], "New answer text")
    assert len(kb._entries) == initial_count + 1
