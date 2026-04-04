"""Simulator subpackage — exports all simulator components."""

from helixdesk.simulator.clock import SimClock
from helixdesk.simulator.email_gen import EmailEvent, EmailGenerator
from helixdesk.simulator.employee_sim import EmployeeSimulator, TickResolutionEvent
from helixdesk.simulator.knowledge_base import KBEntry, KnowledgeBase
from helixdesk.simulator.trend_watchdog import TrendWatchdog

__all__ = [
    "SimClock",
    "EmailEvent",
    "EmailGenerator",
    "EmployeeSimulator",
    "TickResolutionEvent",
    "KBEntry",
    "KnowledgeBase",
    "TrendWatchdog",
]
