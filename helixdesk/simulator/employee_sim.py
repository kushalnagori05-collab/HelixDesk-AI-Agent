"""EmployeeSimulator — models human employee behaviour for ticket resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TickResolutionEvent:
    """Outcome of a ticket during an employee tick."""
    ticket_id: str
    resolved: bool
    csat_score: int | None  # 1-5, only set if resolved


@dataclass
class _AssignedTicket:
    """Internal record of a ticket assigned to an employee."""
    ticket_id: str
    sla_deadline_minutes: float
    assigned_at: float


@dataclass
class _EmployeeState:
    """Internal state for a single employee."""
    load: int = 0
    total_resolve_time: float = 0.0
    resolve_count: int = 0
    assigned_tickets: list[_AssignedTicket] = field(default_factory=list)

    @property
    def avg_resolve_minutes(self) -> float:
        if self.resolve_count == 0:
            return 0.0
        return self.total_resolve_time / self.resolve_count


class EmployeeSimulator:
    """Simulates a pool of N employees handling ticket queues.

    Each employee has a load (open ticket count), a base resolve rate
    (probability of resolving a ticket per tick), and an overload penalty
    that reduces effectiveness when load exceeds a threshold.
    """

    # CSAT distribution when a ticket IS resolved — skewed positive
    _CSAT_DISTRIBUTION = [3, 4, 4, 4, 5, 5]

    def __init__(self, config: dict, seed: int):
        self.rng = np.random.default_rng(seed)
        emp_cfg = config["employee_sim"]
        sla_cfg = config["sla"]

        self.n_employees: int = config["env"]["n_employees"]
        self.base_resolve_rate: float = emp_cfg["base_resolve_rate"]
        self.overload_penalty: float = emp_cfg["overload_penalty"]
        self.ignore_probability: float = emp_cfg["ignore_probability"]
        self.max_employee_load: int = sla_cfg["max_employee_load"]

        self._employees: list[_EmployeeState] = []
        self.reset()

    def reset(self) -> None:
        """Clear all loads and stats for every employee."""
        self._employees = [_EmployeeState() for _ in range(self.n_employees)]

    def assign(self, employee_idx: int, ticket_id: str, sla_deadline_minutes: float) -> None:
        """Assign a ticket to an employee.

        Args:
            employee_idx: Index of the target employee (0 to n_employees-1).
            ticket_id: Unique ticket identifier.
            sla_deadline_minutes: Absolute simulation time (in minutes) by which
                this ticket must be resolved.

        Raises:
            ValueError: If the employee's load >= max_employee_load.
        """
        emp = self._employees[employee_idx]
        if emp.load >= self.max_employee_load:
            raise ValueError(
                f"Employee {employee_idx} is at max load ({self.max_employee_load}). "
                f"Cannot assign ticket {ticket_id}."
            )
        emp.assigned_tickets.append(
            _AssignedTicket(
                ticket_id=ticket_id,
                sla_deadline_minutes=sla_deadline_minutes,
                assigned_at=0.0,  # will be set by caller if needed
            )
        )
        emp.load += 1

    def tick(self, current_time_minutes: float) -> list[TickResolutionEvent]:
        """Advance one simulation tick — employees may resolve or miss tickets.

        For each assigned ticket:
        - Compute effective resolve rate (reduced if employee is overloaded)
        - Check ignore probability
        - If ticket's SLA deadline has passed → missed (unresolved)
        - Otherwise probabilistically resolve

        Args:
            current_time_minutes: Current simulation clock time.

        Returns:
            List of resolution events (resolved or missed) this tick.
        """
        events: list[TickResolutionEvent] = []

        for emp in self._employees:
            remaining_tickets: list[_AssignedTicket] = []

            # Effective resolve rate
            if emp.load > 7:
                effective_rate = max(0.0, self.base_resolve_rate - self.overload_penalty)
            else:
                effective_rate = self.base_resolve_rate

            for ticket in emp.assigned_tickets:
                # Check if employee ignores this ticket entirely
                if self.rng.random() < self.ignore_probability:
                    remaining_tickets.append(ticket)
                    continue

                overdue = current_time_minutes >= ticket.sla_deadline_minutes

                if overdue:
                    # Missed SLA — ticket is removed, counts as failure
                    events.append(TickResolutionEvent(
                        ticket_id=ticket.ticket_id,
                        resolved=False,
                        csat_score=None,
                    ))
                    emp.load -= 1
                elif self.rng.random() < effective_rate:
                    # Resolved on time
                    resolve_time = current_time_minutes - ticket.assigned_at
                    emp.total_resolve_time += max(resolve_time, 0.0)
                    emp.resolve_count += 1
                    csat = int(self.rng.choice(self._CSAT_DISTRIBUTION))
                    events.append(TickResolutionEvent(
                        ticket_id=ticket.ticket_id,
                        resolved=True,
                        csat_score=csat,
                    ))
                    emp.load -= 1
                else:
                    # Not resolved yet — keep in queue
                    remaining_tickets.append(ticket)

            emp.assigned_tickets = remaining_tickets

        return events

    def get_loads(self) -> list[int]:
        """Return current open ticket count for each employee."""
        return [emp.load for emp in self._employees]

    def get_avg_resolve_times(self) -> list[float]:
        """Return average resolve time (minutes) for each employee."""
        return [emp.avg_resolve_minutes for emp in self._employees]
