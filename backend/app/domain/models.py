"""Core domain entities represented as immutable dataclasses.

Each model is intentionally lightweight and independent of any
persistence concerns. They can be freely used across the application
without risk of mutation.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Tuple


@dataclass(frozen=True)
class Org:
    """Organization within which interviews are scheduled.

    Example:
        >>> Org(id="org_1", name="Acme Corp")
    """

    id: str
    name: str


@dataclass(frozen=True)
class User:
    """Represents an employee or system user.

    Example:
        >>> User(
        ...     id="user_1",
        ...     org_id="org_1",
        ...     email="a@example.com",
        ...     name="Alice",
        ... )
    """

    id: str
    org_id: str
    email: str
    name: str


@dataclass(frozen=True)
class Candidate:
    """Job candidate who is being interviewed.

    Example:
        >>> Candidate(
        ...     id="cand_1",
        ...     org_id="org_1",
        ...     email="c@example.com",
        ...     name="Carol",
        ... )
    """

    id: str
    org_id: str
    email: str
    name: str


@dataclass(frozen=True)
class Panelist:
    """Interviewer participating in an interview panel.

    Example:
        >>> Panelist(id="pan_1", user_id="user_1", expertise="Python")
    """

    id: str
    user_id: str
    expertise: str | None = None


@dataclass(frozen=True)
class Interview:
    """An interview scheduled between a candidate and panelists.

    Example:
        >>> Interview(
        ...     id="int_1",
        ...     candidate_id="cand_1",
        ...     panelist_ids=("pan_1",),
        ...     scheduled_at=datetime(2024, 1, 1, 9, 0),
        ... )
    """

    id: str
    candidate_id: str
    panelist_ids: Tuple[str, ...]
    scheduled_at: datetime


@dataclass(frozen=True)
class Holiday:
    """Calendar holiday on which interviews should not be scheduled.

    Example:
        >>> Holiday(date=date(2024, 1, 1), description="New Year")
    """

    date: date
    description: str


@dataclass(frozen=True)
class CalendarAccount:
    """User calendar account used for scheduling.

    Example:
        >>> CalendarAccount(
        ...     id="acc_1",
        ...     user_id="user_1",
        ...     provider="google",
        ...     email="a@example.com",
        ... )
    """

    id: str
    user_id: str
    provider: str
    email: str


@dataclass(frozen=True)
class CalendarEvent:
    """Calendar event representing a scheduled interview or block.

    Example:
        >>> CalendarEvent(
        ...     id="evt_1",
        ...     account_id="acc_1",
        ...     start=datetime(2024, 1, 1, 9, 0),
        ...     end=datetime(2024, 1, 1, 10, 0),
        ...     summary="Interview",
        ... )
    """

    id: str
    account_id: str
    start: datetime
    end: datetime
    summary: str | None = None


@dataclass(frozen=True)
class Message:
    """Message sent between participants.

    Example:
        >>> Message(
        ...     id="msg_1",
        ...     sender_id="user_1",
        ...     recipient_id="cand_1",
        ...     body="Welcome!",
        ...     sent_at=datetime(2024, 1, 1, 8, 0),
        ... )
    """

    id: str
    sender_id: str
    recipient_id: str
    body: str
    sent_at: datetime


@dataclass(frozen=True)
class RunMeta:
    """Metadata about a service run or job execution.

    Example:
        >>> RunMeta(
        ...     run_id="run_1",
        ...     created_at=datetime(2024, 1, 1, 7, 0),
        ...     status="ok",
        ... )
    """

    run_id: str
    created_at: datetime
    status: str
