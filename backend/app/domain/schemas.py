"""Pydantic models used for request and response bodies.

These data transfer objects (DTOs) mirror the domain models but add
validation and serialization helpers for the API layer.
"""

from datetime import date, datetime
from typing import List

from pydantic import BaseModel, EmailStr


class Org(BaseModel):
    """Organization data.

    Example:
        >>> Org(id="org_1", name="Acme Corp")
    """

    id: str
    name: str

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {"example": {"id": "org_1", "name": "Acme Corp"}}


class User(BaseModel):
    """System user belonging to an organisation.

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
    email: EmailStr
    name: str

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "user_1",
                "org_id": "org_1",
                "email": "a@example.com",
                "name": "Alice",
            }
        }


class Candidate(BaseModel):
    """Job candidate details.

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
    email: EmailStr
    name: str

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "cand_1",
                "org_id": "org_1",
                "email": "c@example.com",
                "name": "Carol",
            }
        }


class Panelist(BaseModel):
    """Interviewer participating in a panel.

    Example:
        >>> Panelist(id="pan_1", user_id="user_1", expertise="Python")
    """

    id: str
    user_id: str
    expertise: str | None = None

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "pan_1",
                "user_id": "user_1",
                "expertise": "Python",
            }
        }


class Interview(BaseModel):
    """Interview information for scheduling.

    Example:
        >>> Interview(
        ...     id="int_1",
        ...     candidate_id="cand_1",
        ...     panelist_ids=["pan_1"],
        ...     scheduled_at=datetime(2024, 1, 1, 9, 0),
        ... )
    """

    id: str
    candidate_id: str
    panelist_ids: List[str]
    scheduled_at: datetime

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "int_1",
                "candidate_id": "cand_1",
                "panelist_ids": ["pan_1"],
                "scheduled_at": "2024-01-01T09:00:00",
            }
        }


class Holiday(BaseModel):
    """Holiday definition used for blackout dates.

    Example:
        >>> Holiday(date=date(2024, 1, 1), description="New Year")
    """

    date: date
    description: str

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "date": "2024-01-01",
                "description": "New Year",
            }
        }


class CalendarAccount(BaseModel):
    """Calendar account associated with a user.

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
    email: EmailStr

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "acc_1",
                "user_id": "user_1",
                "provider": "google",
                "email": "a@example.com",
            }
        }


class CalendarEvent(BaseModel):
    """Event on a calendar.

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

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "evt_1",
                "account_id": "acc_1",
                "start": "2024-01-01T09:00:00",
                "end": "2024-01-01T10:00:00",
                "summary": "Interview",
            }
        }


class Message(BaseModel):
    """Message exchanged in the system.

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

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "id": "msg_1",
                "sender_id": "user_1",
                "recipient_id": "cand_1",
                "body": "Welcome!",
                "sent_at": "2024-01-01T08:00:00",
            }
        }


class RunMeta(BaseModel):
    """Metadata about execution runs.

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

    class Config:
        allow_mutation = False
        frozen = True
        json_schema_extra = {
            "example": {
                "run_id": "run_1",
                "created_at": "2024-01-01T07:00:00",
                "status": "ok",
            }
        }
