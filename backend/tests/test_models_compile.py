"""Smoke tests for domain models and schemas."""

from datetime import date, datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.domain import models, schemas  # noqa: E402


def test_models_and_schemas_compile() -> None:
    """Instantiate domain models and pydantic schemas."""

    org = models.Org(id="org_1", name="Acme")
    user = models.User(
        id="user_1", org_id=org.id, email="u@example.com", name="Alice"
    )
    candidate = models.Candidate(
        id="cand_1", org_id=org.id, email="c@example.com", name="Carol"
    )
    panelist = models.Panelist(id="pan_1", user_id=user.id, expertise="Python")
    interview = models.Interview(
        id="int_1",
        candidate_id=candidate.id,
        panelist_ids=(panelist.id,),
        scheduled_at=datetime(2024, 1, 1, 9, 0),
    )
    holiday = models.Holiday(date=date(2024, 1, 1), description="New Year")
    account = models.CalendarAccount(
        id="acc_1", user_id=user.id, provider="google", email="u@example.com"
    )
    event = models.CalendarEvent(
        id="evt_1",
        account_id=account.id,
        start=datetime(2024, 1, 1, 9, 0),
        end=datetime(2024, 1, 1, 10, 0),
        summary="Interview",
    )
    message = models.Message(
        id="msg_1",
        sender_id=user.id,
        recipient_id=candidate.id,
        body="Welcome!",
        sent_at=datetime(2024, 1, 1, 8, 0),
    )
    run_meta = models.RunMeta(
        run_id="run_1", created_at=datetime(2024, 1, 1, 7, 0), status="ok"
    )

    schema_org = schemas.Org(id="org_1", name="Acme")
    schema_user = schemas.User(
        id="user_1", org_id="org_1", email="u@example.com", name="Alice"
    )
    schema_candidate = schemas.Candidate(
        id="cand_1", org_id="org_1", email="c@example.com", name="Carol"
    )
    schema_panelist = schemas.Panelist(
        id="pan_1", user_id="user_1", expertise="Python"
    )
    schema_interview = schemas.Interview(
        id="int_1",
        candidate_id="cand_1",
        panelist_ids=["pan_1"],
        scheduled_at=datetime(2024, 1, 1, 9, 0),
    )
    schema_holiday = schemas.Holiday(
        date=date(2024, 1, 1), description="New Year"
    )
    schema_account = schemas.CalendarAccount(
        id="acc_1",
        user_id="user_1",
        provider="google",
        email="u@example.com",
    )
    schema_event = schemas.CalendarEvent(
        id="evt_1",
        account_id="acc_1",
        start=datetime(2024, 1, 1, 9, 0),
        end=datetime(2024, 1, 1, 10, 0),
        summary="Interview",
    )
    schema_message = schemas.Message(
        id="msg_1",
        sender_id="user_1",
        recipient_id="cand_1",
        body="Welcome!",
        sent_at=datetime(2024, 1, 1, 8, 0),
    )
    schema_run_meta = schemas.RunMeta(
        run_id="run_1", created_at=datetime(2024, 1, 1, 7, 0), status="ok"
    )

    assert all(
        [
            org,
            user,
            candidate,
            panelist,
            interview,
            holiday,
            account,
            event,
            message,
            run_meta,
            schema_org,
            schema_user,
            schema_candidate,
            schema_panelist,
            schema_interview,
            schema_holiday,
            schema_account,
            schema_event,
            schema_message,
            schema_run_meta,
        ]
    )
