# app_candidate_availability_pg_fastapi.py
"""
Install (recommended minimal set):
  pip install -U "fastapi[standard]" "uvicorn" "sqlalchemy>=2" "asyncpg" \
                 "pydantic>=2" "langgraph" "langchain[openai]" \
                 "langgraph-checkpoint-postgres" "langgraph-checkpoint-sqlite"

Environment:
  export OPENAI_API_KEY=...     # if using OpenAI via init_chat_model
  export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
  # If DATABASE_URL is absent, defaults to localhost postgres; script will still run
  # with SQLite checkpoint fallback for the graph (NOT for your app DB).

Run:
  uvicorn app_candidate_availability_pg_fastapi:app --reload

What this provides:
- Durable LangGraph checkpointing: Postgres if available (with .setup()), else SQLite fallback.
- FastAPI + async startup background poller reading candidates from a SQL view/table.
- Pause after sending an email using interrupt(...). Resume later with POST /resume.
- Free-text availability parsing via LLM structured output (Pydantic v2).
- Explicit persistence of parsed availability via an async tool.
"""

from __future__ import annotations

import os
import re
import json
import uuid
import asyncio
import contextlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

# =========================
# --- Database (App DB) ---
# =========================

DB_URI_APP = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)
engine: AsyncEngine = create_async_engine(DB_URI_APP, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Minimal app schema
DDL = """
-- External candidates source (in prod, point this to your real VIEW or staging table).
CREATE TABLE IF NOT EXISTS candidates_view (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  processed BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Email thread tracking (maps thread_id to candidate).
CREATE TABLE IF NOT EXISTS email_threads (
  thread_id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  email TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Persisted availability (strict JSON snapshot from the LLM parser/tool).
CREATE TABLE IF NOT EXISTS availability (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

# ===============================
# --- LangGraph Checkpointer  ---
# ===============================

# We will compile the graph in startup after constructing the checkpointer.
checkpointer = None
app_graph = None  # set on startup

def build_checkpointer():
    """
    Prefer Postgres saver; fallback to SQLite saver if import/connection fails.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # pip: langgraph-checkpoint-postgres
        # Use the plain (sync) Postgres client string; this saver is separate from SQLAlchemy engine.
        # Strip "+asyncpg" for the saver if present.
        pg_conn = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
        pg_conn = pg_conn.replace("+asyncpg", "")  # saver expects 'postgresql://...'
        saver = PostgresSaver.from_conn_string(pg_conn)
        saver.setup()  # create checkpoint tables if missing
        return saver
    except Exception as e:
        print(f"[checkpointer] Postgres saver unavailable ({e!r}); falling back to SQLite.")
        from langgraph.checkpoint.sqlite import SqliteSaver  # pip: langgraph-checkpoint-sqlite
        os.makedirs(".checkpoints", exist_ok=True)
        saver = SqliteSaver(".checkpoints/langgraph.db")
        saver.setup()
        return saver

# ================
# --- LLM & IO ---
# ================

_llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

class Slot(BaseModel):
    date: str = Field(..., description="ISO date (YYYY-MM-DD) or best-effort date text")
    start_local: str = Field(..., description="HH:MM 24h")
    end_local: str = Field(..., description="HH:MM 24h")
    tz: str = Field("Asia/Kolkata", description="IANA timezone")
    flexible: bool = Field(False, description="True if ranges/alternatives implied")
    notes: str = Field("", description="Nuance from email")

class AvailabilityExtraction(BaseModel):
    slots: List[Slot]
    confidence: float = Field(..., ge=0, le=1)
    normalized_to: str = Field("Asia/Kolkata")
    raw_excerpt: str

# =========================
# --- Tools (async) ---
# =========================

@tool
async def fetch_candidate_details(candidate_id: str) -> dict:
    """Fetch candidate {id, name, email} from candidates_view."""
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("SELECT id, name, email FROM candidates_view WHERE id=:cid").bindparams(cid=candidate_id)
        )).mappings().first()
        if not row:
            raise ValueError(f"Unknown candidate_id: {candidate_id}")
        return {"id": row["id"], "name": row["name"], "email": row["email"]}

@tool
async def send_candidate_email(to: str, subject: str, body: str) -> str:
    """
    Mock: 'send' email and create a thread_id; persist thread mapping for /resume.
    Replace this with Microsoft Graph send + thread mapping in production.
    """
    thread_id = f"mock-thread-{uuid.uuid4().hex[:8]}"
    # Try to extract candidate_id from subject marker [CID:...]
    m = re.search(r"\[CID:(.*?)\]", subject or "")
    candidate_id = m.group(1).strip() if m else "UNKNOWN"
    async with AsyncSessionLocal() as s:
        await s.execute(text(
            "INSERT INTO email_threads(thread_id, candidate_id, email) VALUES (:tid, :cid, :email)"
        ), {"tid": thread_id, "cid": candidate_id, "email": to})
        await s.commit()
    return thread_id

@tool
async def read_inbox(thread_id: str, newer_than_minutes: int = 10080) -> str:
    """
    Mock placeholder. Real build: read latest inbound email body from MS Graph by thread.
    We rely on /resume for now.
    """
    return ""

@tool
async def llm_parse_availability(email_text: str) -> dict:
    """Use LLM structured output to parse free-text availability into strict JSON."""
    parser = _llm.with_structured_output(AvailabilityExtraction)
    prompt = (
        "Extract interview availability from the email text.\n"
        "- Prefer concrete dates; if only weekdays are given, choose the next occurrence.\n"
        "- Return HH:MM 24h times; assume Asia/Kolkata when unspecified.\n"
        "- Mark flexible=True if ranges/alternatives implied.\n"
        "- Include a short raw_excerpt supporting the extraction."
    )
    result: AvailabilityExtraction = parser.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": email_text},
    ])
    return result.model_dump()

@tool
async def persist_availability(candidate_id: str, extraction: dict) -> str:
    """
    Persist the parsed availability (JSONB) for downstream systems to consume.
    """
    doc = {
        "candidate_id": candidate_id,
        "extraction": extraction,
        "saved_at": datetime.utcnow().isoformat()
    }
    async with AsyncSessionLocal() as s:
        await s.execute(text(
            "INSERT INTO availability (id, candidate_id, payload) "
            "VALUES (:id, :cid, :payload::jsonb)"
        ), {"id": f"avail-{uuid.uuid4().hex[:8]}", "cid": candidate_id, "payload": json.dumps(doc)})
        await s.commit()
    return "ok"

# =========================
# --- Agent & Graph ---
# =========================

candidate_tools = [
    fetch_candidate_details,
    send_candidate_email,
    read_inbox,
    llm_parse_availability,
    persist_availability,
]

candidate_agent = create_react_agent(
    model=_llm,
    tools=candidate_tools,
    name="candidate_availability_agent",
    prompt=(
        "You are a candidate-availability agent.\n"
        "Goal: For a given candidate_id, fetch details, email them to request availability, then WAIT.\n"
        "Protocol:\n"
        "1) Call fetch_candidate_details(candidate_id) to get name/email.\n"
        "2) Compose a brief, polite email asking for availability. Include the marker [CID:<candidate_id>] in the SUBJECT.\n"
        "3) Call send_candidate_email(to, subject, body) and capture the returned thread_id.\n"
        "4) Immediately STOP and output exactly: AWAIT_REPLY(thread_id=<the_returned_thread_id>)\n"
        "   (Do not call read_inbox yet.)\n"
        "5) When resumed, you will receive the candidate's reply text in the next user message.\n"
        "   Call llm_parse_availability(email_text) to produce structured slots.\n"
        "6) Finally, call persist_availability(candidate_id=<id>, extraction=<tool_result>) to save.\n"
        "7) End with a one-line confirmation.\n"
        "Be concise and follow the protocol strictly."
    ),
)

def supervisor_node(state: MessagesState):
    # Minimal: delegate to worker on first user message; conditional edges handle the rest.
    return {"messages": []}

def _extract_thread_id_from_text(text: str) -> Optional[str]:
    m = re.search(r"AWAIT_REPLY\(\s*thread_id=([^)]+)\)", text or "")
    return m.group(1).strip() if m else None

def await_reply_node(state: MessagesState):
    """
    Pause the run and wait for real email reply.
    On resume, inject the reply so the agent can parse & persist.
    """
    # Interrupt immediately (per resume semantics, this node restarts; interrupt() returns resume payload).
    payload = interrupt({
        "reason": "waiting_for_candidate_reply",
        "instruction": "POST /resume with {'thread_id': <tid>, 'reply_text': '...'}"
    })
    reply_text = (payload or {}).get("reply_text", "") if isinstance(payload, dict) else str(payload or "")
    injected = {
        "role": "user",
        "content": f"Candidate replied:\n\n{reply_text}\n\nPlease parse availability using llm_parse_availability and then persist_availability."
    }
    return {"messages": [injected]}

# Build graph
graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("candidate_availability_agent", candidate_agent)
graph.add_node("await_reply", await_reply_node)

graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", "candidate_availability_agent")

def _route_after_worker(state: MessagesState):
    last = state["messages"][-1]
    content = last.get("content", "") if isinstance(last, dict) else str(last)
    if isinstance(content, list):
        content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
    return "await_reply" if "AWAIT_REPLY(" in str(content) else "supervisor"

graph.add_conditional_edges(
    "candidate_availability_agent",
    _route_after_worker,
    {"await_reply": "await_reply", "supervisor": "supervisor"},
)

graph.add_edge("await_reply", "candidate_availability_agent")
graph.add_edge("supervisor", END)

# app_graph compiled in startup after checkpointer is created.

# =========================
# --- FastAPI App ---
# =========================

app = FastAPI(title="Candidate Availability Orchestrator (LangGraph + Postgres + FastAPI)")

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))

class StartRunRequest(BaseModel):
    candidate_id: str

class ResumeRequest(BaseModel):
    thread_id: str
    reply_text: str

@app.on_event("startup")
async def on_startup():
    # Create app tables
    async with engine.begin() as conn:
        await conn.execute(text(DDL))

    # Setup graph checkpointer and compile
    global checkpointer, app_graph
    checkpointer = build_checkpointer()
    # Enable durable runs/checkpoints
    app_graph = graph.compile(checkpointer=checkpointer)

    # Start background poller
    app.state.poller_task = asyncio.create_task(background_poller())

@app.on_event("shutdown")
async def on_shutdown():
    task: asyncio.Task = getattr(app.state, "poller_task", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

@app.get("/health")
async def health():
    return {"ok": True, "time_utc": datetime.utcnow().isoformat()}

@app.post("/seed-demo")
async def seed_demo():
    async with AsyncSessionLocal() as s:
        for cid, name, email in [
            ("CAND-001", "Asha Rao", "asha.rao@example.com"),
            ("CAND-002", "Vikram Desai", "vikram.desai@example.com"),
        ]:
            await s.execute(text(
                "INSERT INTO candidates_view(id, name, email, processed) "
                "VALUES (:id, :name, :email, FALSE) "
                "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, email=EXCLUDED.email, processed=FALSE, updated_at=NOW()"
            ), {"id": cid, "name": name, "email": email})
        await s.commit()
    return {"ok": True}

@app.post("/run")
async def run_once(req: StartRunRequest):
    """
    Kick off a run for a single candidate.
    The run will pause after emailing the candidate.
    """
    # Use a deterministic thread key so we can infer it later.
    thread_key = f"thread-{req.candidate_id}"
    cfg = {"configurable": {"thread_id": thread_key}}
    initial = {
        "messages": [{
            "role": "user",
            "content": f"For candidate_id {req.candidate_id}, request availability by email, then wait for the reply."
        }]
    }

    # Stream until the first pause
    async for _ in app_graph.astream(initial, cfg, stream_mode="updates"):
        pass

    # Mark candidate processed (emailed)
    async with AsyncSessionLocal() as s:
        await s.execute(text(
            "UPDATE candidates_view SET processed=TRUE, updated_at=NOW() WHERE id=:cid"
        ), {"cid": req.candidate_id})
        await s.commit()

    # Try to return the newly created thread_id for convenience (optional)
    async with AsyncSessionLocal() as s:
        row = (await s.execute(text(
            "SELECT thread_id FROM email_threads WHERE candidate_id=:cid ORDER BY created_at DESC LIMIT 1"
        ).bindparams(cid=req.candidate_id))).mappings().first()
    thread_id = row["thread_id"] if row else None

    return {"status": "started", "candidate_id": req.candidate_id, "thread_key": thread_key, "thread_id": thread_id}

@app.post("/resume")
async def resume(req: ResumeRequest):
    """
    Continue a paused run with the real candidate reply.
    We derive the LangGraph thread key as 'thread-<candidate_id>' using the stored mapping.
    """
    # Lookup candidate_id for the thread
    async with AsyncSessionLocal() as s:
        row = (await s.execute(text(
            "SELECT candidate_id FROM email_threads WHERE thread_id=:tid"
        ).bindparams(tid=req.thread_id))).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown thread_id")
        candidate_id = row["candidate_id"]

    cfg = {"configurable": {"thread_id": f"thread-{candidate_id}"}}
    resume_cmd = Command(resume={"reply_text": req.reply_text})

    async for _ in app_graph.astream(resume_cmd, cfg, stream_mode="updates"):
        pass

    return {"status": "resumed", "candidate_id": candidate_id}

# =========================
# --- Background Poller ---
# =========================

async def background_poller():
    """
    Periodically scans candidates_view for unprocessed rows and starts runs for each.
    Each run pauses after sending the email. External webhook should POST /resume later.
    """
    while True:
        try:
            # Pull a small batch
            async with AsyncSessionLocal() as s:
                rows = (await s.execute(text(
                    "SELECT id FROM candidates_view WHERE processed=FALSE ORDER BY created_at ASC LIMIT 10"
                ))).mappings().all()

            for r in rows:
                cid = r["id"]
                cfg = {"configurable": {"thread_id": f"thread-{cid}"}}
                initial = {
                    "messages": [{
                        "role": "user",
                        "content": f"For candidate_id {cid}, request availability by email, then wait for the reply."
                    }]
                }
                async for _ in app_graph.astream(initial, cfg, stream_mode="updates"):
                    pass

                async with AsyncSessionLocal() as s:
                    await s.execute(text(
                        "UPDATE candidates_view SET processed=TRUE, updated_at=NOW() WHERE id=:cid"
                    ), {"cid": cid})
                    await s.commit()

        except Exception as e:
            print(f"[poller] error: {e}")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)