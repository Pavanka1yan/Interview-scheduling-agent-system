# app_candidate_availability_pg_fastapi.py
"""
Run:
  pip install -U "fastapi[standard]" "uvicorn" "sqlalchemy>=2" "asyncpg" \
                 "pydantic>=2" "langgraph" "langchain[openai]"

Env:
  export OPENAI_API_KEY=...              # if using OpenAI
  export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

Start:
  uvicorn app_candidate_availability_pg_fastapi:app --reload

Notes:
- Postgres durable checkpointing: tries PostgresSaver; falls back to SQLite file if not available.
- Background poller: polls an "external view" table for unprocessed candidates and starts a LangGraph run (which will PAUSE after email).
- Resume: POST /resume to continue a paused run with the real email reply content.
- Replace the send/read/mock functions with actual Outlook/Graph integrations later.
"""

from __future__ import annotations
import os
import re
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy import (
    text, String, Integer, DateTime, Boolean
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

# =========================
# --- Database setup ---
# =========================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")
engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Minimal schema (you can migrate this with Alembic later)
DDL = """
-- External source of candidates (you can point this to a real VIEW in prod).
CREATE TABLE IF NOT EXISTS candidates_view (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  processed BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Track email threads (for observability / correlation).
CREATE TABLE IF NOT EXISTS email_threads (
  thread_id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  email TEXT NOT NULL,
  run_thread_key TEXT NOT NULL,  -- LangGraph thread key used as config
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Store parsed availability snapshots (for downstream systems to consume).
CREATE TABLE IF NOT EXISTS availability (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

# =========================
# --- LangGraph checkpointing (durable) ---
# =========================

# Prefer a Postgres checkpointer if available; otherwise fallback to SQLite file.
def make_checkpointer():
    try:
        # Newer LangGraph often exposes a Postgres saver. If your installed version
        # uses a different path, adjust this import accordingly.
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
        return PostgresSaver(DATABASE_URL)
    except Exception:
        # Safe fallback: SQLite file (still durable, just not Postgres).
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
        os.makedirs(".checkpoints", exist_ok=True)
        return SqliteSaver(".checkpoints/langgraph.db")

checkpointer = make_checkpointer()

# =========================
# --- LLM + Tools ---
# =========================

_llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

class Slot(BaseModel):
    date: str = Field(..., description="ISO date (YYYY-MM-DD) or best-effort date")
    start_local: str = Field(..., description="HH:MM 24h")
    end_local: str = Field(..., description="HH:MM 24h")
    tz: str = Field("Asia/Kolkata", description="IANA timezone")
    flexible: bool = Field(False)
    notes: str = Field("", description="Nuance from email")

class AvailabilityExtraction(BaseModel):
    slots: List[Slot]
    confidence: float = Field(..., ge=0, le=1)
    normalized_to: str = Field("Asia/Kolkata")
    raw_excerpt: str

@tool
def fetch_candidate_details(candidate_id: str) -> dict:
    """
    Read candidate details from Postgres (backed by your external view).
    """
    # This tool runs sync; for simplicity we do a blocking call via asyncio.run_until_complete-like pattern.
    # In real builds, prefer async tools or prefetch details into the agent state.
    loop = asyncio.get_event_loop()
    async def _q():
        async with AsyncSessionLocal() as s:
            row = (await s.execute(
                text("SELECT id, name, email FROM candidates_view WHERE id=:cid")
            .bindparams(cid=candidate_id))).mappings().first()
            if not row:
                raise ValueError(f"Unknown candidate_id: {candidate_id}")
            return {"id": row["id"], "name": row["name"], "email": row["email"]}
    return loop.run_until_complete(_q())

@tool
def send_candidate_email(to: str, subject: str, body: str) -> str:
    """
    Mock: 'send' an email and create a thread_id; also persist the thread in DB for tracking.
    Replace this with MS Graph send + thread id mapping.
    """
    thread_id = f"mock-thread-{uuid.uuid4().hex[:8]}"
    run_thread_key = f"run-{uuid.uuid4().hex[:8]}"

    loop = asyncio.get_event_loop()
    async def _ins():
        async with AsyncSessionLocal() as s:
            # We don't yet know candidate_id in this tool; the agent should pass it in subject/body or state.
            # For simplicity, try to extract candidate_id if embedded, else store just email.
            m = re.search(r"\[CID:(.*?)\]", subject or "")
            candidate_id = m.group(1).strip() if m else "UNKNOWN"
            await s.execute(text(
                "INSERT INTO email_threads(thread_id, candidate_id, email, run_thread_key) "
                "VALUES (:tid, :cid, :email, :rtk)"
            ), {"tid": thread_id, "cid": candidate_id, "email": to, "rtk": run_thread_key})
            await s.commit()
    loop.run_until_complete(_ins())
    return thread_id

@tool
def read_inbox(thread_id: str, newer_than_minutes: int = 10080) -> str:
    """
    Mock: Reading inbox is *not* auto-injected now. This is kept for parity.
    Real builds would pull the latest inbound email body from MS Graph by thread.
    """
    return ""  # Intentionally empty; we rely on /resume to provide the real reply body.

@tool
def llm_parse_availability(email_text: str) -> dict:
    """
    Use LLM structured output to parse free-text availability.
    """
    parser = _llm.with_structured_output(AvailabilityExtraction)
    prompt = (
        "Extract interview availability slots from the email text.\n"
        "- Prefer concrete dates; if only weekdays are given, choose the next occurrence.\n"
        "- Return HH:MM 24h times; assume Asia/Kolkata when unspecified.\n"
        "- Flag flexible=True if ranges/alternatives implied.\n"
        "- Include a short raw_excerpt supporting the extraction."
    )
    result: AvailabilityExtraction = parser.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": email_text},
    ])
    return result.model_dump()

candidate_tools = [
    fetch_candidate_details,
    send_candidate_email,
    read_inbox,
    llm_parse_availability,
]

candidate_agent = create_react_agent(
    model=_llm,
    tools=candidate_tools,
    name="candidate_availability_agent",
    prompt=(
        "You are a candidate-availability agent.\n"
        "Goal: For a given candidate_id, fetch details, email them to request availability, then WAIT.\n"
        "Protocol:\n"
        "1) fetch_candidate_details(candidate_id)\n"
        "2) send_candidate_email(to, subject, body)\n"
        "3) Immediately STOP and output exactly: AWAIT_REPLY(thread_id=<the_returned_thread_id>)\n"
        "4) When resumed with the candidate's reply text in the next user message,\n"
        "   call llm_parse_availability(email_text) and summarize the slots.\n"
        "Be concise and always follow the protocol."
    ),
)

def supervisor_node(state: MessagesState):
    # Minimal router; first user message -> worker. If worker said wait, a conditional edge handles it.
    return {"messages": []}

def _extract_thread_id(text: str) -> Optional[str]:
    m = re.search(r"AWAIT_REPLY\(\s*thread_id=([^)]+)\)", text)
    return m.group(1).strip() if m else None

def await_reply_node(state: MessagesState):
    # Look for the most recent assistant message that announced AWAIT_REPLY(...)
    assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
    thread_id = None
    for m in reversed(assistant_msgs):
        txt = m.get("content", "")
        if isinstance(txt, list):
            txt = " ".join([c.get("text", "") for c in txt if isinstance(c, dict)])
        tid = _extract_thread_id(str(txt))
        if tid:
            thread_id = tid
            break

    resume_payload = interrupt({
        "reason": "waiting_for_candidate_reply",
        "thread_id": thread_id,
        "instruction": "POST /resume with {'thread_id': <same>, 'reply_text': '...'}"
    })

    if isinstance(resume_payload, dict):
        reply_text = resume_payload.get("reply_text", "")
    else:
        reply_text = str(resume_payload or "")

    injected = {
        "role": "user",
        "content": f"Candidate replied:\n\n{reply_text}\n\nPlease parse availability using llm_parse_availability."
    }
    return {"messages": [injected]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("candidate_availability_agent", candidate_agent)
graph.add_node("await_reply", await_reply_node)

graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", "candidate_availability_agent")

def route_after_worker(state: MessagesState):
    last = state["messages"][-1]
    txt = last.get("content", "") if isinstance(last, dict) else str(last)
    if isinstance(txt, list):
        txt = " ".join([c.get("text", "") for c in txt if isinstance(c, dict)])
    return "await_reply" if "AWAIT_REPLY(" in str(txt) else "supervisor"

graph.add_conditional_edges(
    "candidate_availability_agent",
    route_after_worker,
    {"await_reply": "await_reply", "supervisor": "supervisor"},
)

graph.add_edge("await_reply", "candidate_availability_agent")
graph.add_edge("supervisor", END)

app_graph = graph.compile(checkpointer=checkpointer)

# =========================
# --- FastAPI app ---
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
    # Create tables if not present
    async with engine.begin() as conn:
        await conn.execute(text(DDL))

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
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.post("/run")
async def run_once(req: StartRunRequest):
    cfg = {"configurable": {"thread_id": f"thread-{req.candidate_id}"}}
    initial = {
        "messages": [{
            "role": "user",
            "content": f"For candidate_id {req.candidate_id}, request availability by email, then wait for reply."
        }]
    }
    # Stream until interrupt (pause)
    events = []
    async for event in app_graph.astream(initial, cfg, stream_mode="updates"):
        events.append(event)

    # Mark candidate processed (emailed)
    async with AsyncSessionLocal() as s:
        await s.execute(text(
            "UPDATE candidates_view SET processed=TRUE, updated_at=NOW() WHERE id=:cid"
        ), {"cid": req.candidate_id})
        await s.commit()

    return {"status": "started", "thread_key": cfg["configurable"]["thread_id"], "events": events[-3:]}

@app.post("/resume")
async def resume(req: ResumeRequest):
    # Find the run thread key for this email thread
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("SELECT run_thread_key, candidate_id FROM email_threads WHERE thread_id=:tid")
            .bindparams(tid=req.thread_id)
        )).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown thread_id")

        thread_key = row["run_thread_key"]
        candidate_id = row["candidate_id"]

    cfg = {"configurable": {"thread_id": thread_key}}
    resume_cmd = Command(resume={"reply_text": req.reply_text})

    events = []
    async for event in app_graph.astream(resume_cmd, cfg, stream_mode="updates"):
        events.append(event)

    # Extract the final structured availability (if the agent produced it)
    # In practice, you'd inspect app_graph state; here we try to find an llm_parse_availability tool output in messages.
    # As a simpler approach, rely on the agent to print a JSON block or tool return; we persist a snapshot too.
    # We'll store the *latest* available availability tool output if present.
    # NOTE: For robust extraction, integrate callbacks or a node to persist explicitly.
    # Here, we just persist the raw assistant last message and trust the agent to summarize with JSON.
    # You can improve this by pushing tool results into state and grabbing them here.
    # For now, persist the last assistant message alongside candidate_id.
    latest = None
    if events:
        # scan last few events for "messages" payloads
        for ev in reversed(events):
            try:
                # ev is often like {'candidate_availability_agent': {'messages': [...]} } or similar
                node = ev.get("candidate_availability_agent") or ev.get("supervisor") or ev.get("await_reply")
                if node and "messages" in node and node["messages"]:
                    latest = node["messages"][-1]
                    break
            except Exception:
                continue

    snapshot = {"assistant_message": latest}
    async with AsyncSessionLocal() as s:
        await s.execute(text(
            "INSERT INTO availability (id, candidate_id, payload) VALUES (:id, :cid, :payload::jsonb)"
        ), {"id": f"avail-{uuid.uuid4().hex[:8]}", "cid": candidate_id, "payload": json.dumps(snapshot)})
        await s.commit()

    return {"status": "resumed", "thread_key": thread_key, "events": events[-5:]}

# =========================
# --- Background Poller ---
# =========================

async def background_poller():
    """
    Periodically checks the 'candidates_view' for records where processed = FALSE,
    and starts a graph run for each one (which will pause after emailing).
    """
    while True:
        try:
            async with AsyncSessionLocal() as s:
                rows = (await s.execute(
                    text("SELECT id FROM candidates_view WHERE processed=FALSE ORDER BY created_at ASC LIMIT 10")
                )).mappings().all()

            for r in rows:
                cid = r["id"]
                cfg = {"configurable": {"thread_id": f"thread-{cid}"}}
                initial = {
                    "messages": [{
                        "role": "user",
                        "content": f"For candidate_id {cid}, request availability by email, then wait for reply."
                    }]
                }
                # Stream until interrupt; swallow the stream in the poller (logs can be added as needed)
                async for _ in app_graph.astream(initial, cfg, stream_mode="updates"):
                    pass

                # mark processed
                async with AsyncSessionLocal() as s:
                    await s.execute(text(
                        "UPDATE candidates_view SET processed=TRUE, updated_at=NOW() WHERE id=:cid"
                    ), {"cid": cid})
                    await s.commit()

        except Exception as e:
            # Log and keep going
            print(f"[poller] error: {e}")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

# =========================
# --- Utilities to seed demo data (optional) ---
# =========================

@app.post("/seed-demo")
async def seed_demo():
    async with AsyncSessionLocal() as s:
        await s.execute(text(
            "INSERT INTO candidates_view(id, name, email, processed) "
            "VALUES (:id, :name, :email, FALSE) "
            "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, email=EXCLUDED.email, processed=FALSE, updated_at=NOW()"
        ), {"id": "CAND-001", "name": "Asha Rao", "email": "asha.rao@example.com"})
        await s.execute(text(
            "INSERT INTO candidates_view(id, name, email, processed) "
            "VALUES (:id, :name, :email, FALSE) "
            "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, email=EXCLUDED.email, processed=FALSE, updated_at=NOW()"
        ), {"id": "CAND-002", "name": "Vikram Desai", "email": "vikram.desai@example.com"})
        await s.commit()
    return {"ok": True}