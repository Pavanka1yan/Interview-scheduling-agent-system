# agentic_panel_typed.py
# ======================
# POC: Panel identification + 5 slot suggestions, typed & agentic with LangGraph.
# pip install langgraph langchain-openai pydantic pytz

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict, Literal

import pytz
from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# ------------- Constants & Config -------------

UTC = pytz.UTC
TZ_US = pytz.timezone("America/New_York")
TZ_USI = pytz.timezone("Asia/Kolkata")

DEFAULT_WORK_START = 9  # 09:00 local
DEFAULT_WORK_END = 17   # 17:00 local
DEFAULT_DURATION_MIN = 60
DEFAULT_SUGGESTIONS = 5
SLOT_GRID_MIN = 30
HOLD_TTL_SEC = 60 * 60 * 12  # 12 hours tentative hold

# ------------- Types & Models -------------

class Location(str, Enum):
    US = "US"
    USI = "USI"

class ApprovalStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"

class HumanAction(str, Enum):
    approve = "approve"
    reject = "reject"
    revise = "revise"
    question = "question"

class Candidate(BaseModel):
    id: str
    name: str
    email: EmailStr
    location: Location

class Interviewer(BaseModel):
    id: str
    name: str
    email: EmailStr
    location: Location

class Slot(BaseModel):
    """A concrete, 60m-by-default meeting candidate."""
    slot_id: str           # ISO-like key in UTC
    start_utc: datetime
    end_utc: datetime
    interviewer_id: str

    @validator("start_utc", "end_utc", pre=True)
    def _ensure_aware(cls, v):
        if isinstance(v, datetime) and v.tzinfo is not None:
            return v
        # accept "YYYY-MM-DDTHH:MMZ"
        if isinstance(v, str) and v.endswith("Z"):
            return datetime.strptime(v, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        raise ValueError("start_utc/end_utc must be timezone-aware or Z-string")

class Assignment(BaseModel):
    interviewer_id: str
    slot_id: str
    start_utc: str  # Z-string for portability
    end_utc: str

    def to_slot(self) -> Slot:
        return Slot(
            slot_id=self.slot_id,
            start_utc=datetime.strptime(self.start_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC),
            end_utc=datetime.strptime(self.end_utc, "%Y-%m-%dT%H:%MMZ").replace(tzinfo=UTC)
            if False else datetime.strptime(self.end_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC),
            interviewer_id=self.interviewer_id,
        )

class Constraints(BaseModel):
    need: int = Field(DEFAULT_SUGGESTIONS, ge=1, le=10)
    duration: int = Field(DEFAULT_DURATION_MIN, ge=30, le=180)
    days: int = Field(5, ge=1, le=14)
    prefer_interviewer_ids: Optional[List[str]] = None
    avoid_interviewer_ids: Optional[List[str]] = None
    prefer_locations: Optional[List[Location]] = None
    time_hints: Optional[str] = Field(
        None,
        description="e.g. 'mornings IST', 'after 3pm US', 'avoid late evenings USI'",
    )

class Plan(BaseModel):
    candidate_id: str
    suggestions: List[Assignment]
    justification: str

    @root_validator
    def _validate_len(cls, values):
        # allow fewer than 'need' but never 0
        suggestions: List[Assignment] = values.get("suggestions") or []
        if len(suggestions) == 0:
            raise ValueError("Plan has no suggestions.")
        return values

class HumanTriage(BaseModel):
    action: HumanAction
    notes: str = ""
    prefer_locations: Optional[List[Location]] = None
    avoid_interviewer_ids: Optional[List[str]] = None
    prefer_interviewer_ids: Optional[List[str]] = None
    time_hints: Optional[str] = None

# LangGraph state is best as a TypedDict (keeps reducers like add_messages)
class GraphState(TypedDict, total=False):
    candidate_id: str
    candidate: Candidate
    constraints: Constraints
    messages: Annotated[List, add_messages]
    plan: Plan
    approval_status: ApprovalStatus
    human_message: str

# ------------- Mock “DBs” -------------

INTERVIEWERS: List[Interviewer] = [
    Interviewer(id="i1", name="Asha",   email="interviewer1@example.com", location=Location.US),
    Interviewer(id="i2", name="Bharat", email="interviewer2@example.com", location=Location.USI),
    Interviewer(id="i3", name="Chloe",  email="interviewer3@example.com", location=Location.USI),
    Interviewer(id="i4", name="Diego",  email="interviewer4@example.com", location=Location.US),
]

CANDIDATES: List[Candidate] = [
    Candidate(id="c1", name="Alice", email="alice@ex.com", location=Location.US),
    Candidate(id="c2", name="Bala",  email="bala@ex.com",  location=Location.USI),
]

# Holiday calendars (YYYY-MM-DD)
HOLIDAYS: Dict[Location, set[str]] = {
    Location.US: {"2025-11-27", "2025-12-25"},
    Location.USI: {"2025-10-02", "2025-10-31"},
}

# Tentative holds: (interviewer_id, slot_id) -> expires_at_epoch
HOLDS: Dict[Tuple[str, str], float] = {}

# ------------- LLMs -------------

AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
LLM = AzureChatOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_deployment=AZURE_DEPLOYMENT,
    temperature=0,
)

# ------------- Utilities -------------

def tz_for(loc: Location):
    return TZ_US if loc == Location.US else TZ_USI

def day_key(dt: datetime, loc: Location) -> str:
    return dt.astimezone(tz_for(loc)).strftime("%Y-%m-%d")

def is_holiday(loc: Location, dt: datetime) -> bool:
    return day_key(dt, loc) in HOLIDAYS.get(loc, set())

def slot_key(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")

def to_local_str(dt: datetime, loc: Location) -> str:
    return dt.astimezone(tz_for(loc)).strftime("%Y-%m-%d %H:%M (%Z)")

def within_local_9_5(dt: datetime, loc: Location) -> bool:
    local = dt.astimezone(tz_for(loc))
    return DEFAULT_WORK_START <= local.hour < DEFAULT_WORK_END or (local.hour == DEFAULT_WORK_END and local.minute == 0)

def cleanup_expired_holds(now: Optional[float] = None) -> None:
    """Remove expired tentative holds."""
    now = now or time.time()
    expired = [k for k, exp in HOLDS.items() if exp <= now]
    for k in expired:
        del HOLDS[k]

def place_hold(interviewer_id: str, slot_id_str: str) -> None:
    HOLDS[(interviewer_id, slot_id_str)] = time.time() + HOLD_TTL_SEC

def release_hold(interviewer_id: str, slot_id_str: str) -> None:
    HOLDS.pop((interviewer_id, slot_id_str), None)

def is_held(interviewer_id: str, slot_id_str: str) -> bool:
    cleanup_expired_holds()
    return (interviewer_id, slot_id_str) in HOLDS

def parse_time_hint(hint: Optional[str], loc: Optional[Location]) -> Tuple[Optional[int], Optional[int]]:
    """
    Very simple hint parser for 'mornings/afternoons/evenings' and 'after/before H pm/am'.
    Returns (start_hour, end_hour) in the indicated location (if provided), otherwise None.
    """
    if not hint:
        return None, None
    s = hint.lower()
    # coarse windows
    if "morning" in s:
        return 9, 12
    if "afternoon" in s:
        return 13, 17
    if "evening" in s:
        return 16, 18
    # after/before
    import re
    after = re.search(r"after\s+(\d{1,2})\s*(am|pm)?", s)
    before = re.search(r"before\s+(\d{1,2})\s*(am|pm)?", s)
    def to24(h, ap):
        h = int(h)
        if ap == "pm" and h < 12:
            return h + 12
        if ap == "am" and h == 12:
            return 0
        return h
    start = to24(after.group(1), (after.group(2) or "").lower()) if after else None
    end = to24(before.group(1), (before.group(2) or "").lower()) if before else None
    return start, end

def honor_time_hint(dt_start: datetime, dt_end: datetime, loc: Location, hint: Optional[str]) -> bool:
    if not hint:
        return True
    sh, eh = parse_time_hint(hint, loc)
    if sh is None and eh is None:
        return True
    local_start = dt_start.astimezone(tz_for(loc))
    local_end = dt_end.astimezone(tz_for(loc))
    ok_start = (sh is None) or (local_start.hour >= sh)
    ok_end = (eh is None) or (local_end.hour <= eh)
    return ok_start and ok_end

def ensure_overlap_both_sides(start_utc: datetime, end_utc: datetime, cand_loc: Location, iv_loc: Location) -> bool:
    """Both parties must be within local 9–5 for the *entire* slot."""
    return (
        within_local_9_5(start_utc, cand_loc)
        and within_local_9_5(end_utc, cand_loc)
        and within_local_9_5(start_utc, iv_loc)
        and within_local_9_5(end_utc, iv_loc)
    )

def gen_grid_starts(base_local: datetime, work_end_hour: int, duration_min: int) -> List[datetime]:
    """30-min grid within 9–5 local."""
    starts: List[datetime] = []
    end_local = base_local.replace(hour=work_end_hour, minute=0, second=0, microsecond=0)
    cur = base_local
    while cur + timedelta(minutes=duration_min) <= end_local:
        starts.append(cur)
        cur += timedelta(minutes=SLOT_GRID_MIN)
    return starts

# ------------- LLM Tools -------------

@tool
def list_interviewers() -> List[Interviewer]:
    """Return all interviewers."""
    return [i.dict() for i in INTERVIEWERS]  # LLM-friendly dicts

@tool
def get_candidate(candidate_id: str) -> Candidate:
    """Return a candidate by id."""
    for c in CANDIDATES:
        if c.id == candidate_id:
            return c.dict()
    raise ValueError("candidate not found")

@tool
def gen_slots(interviewer_id: str, for_days: int = 5, per_day: int = 5, duration_min: int = DEFAULT_DURATION_MIN) -> List[Slot]:
    """
    Generate mock availability for an interviewer.
    - Excludes holidays
    - 9–5 local, 30m grid
    - Skips already-held slots
    """
    iv = next((i for i in INTERVIEWERS if i.id == interviewer_id), None)
    if iv is None:
        return []
    out: List[Dict] = []
    now = datetime.now(tz=UTC)
    for d in range(for_days):
        day_local = now.astimezone(tz_for(iv.location)).replace(hour=DEFAULT_WORK_START, minute=0, second=0, microsecond=0) + timedelta(days=d)
        if is_holiday(iv.location, day_local):
            continue
        grid = gen_grid_starts(day_local, DEFAULT_WORK_END, duration_min)
        random.shuffle(grid)
        count = 0
        for start_local in grid:
            start_utc = start_local.astimezone(UTC)
            end_utc = start_utc + timedelta(minutes=duration_min)
            sid = slot_key(start_utc)
            if is_held(iv.id, sid):
                continue
            out.append({
                "slot_id": sid,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "interviewer_id": iv.id
            })
            count += 1
            if count >= per_day:
                break
    # return as dicts for tool schema friendliness
    return out

@tool
def reserve_hold(interviewer_id: str, slot_id_str: str) -> bool:
    """Place a tentative hold (12h TTL)."""
    place_hold(interviewer_id, slot_id_str)
    return True

@tool
def release_hold(interviewer_id: str, slot_id_str: str) -> bool:
    """Release a tentative hold."""
    release_hold(interviewer_id, slot_id_str)
    return True

TOOLS = [list_interviewers, get_candidate, gen_slots, reserve_hold, release_hold]

# ------------- Nodes -------------

def node_prepare(state: GraphState) -> GraphState:
    """Hydrate candidate + baseline constraints."""
    cand = next(c for c in CANDIDATES if c.id == state["candidate_id"])
    constraints = Constraints()  # defaults; may be revised by human
    return {
        "candidate": cand,
        "constraints": constraints,
        "messages": [{
            "role": "system",
            "content": (
                "You are a scheduling orchestrator. Use tools to gather availability and propose exactly N viable options.\n"
                "- Prefer same-location interviewers unless human prefers otherwise.\n"
                "- Verify both candidate and interviewer are within their local 9–5 for the entire slot.\n"
                "- Respect simple time hints (mornings/afternoons/evenings, after/before X).\n"
                "- Place holds on any slots you propose so they are not reused elsewhere.\n"
                "Return a structured Plan when ready."
            )
        }]
    }

def node_decide(state: GraphState) -> GraphState:
    """
    Agentic orchestration (single node for brevity):
    - Score interviewers based on constraints (prefer/avoid, locations)
    - Generate slots, filter by overlap + time hints
    - Reserve holds, assemble Plan
    """
    cand: Candidate = state["candidate"]
    cons: Constraints = state["constraints"]
    needed = cons.need
    duration = cons.duration

    ivs = INTERVIEWERS[:]

    # scoring
    prefer_ids = set(cons.prefer_interviewer_ids or [])
    avoid_ids = set(cons.avoid_interviewer_ids or [])
    prefer_locs = set(cons.prefer_locations or [])

    scored: List[Tuple[float, Interviewer]] = []
    for iv in ivs:
        if iv.id in avoid_ids:
            continue
        score = 0.0
        if iv.id in prefer_ids:
            score += 2.0
        if iv.location == cand.location:
            score += 1.0
        if iv.location in prefer_locs:
            score += 0.5
        scored.append((score, iv))
    scored.sort(key=lambda x: x[0], reverse=True)

    # pool slots
    pool: List[Tuple[Interviewer, Slot]] = []
    for _, iv in scored:
        raw = gen_slots.invoke({"interviewer_id": iv.id, "for_days": cons.days, "per_day": 6, "duration_min": duration})
        # normalize to Slot
        slots = [Slot(**s) for s in raw]
        for sl in slots:
            # overlap both sides
            if not ensure_overlap_both_sides(sl.start_utc, sl.end_utc, cand.location, iv.location):
                continue
            # time hints (apply to candidate side if loc mentioned vaguely)
            if cons.time_hints:
                # If hint mentions USI and candidate in US, still apply to candidate (reviewer can add more semantics later)
                if not honor_time_hint(sl.start_utc, sl.end_utc, cand.location, cons.time_hints):
                    continue
            pool.append((iv, sl))

    # choose earliest viable, dedupe by (interviewer, slot_id)
    chosen: List[Assignment] = []
    seen: set[Tuple[str, str]] = set()
    pool.sort(key=lambda p: p[1].start_utc)
    for iv, sl in pool:
        key = (iv.id, sl.slot_id)
        if key in seen:
            continue
        # reserve hold
        reserve_hold.invoke({"interviewer_id": iv.id, "slot_id_str": sl.slot_id})
        chosen.append(Assignment(
            interviewer_id=iv.id,
            slot_id=sl.slot_id,
            start_utc=sl.start_utc.strftime("%Y-%m-%dT%H:%MZ"),
            end_utc=sl.end_utc.strftime("%Y-%m-%dT%H:%MZ")
        ))
        seen.add(key)
        if len(chosen) >= needed:
            break

    justification_bits = []
    if prefer_ids:
        justification_bits.append(f"prioritized interviewers {sorted(prefer_ids)}")
    if prefer_locs:
        justification_bits.append(f"preferred locations {sorted([l.value for l in prefer_locs])}")
    if cons.time_hints:
        justification_bits.append(f"time hint '{cons.time_hints}'")
    if not justification_bits:
        justification_bits.append("same-location preference & earliest feasible windows")

    if not chosen:
        # Edge case: nothing found
        return {
            "plan": Plan(
                candidate_id=cand.id,
                suggestions=[],
                justification="No overlapping 9–5 windows found with current constraints."
            )
        }

    plan = Plan(
        candidate_id=cand.id,
        suggestions=chosen,
        justification="; ".join(justification_bits)
    )
    return {"plan": plan}

def node_critic(state: GraphState) -> GraphState:
    """
    Lightweight guardrail: ensure we have >=1 suggestion; if fewer than need,
    annotate justification so the reviewer knows why it's fewer.
    """
    cons: Constraints = state["constraints"]
    plan: Plan = state["plan"]

    if len(plan.suggestions) < cons.need:
        # keep plan but note shortage
        plan.justification += f" | Only {len(plan.suggestions)}/{cons.need} suggestions available given holidays/holds/time windows."
        return {"plan": plan}
    return {}

def node_request_review(state: GraphState) -> GraphState:
    """Pause for human review; any message is accepted (free text)."""
    plan: Plan = state["plan"]
    cand: Candidate = state["candidate"]
    print("\n--- REVIEW REQUIRED ---")
    if not plan.suggestions:
        print("No options were found. You can revise with hints like 'prefer USI', 'avoid i3', 'mornings IST', or reject.")
    for idx, a in enumerate(plan.suggestions, 1):
        iv = next(i for i in INTERVIEWERS if i.id == a.interviewer_id)
        s = datetime.strptime(a.start_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        e = datetime.strptime(a.end_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        print(f"{idx}. {iv.name} ({iv.location.value}) | {to_local_str(s, cand.location)} → {to_local_str(e, cand.location)} | {a.slot_id}")
    print("Justification:", plan.justification)
    print("Reply freely: e.g., 'approve', 'reject', or 'avoid i3, prefer USI, mornings IST'.\n")
    raise Interrupt({"kind": "approval", "candidate_id": plan.candidate_id})

def node_triage_human(state: GraphState) -> GraphState:
    """Interpret the human's free-text message into structured actions."""
    msg = state.get("human_message", "") or ""
    triage = LLM.with_structured_output(HumanTriage).invoke([
        {"role": "system", "content": "You classify scheduler review messages."},
        {"role": "user", "content": msg}
    ])

    # Merge constraints if revising
    if triage.action == HumanAction.approve:
        return {"approval_status": ApprovalStatus.approved}

    if triage.action == HumanAction.reject:
        return {"approval_status": ApprovalStatus.rejected}

    if triage.action in (HumanAction.revise, HumanAction.question):
        cons: Constraints = state["constraints"]
        new = cons.dict()
        if triage.prefer_interviewer_ids:
            new["prefer_interviewer_ids"] = triage.prefer_interviewer_ids
        if triage.avoid_interviewer_ids:
            new["avoid_interviewer_ids"] = triage.avoid_interviewer_ids
        if triage.prefer_locations:
            # Convert strings to Location
            new["prefer_locations"] = [Location(x) for x in triage.prefer_locations]
        if triage.time_hints:
            new["time_hints"] = triage.time_hints
        return {"constraints": Constraints(**new), "approval_status": ApprovalStatus.pending}

    # Fallback safe:
    return {"approval_status": ApprovalStatus.pending}

def node_finalize(state: GraphState) -> GraphState:
    """Commit or clean up holds based on approval."""
    status = state.get("approval_status", ApprovalStatus.pending)
    plan: Plan = state["plan"]

    if status != ApprovalStatus.approved:
        # Release holds on rejection
        for a in plan.suggestions:
            release_hold(a.interviewer_id, a.slot_id)
        print("❌ Rejected. Released all tentative holds.")
        return {}

    # “Send” mail to candidate (simulated)
    print("\n✅ Approved. Emailing candidate 5 options (simulated). Holds remain until candidate confirms.\n")
    return {}

def approval_router(state: GraphState) -> ApprovalStatus:
    return state.get("approval_status", ApprovalStatus.pending)

# ------------- Graph Build -------------

def build_graph():
    sg = StateGraph(GraphState)
    sg.add_node("prepare", node_prepare)
    sg.add_node("decide", node_decide)
    sg.add_node("critic", node_critic)
    sg.add_node("request_review", node_request_review)
    sg.add_node("triage_human", node_triage_human)
    sg.add_node("finalize", node_finalize)

    sg.add_edge(START, "prepare")
    sg.add_edge("prepare", "decide")
    sg.add_edge("decide", "critic")
    sg.add_edge("critic", "request_review")
    sg.add_conditional_edges(
        "request_review", approval_router,
        {
            ApprovalStatus.approved: "finalize",
            ApprovalStatus.rejected: END,
            ApprovalStatus.pending: "triage_human",
        }
    )
    sg.add_edge("triage_human", "decide")
    sg.add_edge("finalize", END)

    return sg.compile(checkpointer=MemorySaver())

# ------------- Demo -------------

def demo(candidate_id: str = "c1"):
    graph = build_graph()
    # 1) First run pauses for review (Interrupt)
    try:
        graph.invoke({"candidate_id": candidate_id})
    except Interrupt as intr:
        print("Graph paused:", intr.value)

    # 2) Example resume with a natural-language revision:
    #    Try changing these messages to see behavior:
    #    "approve", "reject", "avoid i3, prefer USI, mornings IST"
    graph.invoke(
        {
            "candidate_id": candidate_id,
            "human_message": "avoid i3, prefer USI, mornings IST",
        },
        resume=True,
    )

    # It will loop, regenerate, and pause again:
    try:
        pass
    except Interrupt:
        pass

    # 3) Approve in a follow-up resume:
    graph.invoke(
        {
            "candidate_id": candidate_id,
            "human_message": "approve",
        },
        resume=True,
    )

if __name__ == "__main__":
    demo("c1")