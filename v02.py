# panel_availability_finder_agent.py
from __future__ import annotations

import os
import json
import re
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from zoneinfo import ZoneInfo

# =============== CONFIG (env-driven, sane defaults) ===============
ENV = os.environ

CAL_TZ = ENV.get("CAL__TIMEZONE", "Asia/Kolkata")
CAL_DAY_START = ENV.get("CAL__DAY_START", "09:00")             # inclusive
CAL_DAY_END = ENV.get("CAL__DAY_END", "18:00")                 # exclusive
CAL_SLOT_MIN = int(ENV.get("CAL__SLOT_MINUTES", "60"))
CAL_LOOKAHEAD_WORK_DAYS = int(ENV.get("CAL__LOOKAHEAD_WORK_DAYS", "5"))
CAL_DISCOURAGE_FRIDAY_AFTER = ENV.get("CAL__DISCOURAGE_FRIDAY_AFTER", "16:00")

GRAPH_TENANT_ID = ENV.get("GRAPH__TENANT_ID")
GRAPH_CLIENT_ID = ENV.get("GRAPH__CLIENT_ID")
GRAPH_CLIENT_SECRET = ENV.get("GRAPH__CLIENT_SECRET")

# Example static registries (replace with your own persistence)
PANELS = [
    {"panel_id": "PANEL-A", "skills": ["Python", "ML"],   "timezone": CAL_TZ},
    {"panel_id": "PANEL-B", "skills": ["Java", "Cloud"],  "timezone": CAL_TZ},
    {"panel_id": "PANEL-C", "skills": ["Python", "Cloud"],"timezone": CAL_TZ},
]
PANELISTS = {
    "PANEL-A": ["panel.a1@org.com", "panel.a2@org.com", "panel.a3@org.com"],
    "PANEL-B": ["panel.b1@org.com", "panel.b2@org.com"],
    "PANEL-C": ["panel.c1@org.com", "panel.c2@org.com", "panel.c3@org.com"],
}

# External globals expected from your host app:
# - MESSAGING_PROVIDER: object with .send_email(to, subject, body, cc=None, bcc=None, headers=None)
# - AVAIL_STORE: dict mapping thread_id -> {"candidate_id":..., "availability": {"slots":[...]}}
try:
    MESSAGING_PROVIDER  # type: ignore
except NameError:
    MESSAGING_PROVIDER = None  # your app should overwrite this
try:
    AVAIL_STORE  # type: ignore
except NameError:
    AVAIL_STORE: Dict[str, Dict] = {}  # your candidate-availability agent should fill this


# =============== CALENDAR PROVIDERS (Graph + Mock) ===============
import requests
from msal import ConfidentialClientApplication

@dataclass
class TimeWindow:
    start: dt.datetime
    end: dt.datetime
    tz: str

@dataclass
class Busy:
    start: dt.datetime
    end: dt.datetime

class GraphCalendar:
    """Microsoft Graph calendar provider using /getSchedule (Application permissions)."""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        self._app = ConfidentialClientApplication(
            client_id,
            authority=f"https://login.microsoftonline.com/{tenant_id}",
            client_credential=client_secret,
        )

    def _token(self) -> str:
        scopes = ["https://graph.microsoft.com/.default"]
        r = self._app.acquire_token_silent(scopes, account=None)
        if not r:
            r = self._app.acquire_token_for_client(scopes=scopes)
        if "access_token" not in r:
            raise RuntimeError(f"MSAL token error: {r}")
        return r["access_token"]

    def get_busy(self, schedules: List[str], win: TimeWindow, interval_minutes: int = 30) -> Dict[str, List[Busy]]:
        """
        Returns BUSY intervals per schedule for [win.start, win.end) in tz win.tz.
        Use /users/{id}/getSchedule if you want to scope to a mailbox other than the app's.
        """
        token = self._token()
        url = "https://graph.microsoft.com/v1.0/me/getSchedule"
        payload = {
            "schedules": schedules,
            "startTime": {"dateTime": win.start.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": win.tz},
            "endTime": {"dateTime": win.end.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": win.tz},
            "availabilityViewInterval": interval_minutes,
        }
        r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("value", [])
        out: Dict[str, List[Busy]] = {}
        for ent in data:
            sid = ent.get("scheduleId")
            items = []
            for it in ent.get("scheduleItems", []):
                if it.get("status", "").lower() != "busy":
                    continue
                s = _parse_graph_dt(it["start"], win.tz)
                e = _parse_graph_dt(it["end"], win.tz)
                items.append(Busy(s, e))
            out[sid] = items
        return out

def _parse_graph_dt(obj: Dict, tz: str) -> dt.datetime:
    # Graph provides either local+timeZone or Z; we trust the timeZone we requested
    z = ZoneInfo(tz)
    # obj example: {"dateTime":"2025-01-15T12:00:00.0000000","timeZone":"Asia/Kolkata"}
    return dt.datetime.fromisoformat(obj["dateTime"].split(".")[0]).replace(tzinfo=z)

class MockCalendar:
    """Deterministic mock for local dev."""
    def get_busy(self, schedules: List[str], win: TimeWindow, interval_minutes: int = 30) -> Dict[str, List[Busy]]:
        z = ZoneInfo(win.tz)
        out: Dict[str, List[Busy]] = {}
        cur = win.start
        while cur.date() <= win.end.date():
            # busy 12:00–13:00 and 15:00–15:30
            for s in schedules:
                day = cur.date()
                out.setdefault(s, [])
                out[s] += [
                    Busy(dt.datetime.combine(day, dt.time(12, 0), tzinfo=z), dt.datetime.combine(day, dt.time(13, 0), tzinfo=z)),
                    Busy(dt.datetime.combine(day, dt.time(15, 0), tzinfo=z), dt.datetime.combine(day, dt.time(15, 30), tzinfo=z)),
                ]
            cur += dt.timedelta(days=1)
        return out

def business_windows(tz: str, day_start: str, day_end: str, start_date: dt.date, work_days: int) -> List[TimeWindow]:
    z = ZoneInfo(tz)
    hs, ms = map(int, day_start.split(":"))
    he, me = map(int, day_end.split(":"))
    out: List[TimeWindow] = []
    d = start_date
    while len(out) < work_days:
        if d.weekday() < 5:  # Mon..Fri
            out.append(TimeWindow(
                start=dt.datetime(d.year, d.month, d.day, hs, ms, tzinfo=z),
                end=dt.datetime(d.year, d.month, d.day, he, me, tzinfo=z),
                tz=tz,
            ))
        d += dt.timedelta(days=1)
    return out

def invert_busy_to_free_slots(busy: List[Busy], win: TimeWindow, slot_min: int) -> List[Dict]:
    """Compute fixed-size free slots within [win.start, win.end)."""
    step = dt.timedelta(minutes=slot_min)
    merged = sorted(busy, key=lambda x: x.start)
    cur = win.start
    ranges: List[Tuple[dt.datetime, dt.datetime]] = []
    for b in merged:
        if b.end <= cur:  # overlap or behind
            continue
        if b.start > cur:
            ranges.append((cur, min(b.start, win.end)))
        cur = max(cur, b.end)
        if cur >= win.end:
            break
    if cur < win.end:
        ranges.append((cur, win.end))

    out = []
    for s, e in ranges:
        t = s
        while t + step <= e:
            out.append({
                "date": t.date().isoformat(),
                "start_local": t.strftime("%H:%M"),
                "end_local": (t + step).strftime("%H:%M"),
                "tz": win.tz,
            })
            t += step
    return out


# =============== TOOLS USED BY THE AGENT ===============
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# Simple event recorder hook—replace with your DB event sink if needed
def _record_event(key: str, event: str, payload: Dict | None = None):
    try:
        print(f"[EVENT] {key} :: {event} :: {json.dumps(payload or {}, ensure_ascii=False)}")
    except Exception:
        pass

# Candidate availability access (expects your earlier agent to have stored it)
@tool
def get_candidate_availability(thread_id: str) -> Dict:
    """Return stored availability for the candidate tied to a thread_id."""
    if thread_id not in AVAIL_STORE:
        raise ValueError("No availability for this thread_id")
    return AVAIL_STORE[thread_id]

@tool
def get_panel_list(candidate_id: str) -> List[Dict]:
    """Return panels configured in the system. (Replace with deterministic lookup if needed.)"""
    return [{"panel_id": p["panel_id"], "skills": p["skills"], "timezone": p.get("timezone", CAL_TZ)} for p in PANELS]

@tool
def get_panelists(panel_id: str) -> List[str]:
    """Return emails of panelists for a given panel."""
    return PANELISTS.get(panel_id, [])

# Calendar provider selection (Graph if creds present, else Mock)
CALENDAR_PROVIDER = (
    GraphCalendar(GRAPH_TENANT_ID, GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET)
    if GRAPH_TENANT_ID and GRAPH_CLIENT_ID and GRAPH_CLIENT_SECRET
    else MockCalendar()
)

def _friday_late_flag(date_iso: str, start_local: str) -> bool:
    y, m, d = map(int, date_iso.split("-"))
    weekday = dt.date(y, m, d).weekday()
    dh, dm = map(int, CAL_DISCOURAGE_FRIDAY_AFTER.split(":"))
    return weekday == 4 and int(start_local[:2]) >= dh  # simple hour check

@tool
def gather_panel_availability(panel_id: str) -> Dict:
    """
    Pull panelists' free slots from Outlook for the next N weekdays respecting work hours.
    Returns: { panel_id, slots: [ {date, start_local, end_local, tz, panelists_available, friday_late} ] }
    """
    wins = business_windows(CAL_TZ, CAL_DAY_START, CAL_DAY_END, dt.date.today(), CAL_LOOKAHEAD_WORK_DAYS)
    members = PANELISTS.get(panel_id, [])
    if not members:
        return {"panel_id": panel_id, "slots": []}

    # Aggregate per user
    per_user: Dict[str, List[Dict]] = {u: [] for u in members}
    for w in wins:
        busy_by_user = CALENDAR_PROVIDER.get_busy(members, w, interval_minutes=30)
        for u in members:
            slots = invert_busy_to_free_slots(busy_by_user.get(u, []), w, CAL_SLOT_MIN)
            per_user[u].extend(slots)

    # Intersect by (date,start,end): count how many panelists share the slot
    counter: Dict[Tuple[str, str, str], int] = {}
    for u, slots in per_user.items():
        for s in slots:
            key = (s["date"], s["start_local"], s["end_local"])
            counter[key] = counter.get(key, 0) + 1

    out = []
    for (d, st, en), cnt in counter.items():
        if cnt < 2:   # require >= 2 panelists free
            continue
        out.append({
            "date": d, "start_local": st, "end_local": en, "tz": CAL_TZ,
            "panelists_available": cnt,
            "friday_late": _friday_late_flag(d, st)
        })

    _record_event(panel_id, "PANEL_SLOTS_COMPUTED", {"count": len(out)})
    return {"panel_id": panel_id, "slots": out}


# =============== CP-SAT RANKING (OR-Tools) ===============
# pip install ortools
from ortools.sat.python import cp_model

def _slot_key(s: Dict) -> Tuple[str, str, str]:
    return (s["date"], s["start_local"], s["end_local"])

def _candidate_overlap_score(candidate_slots: List[Dict], slot: Dict) -> int:
    """+2 if exact overlap, +1 if same date, else 0."""
    key = _slot_key(slot)
    ckeys = {_slot_key(cs) for cs in (candidate_slots or [])}
    if key in ckeys:
        return 2
    if any(cs["date"] == slot["date"] for cs in (candidate_slots or [])):
        return 1
    return 0

def _skill_match_score(panel_skills: List[str], candidate_skills: List[str]) -> int:
    cs = {s.lower() for s in (candidate_skills or [])}
    ps = {s.lower() for s in (panel_skills or [])}
    return 2 * len(cs.intersection(ps))

@tool
def rank_panels_cpsat(candidate_slots: List[Dict], candidate_skills: List[str], panel_slots_map: List[Dict]) -> Dict:
    """
    Optimize panel + slot selection with CP-SAT:
      max sum( panel_score + overlap_score + availability_count - friday_penalty )
      subject to selecting at most one final (panel, slot) pair (but we rank all by individual scores).
    Returns ranking + best.
    """
    # Build a flat list of (panel_id, slot, base_score)
    scored_items = []
    for p in panel_slots_map:
        pid = p["panel_id"]
        pskills = p.get("skills", [])
        base = _skill_match_score(pskills, candidate_skills)
        for s in p.get("slots", []):
            overlap = _candidate_overlap_score(candidate_slots, s)
            avail_bonus = min(s.get("panelists_available", 1) - 1, 3)  # each extra panelist +1 up to +3
            friday_penalty = 1 if s.get("friday_late") else 0
            total = base + overlap + avail_bonus - friday_penalty
            scored_items.append({"panel_id": pid, "slot": s, "score": total})

    # If empty, early out
    if not scored_items:
        return {"ranking": [], "selected": None}

    # Build model to pick one with max score (demonstrates CP-SAT; also returns top-N by score for transparency)
    model = cp_model.CpModel()
    xs = []
    for i, it in enumerate(scored_items):
        v = model.NewBoolVar(f"x_{i}")
        xs.append(v)
    # pick at most one
    model.Add(sum(xs) <= 1)
    # objective
    model.Maximize(sum(int(it["score"]) * xs[i] for i, it in enumerate(scored_items)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    selected = None
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, v in enumerate(xs):
            if solver.BooleanValue(v):
                selected = scored_items[i]
                break

    # Also return a pure-ranked list for explainability
    ranked = sorted(scored_items, key=lambda z: z["score"], reverse=True)
    return {"ranking": ranked, "selected": selected}

# =============== EMAIL TO PANEL (proposal) ===============
@tool
def send_panel_proposal_email(candidate_id: str, panel_id: str, to: List[str], proposed_slots: List[Dict]) -> str:
    """
    Email the panel with slot proposals. Returns a thread id:
      thread-panel-<candidate_id>
    Adds `X-Panel-Thread-Key` header and requires subject marker [PID:<candidate_id>].
    """
    global MESSAGING_PROVIDER
    if MESSAGING_PROVIDER is None:
        raise RuntimeError("Messaging provider not initialized")

    thread_id = f"thread-panel-{candidate_id}"
    subject = f"[PID:{candidate_id}] Interview slot proposals for candidate {candidate_id}"
    bullets = "\n".join([f"- {s['date']} {s['start_local']}-{s['end_local']} {s.get('tz','')}" for s in proposed_slots[:8]])
    body = (
        f"Hi Panel ({panel_id}),\n\n"
        f"Please reply with one of:\n"
        f"- ACCEPT(<index>), REJECT ALL, or PROPOSE <YYYY-MM-DD HH:MM-HH:MM>.\n\n"
        f"Proposed slots:\n{bullets}\n\n"
        f"Thanks."
    )
    to0 = to[0] if to else "panel@org.com"
    cc = to[1:] if len(to) > 1 else []
    MESSAGING_PROVIDER.send_email(
        to=to0, subject=subject, body=body, cc=cc,
        headers={"X-Panel-Thread-Key": thread_id}
    )
    _record_event(candidate_id, "PANEL_PROPOSAL_SENT", {"panel_id": panel_id, "slots": proposed_slots})
    return thread_id


# =============== AGENT (LLM-driven) ===============
LLM = init_chat_model("openai:gpt-4o-mini", temperature=0)

from langgraph.prebuilt import create_react_agent

panel_availability_finder_agent = create_react_agent(
    model=LLM,
    tools=[
        get_candidate_availability,
        get_panel_list,
        gather_panel_availability,
        rank_panels_cpsat,
        get_panelists,
        send_panel_proposal_email,
    ],
    name="panel_availability_finder_agent",
    prompt=(
        "You are the Panel Availability Finder Agent.\n"
        "Goal: For a given candidate, compute a high-quality panel+slot proposal and email the panel, then PAUSE.\n\n"
        "Protocol:\n"
        "1) get_candidate_availability(thread_id) -> {candidate_id, availability}\n"
        "2) get_panel_list(candidate_id)\n"
        "3) For each panel_id in that list:\n"
        "   - gather_panel_availability(panel_id) -> ensure weekdays-only and work hours; require >=2 panelists per slot.\n"
        "4) Prepare panel_slots_map = [{panel_id, skills, slots}] matching the panels, where `slots` are from step 3.\n"
        "5) rank_panels_cpsat(candidate_slots=availability.slots, candidate_skills from availability if present or empty, panel_slots_map)\n"
        "6) Take the `selected` result; if None, fall back to the highest-scored item in `ranking`.\n"
        "7) Use get_panelists(panel_id) for recipients, then send_panel_proposal_email(candidate_id, panel_id, to, proposed_slots=[selected.slot] or top overlaps)\n"
        "8) FINISH with exactly one line: AWAIT_PANEL_REPLY(thread_id=<returned>)\n\n"
        "Notes:\n"
        "- Friday late-afternoon slots are de-scored automatically.\n"
        "- Keep outputs concise—only tool calls, then the final AWAIT line."
    ),
)

# =============== OPTIONAL: helper to adapt panel list -> map with slots ===============
# (If your LLM needs a ready-made map in the prompt, you can call this in the agent steps via a tool.)
@tool
def bind_slots_to_panels(panels: List[Dict], slots_by_id: List[Dict]) -> List[Dict]:
    """
    Merge panel metadata with gathered slots:
    Input: panels = [{panel_id, skills,...}], slots_by_id = [{panel_id, slots:[...]}]
    Output: [{panel_id, skills, slots:[...]}]
    """
    sl_map = {x["panel_id"]: x.get("slots", []) for x in slots_by_id}
    out = []
    for p in panels:
        out.append({"panel_id": p["panel_id"], "skills": p.get("skills", []), "slots": sl_map.get(p["panel_id"], [])})
    return out