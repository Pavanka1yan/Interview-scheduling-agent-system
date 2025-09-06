# agentic_interview_scheduler_pause_resume_onefile.py
# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent interview scheduler POC using LangGraph with REAL pause/resume.
# All external services are mocked. LLMs are optional (used for phrasing).
#
# Flow:
#   AVH → DataHub → Panel → Availability(5x30m) → Interviewer Cards (PAUSE) →
#   Candidate Offer(3) → Candidate Reply (PAUSE) → Zoom + Invites + Notices
#
# Pauses:
#   • wait_interviewers: stops until interviewer responses (approve/suggest/decline)
#   • wait_candidate   : stops until candidate selects a slot
#
# Demo Control:
#   run_until_pause(thread_id="t1")
#   inject_interviewer_event(thread_id="t1", ...)   # one or more
#   resume_until_pause(thread_id="t1")
#   inject_candidate_event(thread_id="t1", slot_id="S-...")
#   resume_to_finish(thread_id="t1")
#
# Requirements:
#   pip install langgraph langchain pydantic tqdm python-dateutil pytz jinja2 openai
#
# Optional LLM:
#   export OPENAI_API_KEY=...                    # or use Azure OpenAI env vars
#   export OPENAI_MODEL=gpt-4o-mini              # optional
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, json, random
from datetime import datetime, timedelta, timezone
from typing import TypedDict, List, Dict, Optional, Literal, Any, Tuple

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

import pytz
from jinja2 import Template

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

OUTBOX_DIR = "outbox"
DB_PATH = "scheduler_state.sqlite"

def ensure_dir(p:str):
    if not os.path.exists(p): os.makedirs(p, exist_ok=True)

def write_outbox(name:str, content:str):
    ensure_dir(OUTBOX_DIR)
    path = os.path.join(OUTBOX_DIR, name)
    with open(path, "w", encoding="utf-8") as f: f.write(content)
    return path

def tz(tz_str:str): return pytz.timezone(tz_str)
def now_utc(): return datetime.now(timezone.utc)
def now_iso(): return now_utc().isoformat()
def now_local(tz_str:str): return datetime.now(tz(tz_str))
def as_tz(dt:datetime, tz_str:str):
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz(tz_str))
def iso_in_tz(dt:datetime, tz_str:str): return as_tz(dt, tz_str).isoformat()

def use_llm() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"))

def llm_model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def llm_chat(system: str, user: str) -> str:
    try:
        if use_llm():
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=llm_model_name(),
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        pass
    # deterministic fallback
    return Template("{{ user }}").render(user=user)

# ─────────────────────────────────────────────────────────────────────────────
# State model
# ─────────────────────────────────────────────────────────────────────────────

class Candidate(TypedDict, total=False):
    id: str
    name: str
    email: str
    capability: str
    sub_capability: str
    role: str
    location: str
    timezone: str
    hiring_manager_email: str
    hiring_manager_name: str

class Panelist(TypedDict, total=False):
    id: str
    name: str
    email: str
    capability: str
    sub_capability: str
    role: str
    timezone: str

class Slot(TypedDict, total=False):
    id: str
    start_iso: str
    end_iso: str
    source: Literal["auto","suggested_by_interviewer","candidate_proposed","rebook"]
    approvals: Dict[str, Literal["approved","declined","no_response"]]

class Meeting(TypedDict, total=False):
    join_url: str
    start_iso: str
    end_iso: str
    platform: Literal["zoom"]
    calendar_event_ids: Dict[str,str]

class GraphConfig(TypedDict, total=False):
    business_hours_start: int
    business_hours_end: int
    minutes_per_slot: int
    days_ahead: int
    needed_slots: int
    approvals_required: int
    allow_alt_panelists: bool

class GraphState(TypedDict, total=False):
    candidate: Candidate
    requested_panel_criteria: Dict[str,str]
    panel: List[Panelist]
    alternate_panel: List[Panelist]

    auto_proposed_slots: List[Slot]
    interviewer_responses: Dict[str, Literal["approved","declined","suggested","no_response"]]
    interviewer_suggested_slots: List[Slot]

    candidate_slots_offered: List[Slot]
    candidate_selection: Optional[Slot]

    meeting: Optional[Meeting]

    # Pause bookkeeping
    waiting_for: Optional[Literal["interviewers","candidate"]]
    events: List[Dict[str,Any]]  # event log (injected externally)

    cfg: GraphConfig
    logs: List[str]
    errors: List[str]

# ─────────────────────────────────────────────────────────────────────────────
# Mocks (DataHub, Panel, Calendar, Email, Zoom, HM)
# ─────────────────────────────────────────────────────────────────────────────

class MockDataHub:
    MOCK_CANDIDATES = [{
        "id":"CAND-001","name":"Aarav Shah","email":"aarav.shah@example.com",
        "capability":"Data","sub_capability":"ML Engineering","role":"Senior Engineer",
        "location":"Hyderabad","timezone":"Asia/Kolkata",
        "hiring_manager_email":"hm.vidya@corp.example","hiring_manager_name":"Vidya N",
    }]
    def fetch_new_candidates(self) -> List[Dict]: return self.MOCK_CANDIDATES.copy()

class MockPanelRepo:
    PANEL = [
        {"id":"P-101","name":"R. Kumar","email":"rkumar@corp.example","capability":"Data","sub_capability":"ML Engineering","role":"Senior Engineer","timezone":"Asia/Kolkata"},
        {"id":"P-102","name":"S. Iyer","email":"siyer@corp.example","capability":"Data","sub_capability":"ML Engineering","role":"Senior Engineer","timezone":"Asia/Kolkata"},
        {"id":"P-103","name":"Ananya Gupta","email":"ananya.gupta@corp.example","capability":"Data","sub_capability":"ML Engineering","role":"Principal Engineer","timezone":"Asia/Kolkata"},
        {"id":"P-104","name":"D. Bose","email":"dbose@corp.example","capability":"Data","sub_capability":"ML Engineering","role":"Engineer","timezone":"Asia/Kolkata"},
        {"id":"P-105","name":"M. Krish","email":"mkrish@corp.example","capability":"Data","sub_capability":"ML Engineering","role":"Senior Engineer","timezone":"Asia/Kolkata"},
    ]
    def find_panel(self, capability:str, sub_capability:str, role:str) -> List[Dict]:
        primary = [p for p in self.PANEL if p["capability"]==capability and p["sub_capability"]==sub_capability and (p["role"]==role or role in ("Senior Engineer","Engineer"))]
        return primary[:3]
    def find_alternates(self, capability:str, sub_capability:str, role:str, exclude_emails:List[str]) -> List[Dict]:
        pool = [p for p in self.PANEL if p["capability"]==capability and p["sub_capability"]==sub_capability and p["email"] not in exclude_emails]
        return [p for p in pool if p not in self.find_panel(capability, sub_capability, role)]

class MockCalendar:
    def __init__(self):
        self._busy: Dict[str, List[Tuple[datetime,datetime,str]]] = {}
    def _daily_busy_blocks(self, day:datetime):  # 11:30–12:30 local busy
        s = day.replace(hour=11,minute=30,second=0,microsecond=0)
        e = day.replace(hour=12,minute=30,second=0,microsecond=0)
        return [(s,e)]
    def _overlaps(self, a_s:datetime,a_e:datetime,b_s:datetime,b_e:datetime)->bool:
        return not (a_e <= b_s or a_s >= b_e)
    def _person_busy(self, email:str): return self._busy.get(email,[])
    def find_group_slots(self, attendee_emails:List[str], duration_min:int, days_ahead:int, slots_needed:int, tz_str:str, business_hours:Tuple[int,int]) -> List[Slot]:
        out: List[Slot]=[]
        cursor = now_local(tz_str).replace(hour=business_hours[0], minute=0, second=0, microsecond=0)
        end_window = now_local(tz_str) + timedelta(days=days_ahead)
        while cursor < end_window and len(out) < slots_needed:
            if cursor.weekday() >= 5:
                cursor = cursor + timedelta(days=1)
                cursor = cursor.replace(hour=business_hours[0], minute=0, second=0, microsecond=0)
                continue
            if not (business_hours[0] <= cursor.hour < business_hours[1]):
                cursor += timedelta(minutes=30); continue
            slot_end = cursor + timedelta(minutes=duration_min)
            ok = True
            for email in attendee_emails:
                for (bstart,bend) in self._daily_busy_blocks(cursor):
                    if self._overlaps(cursor, slot_end, bstart,bend): ok=False; break
                if not ok: break
                for (bstart,bend,_label) in self._person_busy(email):
                    if self._overlaps(cursor, slot_end, bstart,bend): ok=False; break
                if not ok: break
            if ok:
                sid = f"S-{abs(hash((cursor.isoformat(), tuple(sorted(attendee_emails))))) & 0xfffffff}"
                out.append({"id":sid,"start_iso":iso_in_tz(cursor,tz_str),"end_iso":iso_in_tz(slot_end,tz_str),"source":"auto","approvals":{e:"no_response" for e in attendee_emails}})
            cursor += timedelta(minutes=30)
        return out
    def create_calendar_event(self, title:str, start_iso:str, end_iso:str, attendees:List[str]) -> Dict[str,str]:
        s = datetime.fromisoformat(start_iso); e = datetime.fromisoformat(end_iso)
        for a in attendees: self._busy.setdefault(a, []).append((s,e,title))
        return {a: f"cal_evt_{abs(hash((a,start_iso,end_iso))) & 0xfffffff}" for a in attendees}
    def still_free_for_group(self, start_iso:str, end_iso:str, attendees:List[str]) -> bool:
        s = datetime.fromisoformat(start_iso); e = datetime.fromisoformat(end_iso)
        for a in attendees:
            for (bstart,bend,_label) in self._person_busy(a):
                if self._overlaps(s,e,bstart,bend): return False
            for (bstart,bend) in self._daily_busy_blocks(s):
                if self._overlaps(s,e,bstart,bend): return False
        return True

class MockOutlookEmail:
    CARD_BASE = {"type":"AdaptiveCard","version":"1.5","body":[{"type":"TextBlock","text":"{{title}}","weight":"Bolder","size":"Medium"},{"type":"TextBlock","text":"{{message}}","wrap":True},{"type":"TextBlock","text":"Proposed slots:","weight":"Bolder"},{"type":"Container","items":[]}],"actions":[{"type":"Action.Submit","title":"Approve","data":{"action":"approve"}},{"type":"Action.Submit","title":"Suggest another time","data":{"action":"suggest"}},{"type":"Action.Submit","title":"Decline","data":{"action":"decline"}}]}
    def send_interviewer_card(self,to_email:str,title:str,message:str,slots:List[Slot]):
        card = json.loads(json.dumps(self.CARD_BASE))
        card["body"][3]["items"] = [{"type":"TextBlock","text":f"- {s['start_iso']} → {s['end_iso']} (id={s['id']})"} for s in slots]
        payload = {"to":to_email,"subject":title,"card_json":card,"plaintext":f"{message}\n\n"+"\n".join([f"- {s['start_iso']} → {s['end_iso']} (id={s['id']})" for s in slots]),"sent_at":now_iso()}
        base = f"interviewer_{to_email.replace('@','_at_')}.json"
        write_outbox(base, json.dumps(payload, indent=2))
        write_outbox(base.replace(".json",".md"), f"# {title}\n\nTo: {to_email}\n\n{payload['plaintext']}")
    def send_internal_notice(self,to_email:str,subject:str,body:str):
        write_outbox(f"internal_{to_email.replace('@','_at_')}.md", f"# {subject}\n\n{body}")

class MockExternalEmail:
    def send_candidate_slots(self,to_email:str,candidate_name:str,slots:List[Slot],hm_email:str):
        text = llm_chat("You are a recruiting coordinator writing concise scheduling emails.",
                        f"Propose 3 slots to {candidate_name}. Ask them to reply with SELECT <slot-id>.\nSlots:\n"+"\n".join([f"- {s['start_iso']} → {s['end_iso']} (id={s['id']})" for s in slots])+f"\nCC HM: {hm_email}")
        write_outbox(f"candidate_{to_email.replace('@','_at_')}.md", text)
    def send_confirmation(self,to_email:str,meeting_info:Dict,cc:List[str]):
        text = llm_chat("You are a recruiting coordinator sending confirmations.",
                        f"Confirm interview.\nJoin URL: {meeting_info['join_url']}\nTime: {meeting_info['start_iso']} → {meeting_info['end_iso']}\nCC: {', '.join(cc)}")
        write_outbox(f"candidate_confirm_{to_email.replace('@','_at_')}.md", text)

class MockZoom:
    def create_meeting(self,topic:str,start_iso:str,end_iso:str,host_email:str)->Dict:
        return {"join_url":f"https://zoom.example/j/{abs(hash((topic,start_iso)))%10**9}","start_iso":start_iso,"end_iso":end_iso,"platform":"zoom","host":host_email,"created_at":now_iso()}

class HiringManagerEscalator:
    def __init__(self, mail:MockOutlookEmail): self.mail = mail
    def request_input(self, hm_email:str, subject:str, body:str):
        self.mail.send_internal_notice(hm_email, subject, body)

# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────

class IntakeAgent:
    def __init__(self): self.datahub = MockDataHub()
    def act(self, state:GraphState)->GraphState:
        cands = self.datahub.fetch_new_candidates()
        if not cands: state.setdefault("errors",[]).append("No candidates from DataHub"); return state
        cand = cands[0]
        state["candidate"] = cand
        state["requested_panel_criteria"] = {"capability":cand["capability"],"sub_capability":cand["sub_capability"],"role":cand["role"]}
        state.setdefault("logs",[]).append("Intake: candidate loaded")
        return state

class PanelAgent:
    def __init__(self): self.repo = MockPanelRepo()
    def act(self, state:GraphState)->GraphState:
        crit = state.get("requested_panel_criteria",{})
        prim = self.repo.find_panel(crit.get("capability",""), crit.get("sub_capability",""), crit.get("role",""))
        if not prim: state.setdefault("errors",[]).append("No panelists found")
        state["panel"] = prim
        state["alternate_panel"] = self.repo.find_alternates(crit.get("capability",""), crit.get("sub_capability",""), crit.get("role",""), [p["email"] for p in prim])
        state.setdefault("logs",[]).append(f"Panel: primary={len(prim)} alternates={len(state['alternate_panel'])}")
        return state

class ComplianceAgent:
    def act(self, state:GraphState)->GraphState:
        if not state.get("candidate",{}).get("timezone"): state.setdefault("errors",[]).append("Candidate timezone missing")
        if state.get("candidate",{}).get("role") in ("Senior Engineer","Principal Engineer"):
            if len(state.get("panel",[])) < 2: state.setdefault("errors",[]).append("Quorum not met (need ≥2 interviewers)")
        state.setdefault("logs",[]).append("Compliance: OK")
        return state

class AvailabilityAgent:
    def __init__(self, cal:MockCalendar): self.cal=cal
    def act(self, state:GraphState)->GraphState:
        emails = [p["email"] for p in state.get("panel",[])]
        if not emails: state.setdefault("errors",[]).append("No panel emails"); return state
        cfg = state["cfg"]
        slots = self.cal.find_group_slots(emails, cfg["minutes_per_slot"], cfg["days_ahead"], cfg["needed_slots"], state["candidate"]["timezone"], (cfg["business_hours_start"], cfg["business_hours_end"]))
        state["auto_proposed_slots"] = slots
        state.setdefault("logs",[]).append(f"Availability: proposed {len(slots)}")
        return state

class InterviewerCommsAgent:
    def __init__(self, mail:MockOutlookEmail): self.mail=mail
    def act(self, state:GraphState)->GraphState:
        slots = state.get("auto_proposed_slots",[])
        panel = state.get("panel",[])
        msg = llm_chat("You are HR asking for quick approvals.","Approve or suggest a time for the proposed interview slots.")
        for p in panel:
            self.mail.send_interviewer_card(p["email"], "Interview availability confirmation", msg, slots)
        state["waiting_for"] = "interviewers"
        state.setdefault("interviewer_responses", {p["email"]:"no_response" for p in panel})
        state.setdefault("logs",[]).append("InterviewerComms: cards sent; waiting_for=interviewers")
        return state

class CandidateCommsAgent:
    def __init__(self, ext:MockExternalEmail): self.ext = ext
    def act(self, state:GraphState)->GraphState:
        # Merge approvals & suggestions to pick 3 to offer
        approvals_required = state["cfg"]["approvals_required"]
        auto = state.get("auto_proposed_slots", [])
        suggested = state.get("interviewer_suggested_slots", [])
        responses = state.get("interviewer_responses", {})
        def approved_count(s:Slot)->int: return sum(1 for v in s["approvals"].values() if v=="approved")
        # Apply approvals to autos
        for s in auto:
            for email, resp in responses.items():
                if resp=="approved": s["approvals"][email]="approved"
                elif resp=="declined": s["approvals"][email]="declined"
        approved = [s for s in auto if approved_count(s)>=approvals_required]
        ranked = approved + suggested + [s for s in auto if s not in approved]
        chosen = ranked[:3]
        if not chosen:
            state.setdefault("errors",[]).append("No slots to offer candidate")
            return state
        state["candidate_slots_offered"] = chosen
        c = state["candidate"]
        self.ext.send_candidate_slots(c["email"], c["name"], chosen, c["hiring_manager_email"])
        state["waiting_for"] = "candidate"
        state.setdefault("logs",[]).append(f"CandidateComms: offered {len(chosen)}; waiting_for=candidate")
        return state

class SchedulerAgent:
    def __init__(self, cal:MockCalendar, zoom:MockZoom, ext:MockExternalEmail, mail:MockOutlookEmail):
        self.cal, self.zoom, self.ext, self.mail = cal, zoom, ext, mail
    def _escalate(self, state:GraphState, reason:str):
        HiringManagerEscalator(self.mail).request_input(state["candidate"]["hiring_manager_email"], "Scheduling: input needed", reason)
        state.setdefault("logs",[]).append(f"Escalation→HM: {reason}")
    def act(self, state:GraphState)->GraphState:
        sel = state.get("candidate_selection")
        if not sel:
            self._escalate(state, "Candidate has not selected a slot yet; re-offer or nudge if needed.")
            return state
        attendees = [p["email"] for p in state.get("panel",[])] + [state["candidate"]["hiring_manager_email"]]
        if not self.cal.still_free_for_group(sel["start_iso"], sel["end_iso"], attendees):
            state.setdefault("logs",[]).append("Scheduler: selected slot now busy; attempting alternates")
            alts = [s for s in state.get("candidate_slots_offered",[]) if s["id"] != sel["id"]]
            picked = None
            for s in alts:
                if self.cal.still_free_for_group(s["start_iso"], s["end_iso"], attendees):
                    picked = s; break
            if not picked:
                st = datetime.fromisoformat(state["candidate_slots_offered"][0]["start_iso"]) + timedelta(days=1)
                picked = {"id":f"RE-{abs(hash(st.isoformat()))&0xfffffff}","start_iso":st.isoformat(),"end_iso":(st+timedelta(minutes=state['cfg']['minutes_per_slot'])).isoformat(),"source":"rebook","approvals":{a:"approved" for a in attendees}}
                state.setdefault("logs",[]).append("Scheduler: rebooked one day later")
            sel = picked
            state["candidate_selection"] = picked
        meeting = self.zoom.create_meeting(f"Interview: {state['candidate']['name']}", sel["start_iso"], sel["end_iso"], state["panel"][0]["email"])
        cal_ids = self.cal.create_calendar_event("Interview", meeting["start_iso"], meeting["end_iso"], attendees)
        meeting["calendar_event_ids"] = cal_ids
        state["meeting"] = meeting
        self.ext.send_confirmation(state["candidate"]["email"], meeting, cc=attendees)
        for a in attendees:
            self.mail.send_internal_notice(a, "Interview scheduled", f"Zoom: {meeting['join_url']}\nTime: {meeting['start_iso']} → {meeting['end_iso']}")
        state.setdefault("logs",[]).append("Scheduler: meeting scheduled & notices sent")
        return state

# ─────────────────────────────────────────────────────────────────────────────
# Graph with conditional PAUSE points
# ─────────────────────────────────────────────────────────────────────────────

def default_cfg() -> GraphConfig:
    return {"business_hours_start":10,"business_hours_end":18,"minutes_per_slot":30,"days_ahead":5,"needed_slots":5,"approvals_required":2,"allow_alt_panelists":True}

# Build singletons for tool instances
_CAL = MockCalendar()
_MAIL = MockOutlookEmail()
_EXT = MockExternalEmail()
_ZOOM = MockZoom()

_INTK = IntakeAgent()
_PANL = PanelAgent()
_COMP = ComplianceAgent()
_AVAI = AvailabilityAgent(_CAL)
_ICMS = InterviewerCommsAgent(_MAIL)
_CCMS = CandidateCommsAgent(_EXT)
_SCHD = SchedulerAgent(_CAL, _ZOOM, _EXT, _MAIL)

def build_graph():
    builder = StateGraph(GraphState)

    def n_intake(s:GraphState)->GraphState: return _INTK.act(s)
    def n_panel(s:GraphState)->GraphState: return _PANL.act(s)
    def n_compliance(s:GraphState)->GraphState: return _COMP.act(s)
    def n_availability(s:GraphState)->GraphState: return _AVAI.act(s)
    def n_interviewer_comms(s:GraphState)->GraphState: return _ICMS.act(s)

    def n_wait_interviewers(s:GraphState)->GraphState:
        # If approvals or suggestions present in events, apply and continue; else PAUSE
        events = s.get("events", [])
        updated = False
        for ev in events:
            if ev.get("type")=="interviewer_response":
                email = ev.get("email")
                action = ev.get("action")  # approved | declined | suggest
                s.setdefault("interviewer_responses",{}).setdefault(email,"no_response")
                if action=="approve":
                    s["interviewer_responses"][email]="approved"
                    # mark approvals on each auto slot
                    for slot in s.get("auto_proposed_slots",[]):
                        slot["approvals"][email]="approved"
                    updated = True
                elif action=="decline":
                    s["interviewer_responses"][email]="declined"
                    for slot in s.get("auto_proposed_slots",[]):
                        slot["approvals"][email]="declined"
                    updated = True
                elif action=="suggest":
                    s["interviewer_responses"][email]="suggested"
                    tz_str = next((p["timezone"] for p in s.get("panel",[]) if p["email"]==email), s["candidate"]["timezone"])
                    base = now_local(tz_str).replace(hour=14, minute=0, second=0, microsecond=0)+timedelta(days=1)
                    new_slot = {"id":f"SG-{abs(hash((email, base.isoformat())))&0xfffffff}","start_iso":iso_in_tz(base,tz_str),"end_iso":iso_in_tz(base+timedelta(minutes=s['cfg']['minutes_per_slot']),tz_str),"source":"suggested_by_interviewer","approvals":{p["email"]:"no_response" for p in s.get("panel",[])}}
                    s.setdefault("interviewer_suggested_slots",[]).append(new_slot)
                    updated = True
        # clear events consumed
        s["events"] = [e for e in events if e.get("type")!="interviewer_response"]

        # Check if we have enough approvals or at least some proposals
        approvals_required = s["cfg"]["approvals_required"]
        def approved_count(slot:Slot)->int: return sum(1 for v in slot["approvals"].values() if v=="approved")
        enough_approved = any(approved_count(sl)>=approvals_required for sl in s.get("auto_proposed_slots",[]))
        have_suggestions = bool(s.get("interviewer_suggested_slots"))
        if enough_approved or have_suggestions:
            s["waiting_for"] = None
            s.setdefault("logs",[]).append("WaitInterviewers: have approvals/suggestions → continue")
        else:
            s["waiting_for"] = "interviewers"
            s.setdefault("logs",[]).append("WaitInterviewers: still waiting (PAUSE)")
        return s

    def n_candidate_comms(s:GraphState)->GraphState: return _CCMS.act(s)

    def n_wait_candidate(s:GraphState)->GraphState:
        # Look for candidate selection events
        events = s.get("events", [])
        for ev in events:
            if ev.get("type")=="candidate_selection":
                slot_id = ev.get("slot_id")
                # accept only if it was offered
                for sl in s.get("candidate_slots_offered",[]):
                    if sl["id"]==slot_id:
                        s["candidate_selection"] = sl
                        s["waiting_for"] = None
                        s.setdefault("logs",[]).append(f"WaitCandidate: candidate selected {slot_id}")
                        break
        # remove processed events
        s["events"] = [e for e in events if e.get("type")!="candidate_selection"]

        if not s.get("candidate_selection"):
            s["waiting_for"] = "candidate"
            s.setdefault("logs",[]).append("WaitCandidate: still waiting (PAUSE)")
        return s

    def n_scheduler(s:GraphState)->GraphState: return _SCHD.act(s)

    # Pause node: no-op; flow ends here to simulate a pause
    def n_pause(s:GraphState)->GraphState: return s

    # Nodes
    builder.add_node("intake", n_intake)
    builder.add_node("panel_select", n_panel)
    builder.add_node("compliance", n_compliance)
    builder.add_node("find_slots", n_availability)
    builder.add_node("interviewer_comms", n_interviewer_comms)
    builder.add_node("wait_interviewers", n_wait_interviewers)
    builder.add_node("candidate_comms", n_candidate_comms)
    builder.add_node("wait_candidate", n_wait_candidate)
    builder.add_node("scheduler", n_scheduler)
    builder.add_node("pause", n_pause)

    # Edges
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "panel_select")
    builder.add_edge("panel_select", "compliance")
    builder.add_edge("compliance", "find_slots")
    builder.add_edge("find_slots", "interviewer_comms")
    builder.add_edge("interviewer_comms", "wait_interviewers")

    # Conditional: wait_interviewers → (pause | candidate_comms)
    def cond_wait_interviewers(s:GraphState) -> Literal["pause","continue"]:
        return "continue" if s.get("waiting_for") is None else "pause"
    builder.add_conditional_edges("wait_interviewers", cond_wait_interviewers, {"pause":"pause","continue":"candidate_comms"})

    builder.add_edge("candidate_comms", "wait_candidate")

    # Conditional: wait_candidate → (pause | scheduler)
    def cond_wait_candidate(s:GraphState) -> Literal["pause","continue"]:
        return "continue" if s.get("candidate_selection") else "pause"
    builder.add_conditional_edges("wait_candidate", cond_wait_candidate, {"pause":"pause","continue":"scheduler"})

    builder.add_edge("scheduler", END)
    builder.add_edge("pause", END)

    memory = SqliteSaver.from_conn_string(DB_PATH)
    graph = builder.compile(checkpointer=memory)
    return graph

# ─────────────────────────────────────────────────────────────────────────────
# Public API — run, inject events, resume
# ─────────────────────────────────────────────────────────────────────────────

def _invoke(graph, thread_id:str, state:Optional[GraphState]=None) -> GraphState:
    cfg = default_cfg()
    if state is None:
        state = {"cfg":cfg,"logs":[],"errors":[],"events":[]}
    out = graph.invoke(state, config={"configurable":{"thread_id": thread_id}})
    return out

def run_until_pause(thread_id:str="demo-1") -> GraphState:
    """Starts a new scheduling run and stops at the first pause (waiting for interviewers)."""
    ensure_dir(OUTBOX_DIR)
    graph = build_graph()
    s = _invoke(graph, thread_id)
    print("Run→Pause @", s.get("waiting_for"))
    return s

def inject_interviewer_event(thread_id:str, email:str, action:Literal["approve","decline","suggest"]):
    """Inject a single interviewer response event."""
    graph = build_graph()
    # Load current state from checkpoint (graph.invoke with empty dict will load last)
    s = _invoke(graph, thread_id, state={})
    ev = {"type":"interviewer_response","email":email,"action":"approve" if action=="approve" else ("decline" if action=="decline" else "suggest")}
    s.setdefault("events", []).append(ev)
    # Save back and run the wait node to process; we route through wait_interviewers via interviewer_comms edge already added
    # Simplest: directly invoke again; conditional node will decide to pause or continue.
    s = _invoke(graph, thread_id, state=s)
    print("Injected interviewer event:", ev, "| waiting_for:", s.get("waiting_for"))
    return s

def resume_until_pause(thread_id:str):
    """Resume from current pause; will progress until next pause (candidate) or finish if already satisfied."""
    graph = build_graph()
    s = _invoke(graph, thread_id, state={})
    print("Resume→", "Paused @" + str(s.get("waiting_for")) if s.get("waiting_for") else "Continuing")
    return s

def inject_candidate_event(thread_id:str, slot_id:str):
    """Inject candidate selection event for a previously offered slot."""
    graph = build_graph()
    s = _invoke(graph, thread_id, state={})
    ev = {"type":"candidate_selection","slot_id":slot_id}
    s.setdefault("events", []).append(ev)
    s = _invoke(graph, thread_id, state=s)
    print("Injected candidate selection:", slot_id, "| waiting_for:", s.get("waiting_for"))
    return s

def resume_to_finish(thread_id:str):
    """Resume and try to finish scheduling (schedule meeting + notices)."""
    return resume_until_pause(thread_id)

# ─────────────────────────────────────────────────────────────────────────────
# Example demo script (run step-by-step)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Demo: starting run and pausing for interviewer approvals…")
    s = run_until_pause("t1")  # reaches wait_interviewers → pause
    # Inspect offered auto slots (optional)
    autos = s.get("auto_proposed_slots", [])
    print(f"Auto slots proposed: {len(autos)}")
    if autos:
        print("Example slot:", autos[0]["id"], autos[0]["start_iso"], "→", autos[0]["end_iso"])

    # Inject approvals/suggestions from two panelists
    print("\nInjecting interviewer approvals/suggestions…")
    inject_interviewer_event("t1", email="rkumar@corp.example", action="approve")
    inject_interviewer_event("t1", email="siyer@corp.example", action="suggest")

    # Resume — this should proceed past wait_interviewers, send candidate email, then pause at wait_candidate
    print("\nResuming until candidate pause…")
    s2 = resume_until_pause("t1")
    print("Waiting for:", s2.get("waiting_for"))
    offered = s2.get("candidate_slots_offered", [])
    if offered:
        chosen_id = offered[0]["id"]
        print("\nInjecting candidate selection:", chosen_id)
        inject_candidate_event("t1", slot_id=chosen_id)

    # Final resume to finish scheduling
    print("\nFinal resume to finish…")
    s3 = resume_to_finish("t1")
    mtg = s3.get("meeting")
    if mtg:
        print("Scheduled:", mtg["join_url"], mtg["start_iso"], "→", mtg["end_iso"])
    if s3.get("errors"):
        print("Errors:", s3["errors"])
    print("Logs:")
    for line in s3.get("logs", []): print(" -", line)