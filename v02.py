import os, re, requests
import azure.functions as func
from msal import ConfidentialClientApplication
from html import unescape
import re as _re

TENANT_ID=os.getenv("AZ_TENANT_ID")
CLIENT_ID=os.getenv("AZ_CLIENT_ID")
CLIENT_SECRET=os.getenv("AZ_CLIENT_SECRET")
GRAPH_SCOPE=["https://graph.microsoft.com/.default"]
CLIENT_STATE=os.getenv("GRAPH_CLIENT_STATE","supersecret-state")
ORCH_BASE=os.getenv("ORCH_BASE","http://localhost:8000")
ORCH_RESUME_PATH=os.getenv("ORCH_RESUME_PATH","/api/resume")

def _token():
    app=ConfidentialClientApplication(CLIENT_ID, authority=f"https://login.microsoftonline.com/{TENANT_ID}", client_credential=CLIENT_SECRET)
    r=app.acquire_token_silent(GRAPH_SCOPE, None) or app.acquire_token_for_client(scopes=GRAPH_SCOPE)
    return r["access_token"]

def _get_msg(user_id: str, msg_id: str) -> dict:
    tok=_token(); url=f"https://graph.microsoft.com/v1.0/users/{user_id}/messages/{msg_id}"
    params={"$select":"id,subject,bodyPreview,body,internetMessageHeaders","$expand":"internetMessageHeaders"}
    res=requests.get(url, headers={"Authorization":f"Bearer {tok}"}, params=params, timeout=15); res.raise_for_status()
    return res.json()

def _headers(msg: dict)->dict:
    return {h.get("name","").lower():h.get("value") for h in msg.get("internetMessageHeaders",[])}

def _thread_id(msg: dict, hdrs: dict)->str|None:
    if "x-thread-key" in hdrs: return hdrs["x-thread-key"]
    m=re.search(r"\[CID:(.*?)\]", msg.get("subject",""))
    return f"thread-{m.group(1)}" if m else None

def _clean(body: dict|None, preview: str|None)->str:
    text=""
    if body and body.get("contentType")=="html":
        t=_re.sub(r"(?s)<(script|style).*?>.*?</\1>","", body.get("content",""))
        t=_re.sub(r"(?s)<br\s*/?>","\n", t)
        t=_re.sub(r"(?s)<.*?>","", t)
        text=unescape(t)
    if not text: text=preview or ""
    lines=[]
    for line in text.splitlines():
        if line.strip().startswith(">"): continue
        if "From:" in line and "Sent:" in line: break
        lines.append(line)
    return "\n".join(lines).strip()

def main(req: func.HttpRequest) -> func.HttpResponse:
    vt = req.params.get("validationToken")
    if vt: return func.HttpResponse(body=vt, mimetype="text/plain", status_code=200)

    try: data=req.get_json()
    except: return func.HttpResponse("Invalid JSON", status_code=400)

    for n in data.get("value",[]):
        if n.get("clientState") != CLIENT_STATE: continue
        resource = n.get("resource",""); parts = resource.strip("/").split("/")
        try:
            uid = parts[parts.index("users")+1]
            mid = parts[parts.index("messages")+1]
        except Exception:
            uid=None; mid=n.get("resourceData",{}).get("id")
        if not (uid and mid): continue

        try: msg=_get_msg(uid, mid)
        except Exception: continue

        tid=_thread_id(msg, _headers(msg))
        if not tid: continue

        reply=_clean(msg.get("body"), msg.get("bodyPreview"))
        try:
            requests.post(ORCH_BASE.rstrip("/") + ORCH_RESUME_PATH, json={"thread_id": tid, "reply_text": reply}, timeout=10)
        except Exception: pass

    return func.HttpResponse(status_code=202)