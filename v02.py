# file: smtp_mailer.py
import os, time, base64, smtplib, socket
from typing import List, Optional, Tuple, Dict
from email.utils import formataddr, make_msgid
from email.message import EmailMessage

try:
    from msal import ConfidentialClientApplication
except Exception:
    ConfidentialClientApplication = None

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.office365.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

SMTP_USERNAME = os.getenv("SMTP_USERNAME")  # e.g., noreply@s361.onmicrosoft.com
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

OAUTH_ENABLED = os.getenv("SMTP_OAUTH_ENABLED", "true").lower() == "true"
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
OAUTH_SCOPE = "https://outlook.office365.com/.default"

DEFAULT_FROM = os.getenv("MAIL_SENDER", SMTP_USERNAME or "noreply@s361.onmicrosoft.com")
DEFAULT_FROM_NAME = os.getenv("MAIL_SENDER_NAME", "Recruiting Bot")

def _get_oauth2_access_token() -> str:
    if not OAUTH_ENABLED:
        raise RuntimeError("OAuth2 disabled; set SMTP_OAUTH_ENABLED=true.")
    if not ConfidentialClientApplication:
        raise RuntimeError("msal not installed. pip install msal")
    if not (AZURE_TENANT_ID and AZURE_CLIENT_ID and AZURE_CLIENT_SECRET):
        raise RuntimeError("Missing AZURE_TENANT_ID/CLIENT_ID/CLIENT_SECRET")
    app = ConfidentialClientApplication(
        AZURE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
        client_credential=AZURE_CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=[OAUTH_SCOPE])
    if "access_token" not in result:
        raise RuntimeError(f"OAuth token error: {result}")
    return result["access_token"]

def _build_xoauth2(username: str, access_token: str) -> str:
    s = f"user={username}\x01auth=Bearer {access_token}\x01\x01"
    return base64.b64encode(s.encode()).decode()

class SmtpMailer:
    def __init__(
        self,
        server: str = SMTP_SERVER,
        port: int = SMTP_PORT,
        username: Optional[str] = SMTP_USERNAME,
        password: Optional[str] = SMTP_PASSWORD,
        use_oauth2: bool = OAUTH_ENABLED,
        from_email: str = DEFAULT_FROM,
        from_name: str = DEFAULT_FROM_NAME,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ):
        self.server, self.port = server, port
        self.username, self.password = username, password
        self.use_oauth2 = use_oauth2
        self.from_email, self.from_name = from_email, from_name
        self.timeout, self.max_retries, self.backoff_seconds = timeout, max_retries, backoff_seconds
        if self.use_oauth2 and not self.username:
            raise RuntimeError("When using OAuth2, set SMTP_USERNAME to the mailbox address.")

    def _open_and_auth(self) -> smtplib.SMTP:
        client = smtplib.SMTP(self.server, self.port, timeout=self.timeout)
        client.ehlo(); client.starttls(); client.ehlo()
        if self.use_oauth2:
            tok = _get_oauth2_access_token()
            code, resp = client.docmd("AUTH", "XOAUTH2 " + _build_xoauth2(self.username, tok))
            if code != 235:
                try: client.quit()
                finally: ...
                raise RuntimeError(f"XOAUTH2 auth failed: {code} {resp}")
        else:
            if not (self.username and self.password):
                raise RuntimeError("Basic auth requires SMTP_USERNAME and SMTP_PASSWORD.")
            client.login(self.username, self.password)
        return client

    def _retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, socket.timeout, ConnectionError)):
            return True
        if isinstance(exc, smtplib.SMTPResponseException):
            code = getattr(exc, "smtp_code", 0) or getattr(exc, "code", 0)
            return 400 <= code < 500
        return False

    def send(
        self,
        to: List[str],
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[List[str]] = None,
        attachments: Optional[List[Tuple[str, bytes, str]]] = None,  # (filename, bytes, mime)
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        msg = EmailMessage()
        from_email = from_email or self.from_email
        from_name = from_name or self.from_name
        msg["From"] = formataddr((from_name, from_email))
        msg["To"] = ", ".join(to)
        if cc:  msg["Cc"] = ", ".join(cc)
        if reply_to: msg["Reply-To"] = ", ".join(reply_to)
        msg["Subject"] = subject
        msg["Message-ID"] = message_id or make_msgid(domain=from_email.split("@")[-1])
        msg["X-Mailer"] = "InterviewScheduler/SMTP"
        if headers:
            for k, v in headers.items(): msg[k] = str(v)

        if not text_body:
            from html import unescape; import re
            text_body = unescape(re.sub(r"<[^>]+>", "", html_body))
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")

        if attachments:
            for filename, content, mime_type in attachments:
                maintype, subtype = (mime_type.split("/", 1) + ["octet-stream"])[:2]
                msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)

        rcpt = list(to or [])
        if cc: rcpt.extend(cc)
        if bcc: rcpt.extend(bcc)

        attempt, last_exc = 0, None
        while attempt < self.max_retries:
            try:
                with self._open_and_auth() as c:
                    c.sendmail(from_email, rcpt, msg.as_string())
                return msg["Message-ID"]
            except Exception as exc:
                last_exc = exc
                if self._retryable(exc) and attempt < self.max_retries - 1:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    attempt += 1
                    continue
                raise last_exc

# ---- Tool-like helper for candidate availability ----
def send_candidate_availability_email_smtp(
    candidate_email: str,
    candidate_name: str,
    availability_link: str,
    sender_email: Optional[str] = None,
    sender_name: Optional[str] = None,
) -> dict:
    html = f"""
    <div style="font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;font-size:14px;line-height:1.45">
      <p>Hi {candidate_name},</p>
      <p>We’d like to schedule your interview. Please share your available slots here:</p>
      <p><a href="{availability_link}">{availability_link}</a></p>
      <p>Thanks,<br/>Recruiting Team</p>
      <hr/><p style="color:#6b7280">If this wasn’t expected, reply to this email.</p>
    </div>"""
    mailer = SmtpMailer()
    msg_id = mailer.send(
        to=[candidate_email],
        subject="Interview availability request",
        html_body=html,
        reply_to=["recruiting@s361.onmicrosoft.com"],
        from_email=sender_email or DEFAULT_FROM,
        from_name=sender_name or DEFAULT_FROM_NAME,
        headers={"X-Category": "candidate-availability"},
    )
    return {
        "tool_name": "SendAvailabilityRequestSMTP",
        "status": "success",
        "message_id": msg_id,
        "to": candidate_email,
        "from": sender_email or DEFAULT_FROM,
        "candidate_name": candidate_name,
        "availability_link": availability_link,
    }