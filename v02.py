import os, requests, datetime
from msal import ConfidentialClientApplication

TENANT_ID=os.getenv("AZ_TENANT_ID")
CLIENT_ID=os.getenv("AZ_CLIENT_ID")
CLIENT_SECRET=os.getenv("AZ_CLIENT_SECRET")
GRAPH_SCOPE=["https://graph.microsoft.com/.default"]
NOTIFY_URL=os.getenv("GRAPH_NOTIFY_URL")
CLIENT_STATE=os.getenv("GRAPH_CLIENT_STATE","supersecret-state")
USER_ID=os.getenv("GRAPH_USER_ID")

app=ConfidentialClientApplication(CLIENT_ID, authority=f"https://login.microsoftonline.com/{TENANT_ID}", client_credential=CLIENT_SECRET)
tok=app.acquire_token_for_client(scopes=GRAPH_SCOPE)["access_token"]
url="https://graph.microsoft.com/v1.0/subscriptions"
exp=(datetime.datetime.utcnow()+datetime.timedelta(hours=48)).isoformat()+"Z"
body={
  "changeType":"created",
  "notificationUrl": NOTIFY_URL,
  "resource": f"/users/{USER_ID}/messages",
  "expirationDateTime": exp,
  "clientState": CLIENT_STATE
}
r=requests.post(url, json=body, headers={"Authorization":f"Bearer {tok}"}, timeout=30)
print(r.status_code, r.text)