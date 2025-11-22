# backend/security_agent/main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
import os

from .security_agent import SecurityAgent
from .security_agent import allow_outbound_url  # now it exists!

app = FastAPI(title="Security Agent", version="0.2.0")

sec = SecurityAgent()

API_KEYS = set((os.getenv("SEC_API_KEYS") or "orchestrator-key").split(","))

def _require_key(x_api_key: str | None):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

class URLCheckBody(BaseModel):
    urls: List[str]

class PrecheckRequest(BaseModel):
    username: str
    password: str
    message: str

class PostcheckRequest(BaseModel):
    text: str

class PostcheckResponse(BaseModel):
    masked: str
    encrypted: str  # base64 string from Fernet

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/precheck")
def precheck(body: PrecheckRequest):
    if not sec.authenticate_user(body.username, body.password):
        return {"ok": False, "message": "Authentication failed."}

    if not sec.validate_input(body.message):
        return {"ok": False, "message": "Input rejected by security policy."}

    ok, msg = sec.responsible_ai_filter(body.message)
    if not ok:
        return {"ok": False, "message": msg}

    return {"ok": True, "message": body.message}

@app.post("/postcheck", response_model=PostcheckResponse)
def postcheck(body: PostcheckRequest):
    masked = sec.mask_sensitive_data(body.text)
    encrypted = sec.encrypt_data(masked).decode("utf-8")
    return PostcheckResponse(masked=masked, encrypted=encrypted)

@app.post("/check_url")
def check_url(body: URLCheckBody, x_api_key: str | None = Header(default=None)):
    _require_key(x_api_key)
    results = []
    for url in body.urls:
        ok, reason = allow_outbound_url(url)
        results.append({"url": url, "ok": ok, "reason": reason})
    return {"results": results}
