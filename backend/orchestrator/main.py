# backend/orchestrator/main.py
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import re
import os, json
from typing import Optional, Dict, Any, List
from .llm import summarize_retrieval, synthesize_search, polish_chat
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")
from .config import AGENTS  # must define at least: {"report": "...", "retrieval": "..."} (chat optional)
from orchestrator.routes import news as news_routes

# -------------------------
# App & CORS
# -------------------------
app = FastAPI(title="Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")

REPORT_BASE = AGENTS["report"].rstrip("/")
RETRIEVAL_BASE = AGENTS["retrieval"].rstrip("/")
SEC_BASE = AGENTS.get("security", "").rstrip("/")

SEC_API_KEY = os.getenv("SECURITY_API_KEY", "orchestrator-key")
REPORT_URL = os.getenv("REPORT_URL", "http://127.0.0.1:8003")
RETRIEVAL_URL = os.getenv("RETRIEVAL_AGENT_URL", "http://127.0.0.1:8005")

# Security creds (env-configurable; defaults are fine for demo)
SEC_USER = os.getenv("SEC_USER", "admin")
SEC_PASS = os.getenv("SEC_PASS", "admin")

# -------------------------
# Schemas
# -------------------------
class TextRequest(BaseModel):
    query: str

class RetrievalRequest(BaseModel):
    question: str

class ChatBody(BaseModel):
    message: str
    history: list[dict] = []  # optional, unused right now

class SearchBody(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None  # e.g., {"country":"...", "topic":"..."}

class ReportFormRequest(BaseModel):
    disease: str
    region: str
    date_from: str
    date_to: str    

class SearchBody(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None  # e.g., {"country": "..."}
    mode: Optional[str] = None 

class ForecastBody(BaseModel):
    disease: str         # "dengue" | "covid" | "malaria" | ...
    region: str          # e.g. "Sri Lanka"
    date_from: str       # "YYYY-MM-DD"
    date_to: str         # "YYYY-MM-DD"
    horizon_months: int = 6


# --- Curated suggestions for health queries (prefix → canonical disease) ---
DISEASES = {
    "covid 19": ["covid 19", "covid19", "covid-19", "covid", "corona"],
    "dengue":   ["dengue", "den", "dengu"],
    "influenza": ["influenza", "flu", "flu "],  # note: "flu " to avoid matching words like "fluent"
    "malaria":  ["malaria", "malar"],
    "tuberculosis": ["tuberculosis", "tb", "tuber"],
}

# Put your priority countries first; “World” at the end is nice to have
COUNTRIES = [
    "Sri Lanka", "India", "Bangladesh", "Pakistan", "Nepal",
    "United States", "United Kingdom", "Australia", "Canada",
    "Japan", "China", "France", "Germany", "Spain", "Italy",
    "Indonesia", "World",
]

# --- add these helpers near the top of orchestrator/main.py ---
def _needs_llm_fallback(ret_payload: dict) -> bool:
    """
    Decide when to fall back to Chat Agent:
    - adapter error (type contains 'error')
    - empty/no facts.data
    - summary contains 'could not be fetched'
    """
    try:
        facts = ret_payload.get("facts") or {}
        t = (facts.get("type") or "").lower()
        if "error" in t:
            return True
        data = facts.get("data")
        if data in (None, {}) and t not in ("general_health",):
            return True
        summary = (facts.get("summary") or "").lower()
        if "could not be fetched" in summary:
            return True
    except Exception:
        return True
    return False


async def _links_suggestion(user_text: str, httpx_client: "httpx.AsyncClient", retrieval_base: str) -> list[dict]:
    """
    Ask Retrieval Agent for 'links only' so we can show official sources
    even when live data fetch fails.
    """
    try:
        r = await httpx_client.post(
            f"{retrieval_base.rstrip('/')}/search",
            json={"question": user_text, "mode": "links", "filters": {}},
        )
        if r.status_code < 400:
            return (r.json() or {}).get("items", [])[:5]
    except Exception:
        pass
    return []


async def _ollama_summarize(query_text: str, facts_payload: dict) -> str:
    """
    Ask a local Ollama model to write a brief, human-friendly paragraph explaining
    the retrieval results. Never invent numbers – just rephrase.
    Returns a string or "" on failure.
    """
    try:
        msg_user = (
            "You are a public-health explainer. Rephrase the provided factual JSON "
            "into a concise (2–4 sentences) human-friendly summary.\n"
            "Rules:\n"
            "• Do NOT make up numbers or dates; only rephrase what is present.\n"
            "• If a country or disease is present, mention it naturally.\n"
            "• If sources are provided, mention them by name (no URLs) at the end as 'Sources: …'.\n"
            "• If there is insufficient data, say so briefly.\n\n"
            f"User query:\n{query_text}\n\n"
            "Facts JSON (verbatim; use values as-is):\n"
            f"{json.dumps(facts_payload, ensure_ascii=False, indent=2)}"
        )

        # POST /api/chat (Ollama)
        import httpx
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "You turn structured health data into clear, honest summaries."},
                        {"role": "user", "content": msg_user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()
    except Exception:
        return ""

def _canonical_disease_from_prefix(q: str) -> str | None:
    s = (q or "").lower().strip()
    # strongest signal: user starts typing the disease name
    for canonical, aliases in DISEASES.items():
        for a in aliases:
            if s.startswith(a):
                return canonical
    # weaker: disease occurs anywhere early in the string
    for canonical, aliases in DISEASES.items():
        for a in aliases:
            if a in s:
                return canonical
    return None

def _curated_suggestions(q: str, max_items: int = 10) -> list[str]:
    """
    Returns a list of '<disease> in <country>' suggestions if user is typing a known disease.
    We deliberately avoid metrics like 'deaths', 'cases', etc. per your requirement.
    """
    s = (q or "").strip().lower()
    if not s:
        return []

    disease = _canonical_disease_from_prefix(s)
    if not disease:
        return []

    # Generate the pattern list and filter to the user's current prefix
    raw = []
    for c in COUNTRIES:
        raw.append(f"{disease} in {c}")

        # For COVID we also allow a shorter "covid in X" variant that users often type.
        if disease == "covid 19":
            raw.append(f"covid in {c}")

    # Prefix filter: show only suggestions that begin with what the user typed
    seen = set()
    out = []
    for item in raw:
        if item.lower().startswith(s) and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
        if len(out) >= max_items:
            break
    return out


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Report routes (pass-through)
# -------------------------
@app.post("/route/report_from_text")
async def route_report_from_text(body: TextRequest):
    url = f"{REPORT_BASE}/report_from_text"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=body.dict())
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@app.post("/route/report")
async def route_report(req: Request):
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    url = f"{REPORT_BASE}/report"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


# -------------------------
# Static artifact proxy (charts, html, pdfs) from Report Agent
# -------------------------
@app.api_route("/agents/{path:path}", methods=["GET"])
async def proxy_agents(path: str, request: Request):
    upstream = f"{REPORT_BASE}/agents/{path}"

    headers = {}
    for k, v in request.headers.items():
        if k.lower() in {"range", "accept", "accept-encoding", "user-agent"}:
            headers[k] = v

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.get(upstream, headers=headers)
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers={"content-type": r.headers.get("content-type", "application/octet-stream")},
    )


# -------------------------
# Retrieval pass-through (direct call)
# -------------------------
@app.post("/route/retrieval/search")
async def route_retrieval_search(body: RetrievalRequest):
    url = f"{RETRIEVAL_BASE}/search"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=body.dict())
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


# -------------------------
# CHAT ROUTER
#   - Real-world data intent -> Retrieval Agent (/search)
#   - Report/graph intent     -> Report Agent (/report_from_text)
#   - Otherwise               -> canned Chat Agent reply
# -------------------------

# Explicit phrases that mean "make me a report/graph"
REPORT_KEY_PHRASES = [
    "report", "graph", "chart", "figure", "pdf",
    "generate a report", "create a report", "visualize", "visualise",
]

# Disease hints used with "trend/time series" to qualify a reporting intent
REPORT_DISEASE_HINTS = [
    "dengue", "malaria", "covid", "influenza", "flu", "tb", "tuberculosis",
]

# Retrieval keywords: things we have adapters/APIs for or general metrics
RETRIEVAL_KWS = [
    # Flu/ILI
    "flu", "influenza", "ili",
    # COVID & general epi terms
    "covid", "cases", "deaths", "recovered", "incidence", "prevalence",
    "mortality", "death rate", "case fatality",
    # medicines / fda
    "side effect", "adverse", "reaction", "fda", "medicine", "drug",
    # nutrition / usda
    "nutrition", "usda", "vitamin", "protein", "calcium", "iron",
    # world bank topics
    "life expectancy", "under 5 mortality", "under-5 mortality", "under five mortality",
    "health expenditure", "health spending", "malaria incidence", "tb incidence", "tuberculosis incidence",
]

def _wants_retrieval(text: str) -> bool:
    low = text.lower()
    simple_hits = [
        # COVID
        "covid", "cases", "deaths", "recovered",
        # medicines / fda
        "side effect", "adverse", "reaction", "fda", "medicine", "drug",
        # nutrition / usda
        "nutrition", "usda", "vitamin", "protein", "calcium", "iron",
        # general epi
        "incidence", "prevalence", "mortality", "death rate", "case fatality",
        # world bank-ish
        "life expectancy", "under 5 mortality", "under-5 mortality", "under five mortality",
        "health expenditure", "health spending", "malaria incidence", "tb incidence", "tuberculosis incidence",
        # flu
        "flu", "influenza", "ili"
    ]
    if any(k in low for k in simple_hits):
        return True
    patterns = [
        r"\bunder[- ]?5\b.*\bmortality\b",
        r"\blife expectancy\b",
        r"\bhealth (?:expenditure|spending)\b",
        r"\b(?:malaria|tb|tuberculosis)\s+incidence\b",
        r"\b(death|mortality)\s+rate\b",
    ]
    return any(re.search(p, low) for p in patterns)

def _wants_report(text: str) -> bool:
    low = text.lower()
    report_kw = [
        "report", "graph", "chart", "trend",
        "visualize", "visualise", "time series", "timeseries",
        "generate a report"
    ]
    return any(k in low for k in report_kw)

# ---- DROP-IN REPLACEMENT for your /route/chat ----
@app.post("/route/chat")
async def route_chat(body: ChatBody):
    import os
    msg = (body.message or "").strip()

    # --- LLM toggle (env) ---
    USE_LLM = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")

    # --- Security PRECHECK (auth + unsafe content check) ---
    SEC_BASE = AGENTS.get("security", "").rstrip("/")
    SEC_USER = os.getenv("SEC_USER", "admin")
    SEC_PASS = os.getenv("SEC_PASS", "admin")

    if SEC_BASE:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                pre = await client.post(
                    f"{SEC_BASE}/precheck",
                    json={"username": SEC_USER, "password": SEC_PASS, "message": msg},
                )
            pre_data = pre.json()
            if not pre_data.get("ok"):
                return {
                    "type": "blocked",
                    "summary": pre_data.get("message", "Message blocked by security policy."),
                    "sources": [],
                }
        except Exception:
            # fail-open if security agent is unavailable
            pass

    # ---------------- ROUTING ----------------

    # 1) Report intent → Report Agent
    if _wants_report(msg):
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{AGENTS['report'].rstrip('/')}/report_from_text",
                json={"query": msg},
            )
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        data = r.json()
        resp = {
            "type": "report",
            "summary": data.get("summary"),
            "visuals": data.get("visuals", []),
            "report_url": data.get("report_url"),
            "pdf_url": data.get("pdf_url"),
            "sources": data.get("sources", []),
            "disclaimer": data.get("disclaimer"),
        }

        if USE_LLM and resp.get("summary"):
            improved = polish_chat(resp["summary"])
            if improved:
                resp["summary_llm"] = improved

    # 2) Retrieval intent → Retrieval Agent (with fallback to Chat Agent)
    elif _wants_retrieval(msg):
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{AGENTS['retrieval'].rstrip('/')}/search",
                json={"question": msg},
            )
            if r.status_code >= 400:
                raise HTTPException(status_code=r.status_code, detail=r.text)

            data = r.json()

            # ---- NEW: if retrieval failed/empty → LLM fallback + links ----
            if _needs_llm_fallback(data):
                # ask Chat Agent for a concise explainer
                try:
                    chat_r = await client.post(
                        f"{AGENTS['chat'].rstrip('/')}/chat",
                        json={
                            "message": (
                                "The user asked: \"{q}\"\n\n"
                                "Live data could not be fetched. Provide a concise, "
                                "trustworthy explanation using WHO/CDC guidance. "
                                "Avoid speculation; suggest where to find current figures "
                                "(WHO dashboard, Our World in Data)."
                            ).format(q=msg),
                            "history": [],
                        },
                    )
                    chat_text = (chat_r.json() or {}).get("reply", "") if chat_r.status_code < 400 else ""
                except Exception:
                    chat_text = ""

                # also fetch authoritative links so UI can show sources
                links = await _links_suggestion(msg, client, AGENTS["retrieval"])

                resp = {
                    "type": "chat",
                    "reply": chat_text or "Live data was unavailable. Please check WHO and Our World in Data for the latest figures.",
                    "fallback": True,
                    "suggested_links": links,
                }

                # optional polish
                if USE_LLM and resp.get("reply"):
                    improved = polish_chat(resp["reply"])
                    if improved:
                        resp["reply_llm"] = improved

            else:
                # Normal path: retrieval worked
                # Friendly LLM summary (optional)
                llm_text = ""
                try:
                    llm_text = await _ollama_summarize(msg, data.get("facts", {}))
                except Exception:
                    llm_text = ""

                resp = {
                    "type": "retrieval",
                    "query": data.get("query", {}),
                    "facts": data.get("facts", {}),
                    "sources": data.get("sources", []),
                }
                if llm_text:
                    resp["llm_summary"] = llm_text

                if USE_LLM:
                    improved = summarize_retrieval(resp)
                    if improved:
                        resp["summary_llm"] = improved

    # 3) Fallback → canned Chat Agent
    else:
        from app.agents.chat_agent.engine import get_chat_response
        reply = get_chat_response(msg)
        resp = {"type": "chat", "reply": reply}

        if USE_LLM and reply:
            improved = polish_chat(reply)
            if improved:
                resp["reply_llm"] = improved

    # --- Security POSTCHECK (mask + encrypt final user-visible text) ---
    if SEC_BASE:
        try:
            text_for_user = (
                resp.get("summary_llm")
                or resp.get("summary")
                or (resp.get("facts") or {}).get("summary")
                or resp.get("reply_llm")
                or resp.get("reply")
                or ""
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                post = await client.post(
                    f"{SEC_BASE}/postcheck",
                    json={"text": text_for_user},
                )
            post_data = post.json()
            resp["summary_masked"] = post_data.get("masked", text_for_user)
            resp["encrypted"] = post_data.get("encrypted", "")
        except Exception:
            # fail-open if security agent is unavailable
            pass

    return resp

async def _security_precheck(text: str) -> dict:
    """
    Call Security Agent /precheck. Returns dict like {"ok": True/False, "message": "..."}
    If the security agent is down, fail open (ok=True) so your demo keeps working.
    """
    if not SEC_BASE:
        return {"ok": True, "message": text}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{SEC_BASE}/precheck",
                json={"username": SEC_USER, "password": SEC_PASS, "message": text},
            )
        return r.json()
    except Exception:
        return {"ok": True, "message": text}  # fail open for resilience

async def _security_postcheck(text: str) -> dict:
    """
    Call Security Agent /postcheck. Returns {"masked": "...", "encrypted": "<fernet token>"}.
    If the agent is down, return the raw text.
    """
    if not SEC_BASE:
        return {"masked": text, "encrypted": ""}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{SEC_BASE}/postcheck", json={"text": text})
        return r.json()
    except Exception:
        return {"masked": text, "encrypted": ""}


async def _filter_links_via_security(urls: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Ask Security Agent whether each URL is allowed.
    Returns mapping {url: {"ok": bool, "reason": str}}.
    If Security Agent is unreachable, we'll fail open (treat all as ok).
    """
    if not SEC_BASE or not urls:
        # security disabled or nothing to check -> everything ok
        return {u: {"ok": True, "reason": "security-disabled"} for u in urls}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                f"{SEC_BASE}/check_url",
                headers={"x-api-key": SEC_API_KEY},
                json={"urls": urls},
            )
        r.raise_for_status()
        payload = r.json() or {}
        results = payload.get("results", [])
        # Normalize to mapping
        out = {}
        for row in results:
            out[row.get("url", "")] = {
                "ok": bool(row.get("ok", False)),
                "reason": row.get("reason", ""),
            }
        # Fill any missing
        for u in urls:
            out.setdefault(u, {"ok": False, "reason": "no-decision"})
        return out
    except Exception:
        # Fail-open: if security agent is down, mark all ok so the app keeps working
        return {u: {"ok": True, "reason": "security-unreachable"} for u in urls}


@app.post("/route/search")
async def route_search(body: SearchBody):
    """
    Proxy to Retrieval Agent, then validate all returned links with Security Agent.
    We keep only allowed links (default). Flip KEEP_BLOCKED to True if you prefer annotating instead of removing.
    """
    q = (body.query or "").strip()
    filters = body.filters or {}
    mode = (body.mode or "links").lower()

    # 1) Ask retrieval for links-only envelope
    #    (Your retrieval agent already returns {"type":"search","items":[{title, url, source}], "query": {...}})
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{RETRIEVAL_BASE}/search",
            json={"question": q, "mode": "links", "filters": filters},
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    payload = r.json()

    items = payload.get("items", [])
    urls = [it.get("url") for it in items if it.get("url")]

    # 2) Validate with Security Agent
    decisions = await _filter_links_via_security(urls)

    # 3) Attach decision to each item, and (by default) remove blocked links
    KEEP_BLOCKED = False  # set True if you prefer to *show* but mark blocked
    filtered_items = []
    for it in items:
        u = it.get("url")
        if not u:
            continue
        dec = decisions.get(u, {"ok": False, "reason": "no-decision"})
        it["ok"] = bool(dec["ok"])
        it["block_reason"] = dec.get("reason", "")
        if it["ok"] or KEEP_BLOCKED:
            filtered_items.append(it)

    payload["items"] = filtered_items
    payload["security"] = {
        "checked": len(urls),
        "blocked": sum(1 for d in decisions.values() if not d["ok"]),
        "mode": "fail-open" if all(d["reason"] == "security-unreachable" for d in decisions.values()) else "enforced",
    }

    return payload


@app.get("/route/search_suggest")
async def route_search_suggest(q: str):
    # 1) Curated health-aware suggestions
    curated = _curated_suggestions(q, max_items=10)
    if curated:
        return {"suggestions": curated}

    # 2) Fallback to Retrieval Agent suggestions (if any)
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{RETRIEVAL_BASE}/suggest", params={"q": q})
    if r.status_code >= 400:
        # fail-soft: no suggestions
        return {"suggestions": []}
    return r.json()

@app.post("/route/report_form")
async def route_report_form(body: ReportFormRequest):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{REPORT_BASE}/report_form",
            json=body.dict()
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@app.post("/route/forecast")
async def route_forecast(body: ForecastBody):
    import os
    # --- Security agent base + creds (same pattern you use elsewhere) ---
    SEC_BASE = AGENTS.get("security", "").rstrip("/")
    SEC_USER = os.getenv("SEC_USER", "admin")
    SEC_PASS = os.getenv("SEC_PASS", "admin")

    # Build a simple string for the security precheck
    precheck_msg = (
        f"forecast request: disease={body.disease}, region={body.region}, "
        f"{body.date_from}→{body.date_to}, horizon={body.horizon_months}m"
    )

    # --- PRECHECK (auth + unsafe text) ---
    if SEC_BASE:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                pre = await client.post(
                    f"{SEC_BASE}/precheck",
                    json={"username": SEC_USER, "password": SEC_PASS, "message": precheck_msg},
                )
            pre_data = pre.json()
            if not pre_data.get("ok"):
                return {
                    "type": "blocked",
                    "summary": pre_data.get("message", "Message blocked by security policy."),
                    "sources": [],
                }
        except Exception:
            # fail-open if security agent unavailable
            pass

    # --- DELEGATE to Report Generator’s /forecast ---
    REPORT_BASE = AGENTS.get("report", "").rstrip("/")
    if not REPORT_BASE:
        raise HTTPException(status_code=500, detail="Report agent URL missing in orchestrator config.")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{REPORT_BASE}/forecast", json=body.model_dump())
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Forecast agent error: {e}")

    # --- POSTCHECK (mask/encrypt user-visible text) ---
    # We’ll pass a compact summary string for masking/encryption.
    if SEC_BASE:
        try:
            summary_text = (
                f"Forecast for {body.disease} in {body.region} "
                f"{body.date_from}→{body.date_to} (h={body.horizon_months}). "
                f"Method={data.get('method')}. "
                f"Forecast points={len(data.get('forecast', []))}."
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                post = await client.post(f"{SEC_BASE}/postcheck", json={"text": summary_text})
            post_data = post.json()
            data["summary_masked"] = post_data.get("masked", summary_text)
            data["encrypted"] = post_data.get("encrypted", "")
        except Exception:
            pass

    # Shape a consistent orchestrator response
    return {
        "type": "forecast",
        "payload": data,  # includes history, forecast, method, provenance, warnings
    }


@app.post("/route/train_model")
async def route_train_model(req: Request):
    payload = await req.json()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{REPORT_URL}/train_model", json=payload)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()

@app.post("/route/predict_month")
async def route_predict_month(req: Request):
    payload = await req.json()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{REPORT_URL}/predict_month", json=payload)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()

@app.get("/route/news")
async def route_news(limit: int = 12):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{RETRIEVAL_URL}/news", params={"limit": limit})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

@app.get("/route/news/sources")
async def route_news_sources():
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{RETRIEVAL_URL}/news/sources")
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
