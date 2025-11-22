import re
from datetime import datetime
from dateutil import parser as date_parse

# very small list just to avoid crazy inputs; feel free to expand
KNOWN_DISEASES = [
    "dengue","covid-19","covid","influenza","malaria","cholera","measles"
]

def _try_parse_date(s: str):
    # Accepts yyyy-mm-dd or common date words/numbers; falls back to None
    try:
        return date_parse.parse(s, dayfirst=False, yearfirst=True).date().isoformat()
    except Exception:
        return None

def extract(query: str) -> dict:
    """
    Returns a dict shaped for ReportRequest (with safe defaults)
    {
      disease, region, date_from, date_to, timeseries, insights, sources
    }
    """
    q = query.strip()

    # 1) disease
    disease = None
    # try direct match from known list
    for d in KNOWN_DISEASES:
        if re.search(rf"\b{re.escape(d)}\b", q, flags=re.IGNORECASE):
            disease = d.title() if d != "covid-19" else "COVID-19"
            break
    # fallback: first capitalized word as disease (very rough)
    if not disease:
        m = re.search(r"\breport for ([A-Za-z- ]+)", q, re.IGNORECASE)
        if m:
            disease = m.group(1).strip().split()[0].title()  # naive
    disease = disease or "Dengue"

    # 2) region (look for 'in <region>' or 'for <region>' after disease)
    region = None
    m = re.search(r"\b(?:in|for)\s+([A-Z][A-Za-z .-]+)", q)
    if m:
        region = m.group(1).strip()
    region = region or "Sri Lanka"

    # 3) dates — try explicit YYYY-MM-DD first
    #   - patterns like 2025-07-01 to 2025-08-24
    m = re.search(r"(\d{4}-\d{2}-\d{2}).{0,10}(to|until|-).{0,10}(\d{4}-\d{2}-\d{2})", q)
    date_from = None
    date_to = None
    if m:
        date_from = _try_parse_date(m.group(1))
        date_to   = _try_parse_date(m.group(3))

    # fallback: month names or single dates (e.g., "June 2025 to July 2025")
    if not (date_from and date_to):
        m2 = re.search(r"from\s+([A-Za-z0-9 ,/-]+)\s+(?:to|until|through)\s+([A-Za-z0-9 ,/-]+)", q, re.IGNORECASE)
        if m2:
            date_from = _try_parse_date(m2.group(1))
            date_to   = _try_parse_date(m2.group(2))

    # safe default: last 8 weeks ending today
    if not date_to:
        date_to = datetime.utcnow().date().isoformat()
    if not date_from:
        from datetime import timedelta, date
        date_from = (datetime.utcnow().date() - timedelta(days=56)).isoformat()

    # 4) sources (URLs mentioned)
    urls = re.findall(r"(https?://[^\s)]+)", q)
    sources = []
    for u in urls:
        name = "Source"
        if "who.int" in u:
            name = "WHO"
        elif "mohfw" in u or "health.gov" in u or "gov" in u:
            name = "Ministry of Health"
        sources.append({"name": name, "url": u})

    # 5) optional insights (user hints like “increase 12%”, “top region Colombo”)
    weekly_inc = None
    m = re.search(r"(increase|decrease)\s+by\s+([0-9]+%|\d+(?:\.\d+)?%)", q, re.IGNORECASE)
    if m:
        sign = "+" if m.group(1).lower()=="increase" else "-"
        weekly_inc = sign + m.group(2) if not m.group(2).startswith(("+","-")) else m.group(2)
    top_region = None
    m = re.search(r"(?:top|highest)\s+(?:region|area)\s+([A-Z][A-Za-z .-]+)", q)
    if m:
        top_region = m.group(1).strip()

    insights = {}
    if weekly_inc: insights["weekly_increase"] = weekly_inc
    if top_region: insights["top_region"] = top_region

    # We don’t fabricate timeseries from text in Week‑6.
    # Leave it empty; your agent will still render a text‑only report, or
    # you can require users to upload data via the other endpoint when needed.
    return {
        "disease": disease,
        "region": region,
        "date_from": date_from,
        "date_to": date_to,
        "timeseries": [],
        "insights": insights or None,
        "sources": sources or [{"name":"WHO","url":"https://www.who.int"}],
    }
