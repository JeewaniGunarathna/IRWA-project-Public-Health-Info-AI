# backend/retrieval_agent/adapters/worldbank.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import requests

from .base import FactsAdapter

# ---------------- Country mapping (extend as needed) ----------------
NAME_TO_ISO3 = {
    # World/Global
    "world": "WLD", "global": "WLD",

    # South Asia + neighbors
    "sri lanka": "LKA", "india": "IND", "bangladesh": "BGD",
    "pakistan": "PAK", "nepal": "NPL",

    # East/Southeast Asia
    "china": "CHN", "japan": "JPN", "indonesia": "IDN",

    # Americas
    "united states": "USA", "us": "USA", "u.s.": "USA", "usa": "USA",
    "canada": "CAN",

    # Europe
    "united kingdom": "GBR", "uk": "GBR",
    "france": "FRA", "germany": "DEU", "spain": "ESP", "italy": "ITA",

    # Oceania
    "australia": "AUS",
}

def _to_iso3(text: Optional[str]) -> str:
    """
    Map a name/ISO code to ISO3. If nothing fits, default to WLD (World).
    Accepts ISO2/ISO3 codes (passes through uppercased).
    """
    if not text:
        return "WLD"
    s = text.strip().lower()
    if len(s) in (2, 3) and s.isalpha():
        return s.upper()
    return NAME_TO_ISO3.get(s, "WLD")

# ---------------- Indicators & triggers ----------------
# topic_key -> (indicator_code, human_title)
WB_TOPICS: Dict[str, Tuple[str, str]] = {
    "tb_incidence":        ("SH.TBS.INCD",     "TB incidence (per 100,000)"),
    "malaria_incidence":   ("SH.MLR.INCD.P3",  "Malaria incidence (per 1,000 at risk)"),
    "hiv_prevalence":      ("SH.DYN.AIDS.ZS",  "HIV prevalence (% ages 15–49)"),
    "measles_immun":       ("SH.IMM.MEAS",     "Measles immunization (MCV1, %)"),
    "hepb_immun":          ("SH.IMM.HEPB",     "Hepatitis B immunization (HepB3, %)"),
    "maternal_mortality":  ("SH.STA.MMRT",     "Maternal mortality (per 100,000)"),
    "under5_mortality":    ("SH.DYN.MORT",     "Under-5 mortality (per 1,000)"),
}

# words that trigger each topic
TOPIC_TRIGGERS: Dict[str, List[str]] = {
    "tb_incidence":       ["tb incidence", "tuberculosis incidence", "tb"],
    "malaria_incidence":  ["malaria incidence", "malaria"],
    "hiv_prevalence":     ["hiv prevalence", "hiv"],
    "measles_immun":      ["measles immunization", "measles vaccine", "measles"],
    "hepb_immun":         ["hepatitis b immunization", "hepb immunization", "hepatitis b vaccine", "hepb"],
    "maternal_mortality": ["maternal mortality", "maternal death"],
    "under5_mortality":   ["under 5 mortality", "under-five mortality", "under five mortality", "child mortality"],
}

def _detect_topic(text: str) -> Optional[str]:
    s = (text or "").lower()
    for key, words in TOPIC_TRIGGERS.items():
        if any(w in s for w in words):
            return key
    return None

# ---------------- World Bank fetch helpers ----------------
def _wb_series(iso3: str, indicator: str) -> List[dict]:
    """
    Fetch indicator time series from World Bank API and return
    [{date: 'YYYY-01-01', value: float}], oldest → newest.
    """
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json&per_page=70"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data[1] if isinstance(data, list) and len(data) > 1 else []
    series: List[dict] = []
    for row in rows:
        y = row.get("date")
        v = row.get("value")
        if not y:
            continue
        series.append({"date": f"{y}-01-01", "value": (float(v) if v is not None else 0.0)})
    series.sort(key=lambda x: x["date"])
    return series

def _first_last_change(series: List[dict]) -> Tuple[float, float, float]:
    """(start, end, change_pct) across the series window."""
    if not series:
        return 0.0, 0.0, 0.0
    start = float(series[0]["value"] or 0.0)
    end = float(series[-1]["value"] or 0.0)
    if start == 0.0:
        return start, end, 0.0
    change = round(((end - start) / start) * 100.0, 1)
    return start, end, change

def _latest_nonzero(series: List[dict]) -> Optional[Tuple[str, float]]:
    """Most recent non-zero (year, value)."""
    for row in reversed(series):
        val = float(row.get("value") or 0.0)
        if val != 0.0:
            return row["date"].split("-")[0], val
    return None

# ---------------- Adapter ----------------
class WorldBankAdapter(FactsAdapter):
    """
    Real data adapter for World Bank health indicators.
    Supports queries like:
      - "tb incidence in India"
      - "malaria incidence world"
      - "hiv prevalence in Sri Lanka"
      - "measles immunization Bangladesh"
      - "hepatitis b immunization in Nepal"
      - "maternal mortality Pakistan"
      - "under 5 mortality United States"
    """

    def supports(self, query: Dict[str, Any]) -> bool:
        text = (query.get("question") or query.get("raw") or "").lower()
        return _detect_topic(text) is not None

    def fetch(self, query: Dict[str, Any]) -> Dict[str, Any]:
        text = (query.get("question") or query.get("raw") or "")
        topic = _detect_topic(text)
        if not topic:
            return {"type": "worldbank", "summary": "No indicator matched.", "data": {}, "sources": []}

        indicator, title = WB_TOPICS[topic]
        iso3 = _to_iso3(query.get("country") or query.get("region"))

        series = _wb_series(iso3, indicator)
        start, end, change_pct = _first_last_change(series)
        latest = _latest_nonzero(series)

        if latest:
            y, v = latest
            summary = f"{iso3}: {title} = {v} ({y}). Change across series: {change_pct}%."
        else:
            summary = f"{iso3}: {title} — no recent non-zero data."

        return {
            "type": f"worldbank_{indicator.lower()}",
            "summary": summary,
            "data": {
                "country": iso3,
                "indicator": indicator,
                "title": title,
                "series": series,          # [{date, value}] oldest → newest
                "start": start,
                "end": end,
                "change_pct": change_pct,
                "latest": ({"year": latest[0], "value": latest[1]} if latest else None),
            },
            "sources": [
                {
                    "name": "World Bank API",
                    "url": f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json",
                }
            ],
        }
