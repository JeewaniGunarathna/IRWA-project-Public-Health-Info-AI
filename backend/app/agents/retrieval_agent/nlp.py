# backend/retrieval_agent/nlp.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import re
import dateparser
import spacy

# Load spaCy small English model once
_nlp = spacy.load("en_core_web_sm")

# --- Domain dictionaries (expand anytime) ---
DISEASE_SYNONYMS = {
    "covid": "covid-19",
    "covid-19": "covid-19",
    "coronavirus": "covid-19",
    "sars-cov-2": "covid-19",
    "dengue": "dengue",
    "malaria": "malaria",
    "tb": "tuberculosis",
    "tuberculosis": "tuberculosis",
    "influenza": "influenza",
    "flu": "influenza",
    "hiv": "hiv",
    "aids": "hiv",
    "measles": "measles",
}

# minimal country/region list (you already had one in worldbank.py – you can import/reuse)
COUNTRY_ALIASES = {
    "sri lanka": "Sri Lanka",
    "india": "India",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "pakistan": "Pakistan",
    "bangladesh": "Bangladesh",
    "nepal": "Nepal",
    "china": "China",
    "japan": "Japan",
    "australia": "Australia",
    "canada": "Canada",
    "france": "France",
    "germany": "Germany",
    "spain": "Spain",
    "italy": "Italy",
    "indonesia": "Indonesia",
    "world": "World",
    "global": "World",
}

# quick regex for “in <place>”
IN_RX = re.compile(r"\b(?:in|for|within)\s+([a-z][a-z\s\.-]{2,})\b", re.I)

def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()

def _pick_disease(doc) -> Optional[str]:
    low = doc.text.lower()
    # dictionary-based match first (fast & robust)
    for k, canon in DISEASE_SYNONYMS.items():
        if re.search(rf"\b{k}\b", low):
            return canon

    # spaCy entities (MEDICAL entities are not in small model; but GPE/ORG can hint)
    # We keep it simple; dictionary covers >90% for our scope.
    return None

def _pick_country(doc) -> Optional[str]:
    # 1) explicit "in <place>" phrase
    m = IN_RX.search(doc.text)
    if m:
        cand = _norm(m.group(1))
        cand = re.sub(r"[^\w\s\-\.]", "", cand)
        if cand in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[cand]

    # 2) spaCy NER (GPE/LOC/ORG often catches countries)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "ORG"):
            cand = _norm(ent.text)
            if cand in COUNTRY_ALIASES:
                return COUNTRY_ALIASES[cand]

    # 3) dictionary scan fallback
    low = _norm(doc.text)
    for k, canon in COUNTRY_ALIASES.items():
        if re.search(rf"\b{k}\b", low):
            return canon
    return None

def _pick_dates(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to parse time windows like:
      - 'from 2021 to 2023'
      - 'Jan 2020 - Mar 2021'
      - 'last year', 'past 6 months', etc.
    Returns ISO dates (YYYY-MM-DD) where possible.
    """
    low = raw.lower()

    # explicit from..to..
    m = re.search(r"\bfrom\s+(.+?)\s+(?:to|until|-)\s+(.+)$", low)
    if m:
        d1 = dateparser.parse(m.group(1), settings={"PREFER_DAY_OF_MONTH": "first"})
        d2 = dateparser.parse(m.group(2), settings={"PREFER_DAY_OF_MONTH": "last"})
        return (d1.date().isoformat() if d1 else None, d2.date().isoformat() if d2 else None)

    # single year
    m = re.search(r"\b(20\d{2}|19\d{2})\b", low)
    if m:
        y = int(m.group(1))
        return (f"{y}-01-01", f"{y}-12-31")

    # relative time like “past 6 months”, “last year”
    rel = dateparser.parse(low)
    # For generic prompts we don’t guess both sides; return (None, yyyy-mm-dd) if only an end exists
    if rel:
        return (None, rel.date().isoformat())

    return (None, None)

def parse_query(raw: str) -> Dict:
    """
    Main entry: return a normalized structure you can hand to your agent.
    """
    doc = _nlp(raw)
    disease = _pick_disease(doc)
    country = _pick_country(doc)
    date_from, date_to = _pick_dates(raw)

    intent = "general"
    # tiny intent hints
    if re.search(r"\b(graph|chart|trend|time\s*series|report)\b", raw, re.I):
        intent = "report"
    elif re.search(r"\b(source|link|reference|where)\b", raw, re.I):
        intent = "links"

    return {
        "raw": raw,
        "disease": disease,       # e.g. "covid-19"
        "country": country,       # e.g. "Sri Lanka" or "World"
        "date_from": date_from,   # "YYYY-MM-DD" or None
        "date_to": date_to,       # "YYYY-MM-DD" or None
        "intent": intent,
    }
