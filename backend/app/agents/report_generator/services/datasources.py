# backend/agents/report_generator/services/datasources.py
from __future__ import annotations
import datetime as dt
from typing import List, Dict, Optional
import httpx

def _iso(d: dt.date) -> str:
    return d.isoformat()

def _parse_date(s: str) -> dt.date:
    # accepts YYYY-MM-DD
    return dt.date.fromisoformat(s)

def _clamp_dates(date_from: str, date_to: str) -> tuple[dt.date, dt.date]:
    d1 = _parse_date(date_from)
    d2 = _parse_date(date_to)
    if d2 < d1:
        d2 = d1
    return d1, d2

async def fetch_covid_timeseries(country: str, date_from: str, date_to: str) -> Optional[List[Dict]]:
    """
    Pull daily cases for a country between date_from and date_to
    using disease.sh historical endpoint.

    Returns list of {"date": "YYYY-MM-DD", "value": <int>} or None if not found.
    """
    d1, d2 = _clamp_dates(date_from, date_to)

    # disease.sh returns a long history; we’ll slice to the requested window
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays=all"

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            return None
        data = r.json()

    # structure: data["timeline"]["cases"] = {"3/1/20": 10, ...}
    tl = data.get("timeline", {})
    raw_cases = tl.get("cases")
    if not isinstance(raw_cases, dict) or not raw_cases:
        return None

    # disease.sh dates are like "3/1/20"; convert to ISO and sort
    series = []
    for mdY, cum in raw_cases.items():
        # mm/dd/yy -> YYYY-MM-DD
        try:
            d = dt.datetime.strptime(mdY, "%m/%d/%y").date()
        except ValueError:
            continue
        series.append((d, int(cum)))

    series.sort(key=lambda x: x[0])

    # Convert cumulative to daily new cases
    daily = []
    prev = None
    for d, cum in series:
        val = 0 if prev is None else max(cum - prev, 0)
        prev = cum
        daily.append((d, val))

    # window filter
    window = [(d, v) for (d, v) in daily if d1 <= d <= d2]
    if not window:
        return None

    return [{"date": _iso(d), "value": int(v)} for d, v in window]


async def fetch_timeseries_if_possible(
    disease: str,
    region: str,
    date_from: str,
    date_to: str,
) -> Optional[List[Dict]]:
    """
    Entry point used by the report generator. Add more branches later
    (e.g., dengue -> WHO adapter) with the same return shape.
    """
    if not disease or not region:
        return None

    dz = disease.strip().lower()
    # try COVID from disease.sh
    if dz in {"covid", "covid-19", "coronavirus"}:
        ts = await fetch_covid_timeseries(region, date_from, date_to)
        return ts

    # TODO: add more diseases here
    return None
