# agents/report_generator/services/timeseries.py
from __future__ import annotations

import math
import random
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

# -------------------------
# Basic date helpers
# -------------------------
def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _format_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


# -------------------------
# Country normalization (ISO3)
# -------------------------
NAME_TO_ISO3 = {
    "sri lanka": "LKA",
    "india": "IND",
    "united states": "USA", "u.s.": "USA", "us": "USA", "usa": "USA",
    "united kingdom": "GBR", "uk": "GBR",
    "pakistan": "PAK",
    "bangladesh": "BGD",
    "nepal": "NPL",
    "china": "CHN",
    "japan": "JPN",
    "australia": "AUS",
    "canada": "CAN",
    "france": "FRA",
    "germany": "DEU",
    "spain": "ESP",
    "italy": "ITA",
    "indonesia": "IDN",
    "world": "WLD",  # special handling for sources that support it
}

ISO2_TO_ISO3 = {
    "LK": "LKA", "IN": "IND", "US": "USA", "GB": "GBR", "PK": "PAK", "BD": "BGD",
    "NP": "NPL", "CN": "CHN", "JP": "JPN", "AU": "AUS", "CA": "CAN",
    "FR": "FRA", "DE": "DEU", "ES": "ESP", "IT": "ITA", "ID": "IDN",
}

def to_iso3(text: Optional[str]) -> str:
    if not text:
        return "LKA"
    s = text.strip()
    if len(s) in (2, 3) and s.isalpha():
        up = s.upper()
        return ISO2_TO_ISO3.get(up, up)  # pass through ISO3 / map ISO2→ISO3
    iso3 = NAME_TO_ISO3.get(s.lower())
    return iso3 or "LKA"

def generate_synthetic_timeseries(
    date_from: str,
    date_to: str,
    weekly_increase_str: Optional[str] = None,
    points: int = 60,              # more points → smoother chart
    start_value: float = 100.0,
) -> List[dict]:
    """
    Synthetic curve with (optional) growth, seasonality, and light noise.
    Produces {date, value} points between the two dates.
    """
    d0 = _parse_date(date_from)
    d1 = _parse_date(date_to)
    if d1 <= d0:
        d1 = d0 + timedelta(days=7)

    total_days = max(1, (d1 - d0).days)
    # even spacing across the window
    step_days = max(1, total_days // max(1, points - 1))

    # growth factor per step (if % provided)
    growth_perc = 0.0
    if weekly_increase_str:
        try:
            growth_perc = float(weekly_increase_str.strip().replace("%", ""))
        except Exception:
            growth_perc = 0.0
    # approximate growth per step
    step_growth = (growth_perc / 100.0) / max(1, (points - 1))

    vals = []
    base = float(start_value)
    cur = d0
    # seasonality: 30-day-ish cycle (sinusoid) + light noise
    for i in range(points):
        # seasonal multiplier (±10%)
        seasonal = 1.0 + 0.10 * math.sin(2 * math.pi * (i / 30.0))
        # small random noise (±3%)
        noise = 1.0 + random.uniform(-0.03, 0.03)
        value = base * seasonal * noise
        vals.append({"date": _format_date(cur), "value": round(max(0.0, value), 2)})

        # grow slightly for next point
        base *= (1.0 + step_growth)
        cur = cur + timedelta(days=step_days)

    # pin the last point exactly on date_to
    if vals[-1]["date"] != _format_date(d1):
        vals[-1]["date"] = _format_date(d1)
    return vals



# -------------------------
# COVID (disease.sh / JHU) — daily new cases
# -------------------------
def fetch_covid_timeseries(country: str, date_from: str, date_to: str) -> Optional[List[dict]]:
    country = (country or "").strip() or "World"
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays=all"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    timeline = None
    if isinstance(data, dict):
        if "timeline" in data:
            timeline = data.get("timeline") or {}
        else:
            if "cases" in data and isinstance(data["cases"], dict):
                timeline = {"cases": data["cases"]}
    if not timeline or "cases" not in timeline or not isinstance(timeline["cases"], dict):
        return None

    cumulative = []
    for k, v in timeline["cases"].items():
        try:
            dt = datetime.strptime(k, "%m/%d/%y")
            cumulative.append((dt, float(v or 0)))
        except Exception:
            continue
    if not cumulative:
        return None
    cumulative.sort(key=lambda x: x[0])

    daily = []
    prev = None
    for dt, val in cumulative:
        if prev is None:
            daily_val = 0.0
        else:
            daily_val = max(0.0, val - prev)
        daily.append((dt, daily_val))
        prev = val

    d0 = _parse_date(date_from)
    d1 = _parse_date(date_to)
    if d1 < d0:
        d1 = d0

    daily = [(dt, val) for dt, val in daily if d0 <= dt <= d1]
    if not daily:
        return None

    values = [v for _, v in daily]
    if not any(v > 0 for v in values):
        return None

    return [{"date": _format_date(dt), "value": round(val, 2)} for dt, val in daily]


# -------------------------
# Interpolation helpers (for yearly → finer resolution)
# -------------------------
def _interpolate_yearly_to_monthly(year_values: Dict[int, float], d0: datetime, d1: datetime) -> List[dict]:
    """
    Linear interpolation between yearly anchors; output points at each month boundary.
    """
    if not year_values:
        return []

    # Build sorted anchors
    anchors = sorted((y, float(v)) for y, v in year_values.items())
    years = [y for y, _ in anchors]
    vals = [v for _, v in anchors]

    # Helper to get value at an arbitrary date by linear interpolation across year anchors
    def _value_at(dt: datetime) -> float:
        y = dt.year
        if y <= years[0]:
            return vals[0]
        if y >= years[-1]:
            return vals[-1]
        # find segment
        for i in range(1, len(years)):
            if y <= years[i]:
                y0, v0 = years[i - 1], vals[i - 1]
                y1, v1 = years[i], vals[i]
                # fraction within the year span using month+day rough fraction
                total_months = (y1 - y0) * 12
                months_since = (y - y0) * 12 + (dt.month - 1)
                frac = 0.0 if total_months == 0 else months_since / total_months
                return v0 + (v1 - v0) * max(0.0, min(1.0, frac))
        return vals[-1]

    # Generate month-by-month
    points = []
    cur = datetime(d0.year, d0.month, 1)
    end = datetime(d1.year, d1.month, 1)
    while cur <= end:
        points.append({"date": _format_date(cur), "value": round(_value_at(cur), 2)})
        # next month
        y, m = cur.year, cur.month
        if m == 12:
            cur = datetime(y + 1, 1, 1)
        else:
            cur = datetime(y, m + 1, 1)

    # Ensure last point at d1 exactly
    if points and points[-1]["date"] != _format_date(d1):
        points[-1]["date"] = _format_date(d1)
    return points


# -------------------------
# Malaria (World Bank: SH.MLR.INCD.P3) — yearly incidence per 1,000 at risk
# -------------------------
def fetch_malaria_timeseries(country: str, date_from: str, date_to: str) -> Optional[List[dict]]:
    iso3 = to_iso3(country)
    indicator = "SH.MLR.INCD.P3"
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json&per_page=70"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = data[1] if isinstance(data, list) and len(data) > 1 else []
    except Exception:
        return None

    year_values: Dict[int, float] = {}
    for row in rows:
        try:
            y = int(row.get("date"))
            v = row.get("value")
            if v is None:
                continue
            year_values[y] = float(v)
        except Exception:
            continue

    if not year_values:
        return None

    d0 = _parse_date(date_from)
    d1 = _parse_date(date_to)
    if d1 < d0:
        d1 = d0
    return _interpolate_yearly_to_monthly(year_values, d0, d1)


# -------------------------
# Dengue (WHO GHO: DENGUE_CASES) — yearly cases
# -------------------------
def fetch_dengue_timeseries(country: str, date_from: str, date_to: str) -> Optional[List[dict]]:
    """
    WHO GHO API example:
      https://ghoapi.azureedge.net/ghoapi/api/DENGUE_CASES?$filter=SpatialDim eq 'LKA'
    """
    iso3 = to_iso3(country)
    # Some countries might not exist in GHO or use WLD for global
    flt = f"SpatialDim%20eq%20%27{iso3}%27"
    url = f"https://ghoapi.azureedge.net/ghoapi/api/DENGUE_CASES?$filter={flt}"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = data.get("value", []) if isinstance(data, dict) else []
    except Exception:
        return None

    year_values: Dict[int, float] = {}
    for row in rows:
        try:
            y = int(row.get("TimeDim"))
            v = row.get("NumericValue")
            if v is None:
                continue
            year_values[y] = float(v)
        except Exception:
            continue

    if not year_values:
        return None

    d0 = _parse_date(date_from)
    d1 = _parse_date(date_to)
    if d1 < d0:
        d1 = d0
    return _interpolate_yearly_to_monthly(year_values, d0, d1)
