from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os
import pandas as pd

# Re-use your existing ISO helpers
from .timeseries import to_iso3

# Default location (can be overridden by env)
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "timeseries_monthly.csv"

def _csv_path() -> Path:
    env_path = os.getenv("LOCAL_DATA_CSV", "").strip()
    return Path(env_path) if env_path else DEFAULT_CSV

def fetch_local_timeseries(
    disease: str,
    region: str,
    date_from: str,
    date_to: str,
) -> Optional[List[dict]]:
    """
    Load monthly data for (disease, region, date-range) from the merged CSV.
    Returns a list of {date, value} dicts or None if nothing found.
    """
    csv_path = _csv_path()
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Normalize filters
    dis = (disease or "").strip().lower()
    iso3 = to_iso3(region or "")

    # Filter rows
    q = df[
        (df["disease"].str.lower() == dis)
        & (df["country_iso3"].str.upper() == iso3)
        & (df["date"] >= date_from)
        & (df["date"] <= date_to)
    ].copy()

    if q.empty:
        return None

    # Keep only columns we need, sort, and format
    q = q[["date", "value"]].sort_values("date")
    # Coerce numeric and drop NaNs
    q["value"] = pd.to_numeric(q["value"], errors="coerce")
    q = q.dropna(subset=["value"])

    if q.empty:
        return None

    out = [{"date": d, "value": float(v)} for d, v in q.itertuples(index=False)]
    return out or None
