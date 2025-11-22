# backend/app/agents/report_generator/services/training.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# Paths
MODULE_DIR = Path(__file__).parent.resolve()
DATA_PATH = (MODULE_DIR.parent / "data" / "timeseries_monthly.csv").resolve()
MODELS_DIR = (MODULE_DIR.parent / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# region mapping helper from your project (same one forecast.py uses)
from .timeseries import to_iso3


def _key(disease: str, region: str) -> str:
    return f"{(disease or '').lower()}__{(region or '').lower().replace(' ', '-')}"


def model_path(disease: str, region: str) -> Path:
    return MODELS_DIR / f"{_key(disease, region)}.pkl"


def _canonical_region_keys(user_region: str) -> List[str]:
    """
    Generate candidate keys we can test against the CSV:
      - ISO3 in UPPER (e.g., 'LKA', 'IND', 'WLD')
      - normalized lowercase names (e.g., 'sri lanka', 'india', 'world')
    """
    region_l = (user_region or "").strip().lower()
    iso3 = (to_iso3(user_region) or "").upper()

    # common aliases for 'world'
    aliases = set([region_l])
    if region_l in {"world", "global"}:
        aliases.update({"world", "global"})

    keys = []
    if iso3:
        keys.append(iso3)  # for country_iso3 column
    keys.extend(sorted(aliases))
    return keys


def load_series(disease: str, region: str) -> Optional[pd.Series]:
    """
    Load monthly series for (disease, region) from the merged CSV.
    Matches by country ISO3 when available; falls back to text columns.
    Returns a monthly-start (MS) Series indexed by Timestamp with float values.
    """
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    # Ensure expected columns exist
    for col in ("disease", "date", "value", "country_iso3", "region", "country", "country_name"):
        if col not in df.columns:
            df[col] = None

    disease_l = (disease or "").strip().lower()
    region_txt = (region or "").strip()
    # Prefer ISO3 matching: resolve region text to ISO3 (e.g., "Sri Lanka" -> "LKA")
    try:
        from .timeseries import to_iso3  # local helper that knows country names
        target_iso3 = (to_iso3(region_txt) or "").upper()
    except Exception:
        target_iso3 = ""

    df["disease_l"] = df["disease"].astype(str).str.strip().str.lower()

    # Build a normalized region string we can fallback to if iso3 missing
    # choose the first present text column among region/country/country_name
    name_col = "region" if "region" in df.columns and df["region"].notna().any() else \
               "country" if "country" in df.columns and df["country"].notna().any() else \
               "country_name" if "country_name" in df.columns and df["country_name"].notna().any() else None
    if name_col:
        df["region_l"] = df[name_col].astype(str).str.strip().str.lower()
    else:
        df["region_l"] = None

    # Filter by disease
    sub = df[df["disease_l"] == disease_l].copy()

    # Primary: ISO3
    if target_iso3 and "country_iso3" in sub.columns and sub["country_iso3"].notna().any():
        sub["iso3"] = sub["country_iso3"].astype(str).str.upper()
        sub = sub[sub["iso3"] == target_iso3]

    # Fallback: text region
    if sub.empty and region_txt:
        sub = df[(df["disease_l"] == disease_l) &
                 (df["region_l"] == region_txt.strip().lower())].copy()

    if sub.empty:
        return None

    # Parse dates & build monthly Series
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"])
    sub = sub.sort_values("date").set_index("date")

    s = pd.to_numeric(sub["value"], errors="coerce")
    # resample to month-start; if multiple values in a month, take mean
    s = s.resample("MS").mean()
    s = s.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Require some length / signal (18 months is a decent minimum for seasonal models)
    if s.isna().all() or (len(s) < 18):
        return None

    return s


def train_model(disease: str, region: str) -> Tuple[Optional[object], str]:
    """
    Fit a simple seasonal SARIMAX and persist it. Returns (result, status).
    status in {"ok","not_enough_data","fit_error:..."}.
    """
    s = load_series(disease, region)
    if s is None:
        return None, "not_enough_data"

    try:
        model = SARIMAX(
            s,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        joblib.dump(res, model_path(disease, region))
        return res, "ok"
    except Exception as e:
        return None, f"fit_error:{e}"


def load_model(disease: str, region: str) -> Optional[object]:
    p = model_path(disease, region)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def predict_single_month(disease: str, region: str, target_ym: str) -> Tuple[Optional[float], str]:
    """
    Predict a single YYYY-MM value.
    If model missing, tries to train from local CSV.
    Returns (yhat, status) where status in {"forecast", "in_sample", "not_enough_data", "fit_error:..."}.
    """
    res = load_model(disease, region)
    if res is None:
        res, st = train_model(disease, region)
        if res is None:
            return None, st  # propagate status

    # Determine steps ahead from last seen month
    idx = res.model.data.row_labels  # DatetimeIndex (monthly)
    last = pd.to_datetime(idx[-1])
    target = pd.to_datetime(f"{target_ym}-01")

    if target <= last:
        pred = res.get_prediction(start=target, end=target)
        yhat = float(pred.predicted_mean.iloc[0])
        return yhat, "in_sample"

    steps = (target.year - last.year) * 12 + (target.month - last.month)
    steps = max(1, steps)
    fc = res.get_forecast(steps=steps)
    yhat = float(fc.predicted_mean.iloc[-1])
    return yhat, "forecast"
