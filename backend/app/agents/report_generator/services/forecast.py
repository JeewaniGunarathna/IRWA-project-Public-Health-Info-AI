from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import math

import pandas as pd

# Try to import statsmodels (works on Python 3.13 with 0.14.4)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# existing helpers from your agent
from .timeseries import to_iso3
from .timeseries import fetch_covid_timeseries, generate_synthetic_timeseries  # live + synthetic
from .local_data import fetch_local_timeseries  # CSV fallback


@dataclass
class ForecastResult:
    history: List[Dict]       # [{date, value}]
    forecast: List[Dict]      # [{date, yhat, yhat_lower, yhat_upper}]
    method: str               # "sarimax" | "naive_seasonal" | "moving_avg" | "synthetic"
    provenance: List[str]     # list of strings describing sources used
    warnings: List[str]       # any notes to show to user


# --------------------- small internal helpers ---------------------

def _is_flat(values: List[float]) -> bool:
    if not values or len(values) < 2:
        return True
    return (max(values) - min(values)) < 1e-9


def _to_monthly_df(series: List[Dict]) -> pd.DataFrame:
    """
    Ensure a clean monthly index DataFrame with columns ['value'].
    Accepts daily or monthly input; resamples daily → monthly sum.
    """
    if not series:
        return pd.DataFrame(columns=["value"])

    df = pd.DataFrame(series).copy()
    # normalize keys and dtypes
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    # detect frequency (daily vs monthly)
    # if many unique days in a month → treat as daily and resample to monthly sum
    if len(df) >= 28 and df["date"].dt.day.nunique() > 5:
        monthly = (
            df.set_index("date")
              .resample("MS")  # Month start
              .sum(numeric_only=True)
              .rename(columns={"value": "value"})
        )
    else:
        # already monthly-ish: align to month-start
        df["date"] = df["date"].values.astype("datetime64[M]")
        monthly = (
            df.groupby("date")["value"]
              .sum()  # sum if duplicates
              .to_frame()
        )

    monthly = monthly.sort_index()
    monthly.index.name = "date"
    return monthly


def _make_future_month_index(last_month: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    return pd.date_range((last_month + pd.offsets.MonthBegin(1)), periods=steps, freq="MS")


# --------------------- region normalization ---------------------

_ALIAS_TO_ISO3 = {
    "world": "WLD",
    "global": "WLD",
    "usa": "USA",
    "united states": "USA",
}

def _region_candidates(region: str) -> List[str]:
    """
    Given a human input like 'Sri Lanka', produce a robust list of candidates
    to try against the local CSV. Your CSV uses ISO3 (e.g., LKA, IND, WLD).
    """
    s = (region or "").strip()
    if not s:
        return []

    cands: List[str] = []

    # 1) exact user input (some datasets might store country names)
    cands.append(s)

    # 2) alias map (world/global/usa → WLD/USA)
    alias = _ALIAS_TO_ISO3.get(s.lower())
    if alias:
        cands.append(alias)

    # 3) to_iso3 from your helper (e.g., 'Sri Lanka' → 'LKA')
    try:
        iso = to_iso3(s)
        if iso and iso.upper() not in cands:
            cands.append(iso.upper())
    except Exception:
        pass

    # 4) If user already typed ISO3-ish, keep the uppercased form
    if len(s) == 3 and s.isalpha():
        up = s.upper()
        if up not in cands:
            cands.append(up)

    # Ensure uniqueness while preserving order
    seen = set()
    uniq: List[str] = []
    for x in cands:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# --------------------- data retrieval with provenance ---------------------

def get_monthly_history(
    disease: str,
    region: str,
    date_from: str,
    date_to: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Try live → local → synthetic, return monthly df, provenance notes, warnings.
    """
    disease_l = (disease or "").lower().strip()
    provenance: List[str] = []
    warnings: List[str] = []

    # 1) live where possible (COVID)
    series: Optional[List[Dict]] = None
    try:
        if "covid" in disease_l:
            # disease.sh expects human country names — keep the raw region
            series = fetch_covid_timeseries(region, date_from, date_to)
            if series:
                provenance.append("live: disease.sh (JHU)")
    except Exception:
        series = None

    # 2) local monthly CSV fallback
    #    Your CSV stores 'country_iso3' (LKA, IND, WLD).
    if not series or len(series) < 2 or _is_flat([float(x.get("value", 0.0)) for x in series]):
        got_local = False
        tried_regions = _region_candidates(region)
        if not tried_regions:
            tried_regions = [region]  # very defensive

        for r in tried_regions:
            try:
                series_local = fetch_local_timeseries(disease_l, r, date_from, date_to)
            except Exception:
                series_local = []

            if series_local:
                series = series_local
                provenance.append(f"local: monthly dataset (region key='{r}')")
                got_local = True
                break

        if not got_local:
            warnings.append("No usable live or local data in the requested window.")

    # 3) synthetic as a last resort (also used when series is too flat)
    if not series or len(series) < 2 or _is_flat([float(x.get("value", 0.0)) for x in series]):
        # Build a simple monthly synthetic sequence
        synth = generate_synthetic_timeseries(
            date_from=date_from,
            date_to=date_to,
            weekly_increase_str=None,
            points=12,
            start_value=100.0,
        )
        series = synth
        provenance.append("synthetic: seasonal baseline (no/flat data)")

    monthly = _to_monthly_df(series)
    return monthly, provenance, warnings


# --------------------- forecasters (no heavy offline training) ---------------------

def _sarimax_forecast(y: pd.Series, steps: int) -> Tuple[pd.Series, pd.DataFrame, str]:
    """
    SARIMAX if available and series is long enough and not flat.
    """
    if not _HAS_SM:
        raise RuntimeError("statsmodels unavailable")
    if len(y) < 24 or _is_flat(y.tolist()):
        raise RuntimeError("series too short/flat for SARIMAX")

    seasonal = 12 if len(y) >= 24 else 0
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, seasonal) if seasonal else (0, 0, 0, 0)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    f = res.get_forecast(steps=steps)
    mean = f.predicted_mean
    conf = f.conf_int(alpha=0.2)  # 80% band for tighter visuals
    method = "sarimax"
    return mean, conf, method


def _naive_seasonal_forecast(y: pd.Series, steps: int) -> Tuple[pd.Series, pd.DataFrame, str]:
    """
    Repeat last 12-month pattern if we have ≥12 points; simple conf band via residual std.
    """
    if len(y) < 12:
        raise RuntimeError("series too short for seasonal naive")
    period = 12
    hist = y.copy()
    future_vals = []
    for i in range(steps):
        future_vals.append(hist.iloc[-period + (i % period)])
    mean = pd.Series(future_vals, index=_make_future_month_index(y.index[-1], steps))

    # estimate residual variance with 12-lag naive
    try:
        resids = hist.iloc[period:] - hist.shift(period).dropna()
        sd = float(resids.std(ddof=1)) if len(resids) >= 3 else float(hist.std(ddof=1))
    except Exception:
        sd = float(hist.std(ddof=1))
    band = 1.28 * sd  # ~80%
    conf = pd.DataFrame({"lower value": mean - band, "upper value": mean + band}, index=mean.index)
    return mean, conf, "naive_seasonal"


def _moving_avg_forecast(y: pd.Series, steps: int) -> Tuple[pd.Series, pd.DataFrame, str]:
    """
    Very safe baseline for short series: continue last MA with wide band.
    """
    if len(y) < 2:
        raise RuntimeError("series too short for moving average")
    w = min(6, len(y))
    avg = float(y.tail(w).mean())
    mean = pd.Series([avg] * steps, index=_make_future_month_index(y.index[-1], steps))

    sd = float(y.tail(w).std(ddof=1)) if w >= 3 else float(y.std(ddof=1))
    band = 1.64 * (sd if not math.isnan(sd) else 0.15 * (avg + 1.0))  # ~90%
    conf = pd.DataFrame({"lower value": mean - band, "upper value": mean + band}, index=mean.index)
    return mean, conf, "moving_avg"


# --------------------- main API used by /forecast ---------------------

def forecast_monthly(
    disease: str,
    region: str,
    date_from: str,
    date_to: str,
    horizon_months: int = 6,
) -> ForecastResult:
    """
    Obtain/prepare monthly history (live→local→synthetic), then forecast with
    SARIMAX → seasonal naive → moving average → synthetic (as last resort).
    """
    # 1) get history
    hist_df, provenance, warnings = get_monthly_history(disease, region, date_from, date_to)
    if hist_df.empty:
        # Full synthetic path
        synth = generate_synthetic_timeseries(date_from, date_to, points=12, start_value=100.0)
        hist_df = _to_monthly_df(synth)
        provenance.append("synthetic: seasonal baseline (history empty)")
        warnings.append("Empty history; using synthetic baseline.")

    y = hist_df["value"].astype(float)
    method = "synthetic"
    mean, conf = None, None

    # 2) choose a method
    tried = []

    try:
        mean, conf, method = _sarimax_forecast(y, horizon_months)
    except Exception as e:
        tried.append(f"sarimax:{e}")

    if mean is None:
        try:
            mean, conf, method = _naive_seasonal_forecast(y, horizon_months)
        except Exception as e:
            tried.append(f"naive:{e}")

    if mean is None:
        try:
            mean, conf, method = _moving_avg_forecast(y, horizon_months)
        except Exception as e:
            tried.append(f"moving_avg:{e}")

    if mean is None:
        # final synthetic: extend with smooth trend from last value
        base_start = y.iloc[-1] if len(y) else 100.0
        synth_out = []
        cur = float(base_start)
        for i in range(horizon_months):
            # small seasonality wiggle
            cur = cur * (1.0 + 0.02 * math.sin((i % 12) / 12.0 * 2 * math.pi))
            synth_out.append(cur)
        idx = _make_future_month_index(y.index[-1] if len(y) else pd.Timestamp(date_from), horizon_months)
        mean = pd.Series(synth_out, index=idx)
        band = max(0.2 * float(base_start), 10.0)
        conf = pd.DataFrame({"lower value": mean - band, "upper value": mean + band}, index=idx)
        method = "synthetic"
        warnings.append("Forecast produced from synthetic extension (no usable model).")

    # 3) format response
    history = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in y.items()]
    forecast = []
    for d, yhat in mean.items():
        lo = float(conf.loc[d, conf.columns[0]])
        hi = float(conf.loc[d, conf.columns[1]])
        forecast.append({
            "date": d.strftime("%Y-%m-%d"),
            "yhat": float(yhat),
            "yhat_lower": float(max(0.0, lo)),
            "yhat_upper": float(max(0.0, hi)),
        })

    return ForecastResult(
        history=history,
        forecast=forecast,
        method=method,
        provenance=provenance,
        warnings=warnings,
    )
