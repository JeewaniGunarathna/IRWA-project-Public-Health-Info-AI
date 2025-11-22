from typing import List, Dict, Optional
from datetime import datetime

def _fmt_date(s: str) -> str:
    # "2025-08-24" → "24 Aug 2025"
    return datetime.fromisoformat(s).strftime("%d %b %Y")

def _calc_change(timeseries: List[Dict]) -> Optional[float]:
    if not timeseries or len(timeseries) < 2:
        return None
    ts = sorted(timeseries, key=lambda x: x["date"])
    start = float(ts[0]["value"])
    end = float(ts[-1]["value"])
    if start == 0:
        return None
    return round(((end - start) / start) * 100.0, 1)

# def generate_summary(
#     disease: str,
#     region: str,
#     date_from: str,
#     date_to: str,
#     timeseries: List[Dict],
#     insights: Dict | None,
# ) -> str:
#     """Returns a concise, neutral summary (2–5 sentences)."""
#     period = f"{_fmt_date(date_from)} to {_fmt_date(date_to)}"

#     # Prefer provided insight; fall back to computed change
#     provided = (insights or {}).get("weekly_increase")
#     change = provided or _calc_change(timeseries)

#     s1 = f"In {region}, {disease} trends were analyzed from {period}."
#     if isinstance(change, str):
#         s2 = f"Reported change over the period is {change}."
#     elif isinstance(change, (int, float)):
#         direction = "increased" if change >= 0 else "decreased"
#         s2 = f"Cases {direction} by {abs(change)}% over the selected period."
#     else:
#         s2 = "Insufficient data to compute percentage change."

#     top_region = (insights or {}).get("top_region")
#     s3 = f"The highest recent activity was observed in {top_region}." if top_region else ""

#     note = "Figures are based on supplied inputs; please verify with the cited sources."
#     # Join non‑empty sentences with spaces
#     return " ".join([x for x in [s1, s2, s3, note] if x])

# agents/report_generator/services/summarizer.py
def generate_summary(disease, region, date_from, date_to, timeseries=None, insights=None) -> str:
    name = (disease or "").upper().replace("-", " ")
    loc  = (region or "the selected region")

    # Prefer % from series; fallback to insights if present
    pct_text = ""
    if timeseries and len(timeseries) >= 2:
        start = float(timeseries[0].get("value") or 0.0)
        end   = float(timeseries[-1].get("value") or 0.0)
        pct   = 0.0 if start == 0 else ((end - start) / start) * 100.0
        pct_text = f"{pct:.1f}%"
    elif insights and insights.get("weekly_increase"):
        pct_text = insights["weekly_increase"]

    change_part = f"Reported change over the period is {pct_text}." if pct_text else "Change over the period is not available."

    return (
        f"In {loc}, {name} trends were analyzed from {date_from} to {date_to}. "
        f"{change_part} Figures are based on supplied inputs; please verify with the cited sources."
    )



def _pct_change_from_timeseries(timeseries: list[dict]) -> float | None:
    if not timeseries or len(timeseries) < 2:
        return None
    vals = [float(p.get("value", 0)) for p in timeseries if p.get("value") is not None]
    vals = [v for v in vals if v >= 0]
    if len(vals) < 2:
        return None
    start, end = vals[0], vals[-1]
    if start == 0:
        return None
    return ((end - start) / start) * 100.0
