# backend/retrieval_agent/adapters/cdc_flu.py
from __future__ import annotations
from typing import Any, Dict, List
from datetime import date, timedelta
import random

from .base import FactsAdapter

def _synthetic_ili_series(weeks: int = 26) -> List[Dict[str, Any]]:
    """Generate a synthetic but non-flat ILI (%) time-series."""
    today = date.today()
    series: List[Dict[str, Any]] = []
    val = 2.0  # start around 2% ILI
    for i in range(weeks):
        d = today - timedelta(weeks=(weeks - i))
        # add some random walk to avoid flat line
        val += random.uniform(-0.35, 0.35)
        val = max(0.0, round(val, 2))
        series.append({"date": d.isoformat(), "value": val})
    return series

class CDCFluAdapter(FactsAdapter):
    """Flu adapter without external deps (works on Python 3.13)."""

    def supports(self, query: Dict[str, Any]) -> bool:
        q = (query.get("question") or "").lower()
        disease = (query.get("disease") or "").lower()
        # Trigger on flu-related words
        return (
            "flu" in q or "influenza" in q or "ili" in q or
            "flu" in disease or "influenza" in disease
        )

    def fetch(self, query: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a synthetic ILI trend
        series = _synthetic_ili_series(weeks=26)
        # compute start→end change
        start = series[0]["value"]
        end = series[-1]["value"]
        pct = 0.0 if start == 0 else ((end - start) / start) * 100.0

        facts = {
            "topic": "us_flu_ili",
            "unit": "% ILI",
            "series": series,
            "start": start,
            "end": end,
            "change_pct": round(pct, 1),
            "note": "Synthetic ILI series (no external dependency)."
        }

        return {
            "query": query,
            "facts": facts,
            "sources": [
                {
                    "name": "CDC FluView (synthetic stub)",
                    "url": "https://www.cdc.gov/flu/weekly/fluviewinteractive.htm",
                }
            ],
        }
