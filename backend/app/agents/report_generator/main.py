import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from enum import Enum
from fastapi import Body
from dateutil import parser as dtparse
from .services.local_data import fetch_local_timeseries
from .services.forecast import forecast_monthly

from .services.training import train_model, predict_single_month

load_dotenv()

# Fallback toggles (default ON)
USE_LOCAL_DATA = os.getenv("USE_LOCAL_DATA", "true").lower() in ("1","true","yes")
USE_SYNTHETIC_FALLBACK = os.getenv("USE_SYNTHETIC_FALLBACK", "true").lower() in ("1","true","yes")

# ---------- Internal services ----------
from .services.nlp import extract as extract_from_text
from .services.visualize import make_line_chart
from .services.summarizer import generate_summary
from .services.render import render_html_report, html_to_pdf_strict
from .services.timeseries import (
    generate_synthetic_timeseries,
    fetch_covid_timeseries,
    fetch_malaria_timeseries,     
    fetch_dengue_timeseries,      
)

# ---------- Paths / Config ----------
MODULE_DIR = Path(__file__).parent.resolve()

ARTIFACTS_ROOT = (MODULE_DIR / "artifacts").resolve()
CHARTS_DIR     = str((ARTIFACTS_ROOT / "charts").resolve())
REPORTS_DIR    = str((ARTIFACTS_ROOT / "reports").resolve())
FORECASTS_DIR = str((ARTIFACTS_ROOT / "forecasts").resolve())
Path(FORECASTS_DIR).mkdir(parents=True, exist_ok=True)

# CSS inside this agent
CSS_PATH = str((MODULE_DIR / "static" / "css" / "style.css").resolve())

# wkhtmltopdf (use .env if provided)
WKHTMLTOPDF = os.getenv(
    "WKHTMLTOPDF",
    r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)

# Ensure folders exist
Path(CHARTS_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


# ---------- Schemas ----------
class Source(BaseModel):
    name: str
    url: str
    date: Optional[str] = None

class Insights(BaseModel):
    weekly_increase: Optional[str] = None
    top_region: Optional[str] = None
    anomalies: Optional[List[str]] = None

class ReportRequest(BaseModel):
    disease: str
    region: str
    date_from: str
    date_to: str
    timeseries: Optional[List[dict]] = []
    insights: Optional[Insights] = None
    sources: Optional[List[Source]] = []

class ReportResponse(BaseModel):
    summary: str
    visuals: List[dict]
    report_url: Optional[str] = None
    pdf_url: Optional[str] = None
    disclaimer: str
    sources: List[Source]

class TextRequest(BaseModel):
    query: str

class DiseaseEnum(str, Enum):
    covid = "covid"
    dengue = "dengue"
    malaria = "malaria"
    influenza = "influenza"
    tb = "tuberculosis"
    hiv = "hiv"
    measles = "measles"

class MetricEnum(str, Enum):
    cases = "cases"
    deaths = "deaths"
    incidence = "incidence"   # placeholder for non-covid diseases, if needed

class AggregateEnum(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"

# === Structured "form" request ===
class ReportForm(BaseModel):
    disease: DiseaseEnum
    region: str
    date_from: str          # YYYY-MM-DD
    date_to: str            # YYYY-MM-DD
    metric: MetricEnum = MetricEnum.cases
    aggregate: AggregateEnum = AggregateEnum.daily
    include_pdf: bool = True

class ForecastForm(BaseModel):
    disease: DiseaseEnum
    region: str
    date_from: str
    date_to: str
    horizon_months: int = 6  # forecast length (months)

class ForecastResponse(BaseModel):
    history: List[dict]
    forecast: List[dict]
    method: str
    provenance: List[str]
    warnings: List[str]
    forecast_url: Optional[str] = None

class TrainForm(BaseModel):
    disease: str
    region: str

class PredictMonthForm(BaseModel):
    disease: str
    region: str
    target_month: str  # "YYYY-MM"

# ---------- App ----------
app = FastAPI(title="Report Generator Agent")

# CORS (Vite on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve artifacts (charts + reports) from this agent's folder
app.mount(
    "/agents/report_generator/artifacts",
    StaticFiles(directory=str(ARTIFACTS_ROOT)),
    name="artifacts",
)

STATIC_DIR = MODULE_DIR / "static"
app.mount(
    "/agents/report_generator/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

@app.get("/health")
def health_check():
    from datetime import datetime
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


def _append_source(lst, name, url, date=None):
    from pydantic import ValidationError
    try:
        # keep it consistent with your Pydantic Source schema
        lst.append(Source(name=name, url=url, date=date))
    except ValidationError:
        # very defensive: if we ever pass a bad shape
        lst.append(Source(name=name, url=str(url), date=date))


# ---------- Helpers ----------
def _public_chart_path_or_fix(disease: str, region: str, date_from: str, date_to: str, suggested_public_path: str) -> str:
    """
    Ensure the chart URL we return actually points to a file that exists.
    If the suggested path doesn't exist (e.g., because of a '-from_' slug mismatch),
    try the canonical filename pattern without '-from'.
    """
    charts_fs = Path(CHARTS_DIR)
    artifacts_prefix = "/agents/report_generator/artifacts/charts"

    # 1) If suggested path exists, use it
    basename = Path(suggested_public_path).name
    fs_path = charts_fs / basename
    if fs_path.exists():
        return f"{artifacts_prefix}/{basename}"

    # 2) Try a canonical filename (no '-from_')
    canonical_name = f"{slugify(disease)}_{slugify(region)}_{date_from.replace('-','')}_{date_to.replace('-','')}.png"
    canonical_fs = charts_fs / canonical_name
    if canonical_fs.exists():
        return f"{artifacts_prefix}/{canonical_name}"

    # 3) If nothing found, fall back to dummy
    dummy = charts_fs / "dummy_chart.png"
    if dummy.exists():
        return f"{artifacts_prefix}/dummy_chart.png"

    # 4) Last resort: return the original (may 404, but we tried)
    return suggested_public_path


def _has_variation(series: List[dict]) -> bool:
    """Return True if there is meaningful variance in the values."""
    if not series or len(series) < 2:
        return False
    vals = [float(p.get("value") or 0.0) for p in series]
    return max(vals) - min(vals) > 1e-6  # anything more than perfectly flat

@app.post("/report", response_model=ReportResponse)
def generate_report(request: ReportRequest):
    """
    Structured entry point. If the caller omits `timeseries`, we try to fetch
    real data based on the disease & region. If missing or flat, synthesize and
    clearly note that in sources.
    """
    disease_l = (request.disease or "").lower().strip()
    region = request.region

    # Ensure sources list
    sources_list: List[Source] = list(request.sources or [])

    # 1) Resolve timeseries: use provided or try to fetch
    ts = request.timeseries or []
    attempted_sources: List[Source] = []

    if not ts:
        real_series = None

        # ---- 1) Try live sources first -------------------------------------------------
        if "covid" in disease_l:
            real_series = fetch_covid_timeseries(region, request.date_from, request.date_to)
            attempted_sources.append(Source(
                name="disease.sh (JHU)",
                url="https://disease.sh/docs/#/COVID-19%3A%20JHUCSSE/get_v3_covid_19_historical__country_",
                date=None,
            ))

        elif "dengue" in disease_l:
            from .services.timeseries import fetch_dengue_timeseries
            real_series = fetch_dengue_timeseries(region, request.date_from, request.date_to)
            attempted_sources.append(Source(
                name="WHO GHO — DENGUE_CASES",
                url="https://ghoapi.azureedge.net/ghoapi/api/DENGUE_CASES",
                date=None,
            ))

        elif "malaria" in disease_l:
            from .services.timeseries import fetch_malaria_timeseries
            real_series = fetch_malaria_timeseries(region, request.date_from, request.date_to)
            attempted_sources.append(Source(
                name="World Bank — SH.MLR.INCD.P3",
                url="https://data.worldbank.org/indicator/SH.MLR.INCD.P3",
                date=None,
            ))

        if real_series:
            ts = real_series

        # ---- 2) Local CSV fallback (only if live returned nothing) ---------------------
        if not ts and USE_LOCAL_DATA:
            local = fetch_local_timeseries(
                disease=request.disease,
                region=request.region,
                date_from=request.date_from,
                date_to=request.date_to,
            )
            if local:
                ts = local
                # mark the source clearly
                try:
                    attempted_sources.append(Source(
                        name="Local dataset (monthly)",
                        url="file://data/timeseries_monthly.csv",
                        date=None,
                    ))
                except Exception:
                    # ultra-defensive: never crash here
                    pass


    # 2) If still empty or effectively flat → synthesize & label clearly
    synthesized = False
    if (not ts or not _has_variation(ts)) and USE_SYNTHETIC_FALLBACK:
        ts = generate_synthetic_timeseries(
            date_from=request.date_from,
            date_to=request.date_to,
            weekly_increase_str=None,
            points=60,
            start_value=100.0,
        )
        synthesized = True


    # Update request mutable fields
    request.timeseries = ts

    # Sources: include the attempted real source (if any), and add a synthetic note when used
    if attempted_sources:
        for s in attempted_sources:
            sources_list.append(s)
    if synthesized:
        sources_list.append(Source(
            name="Synthetic (no/insufficient data in selected window)",
            url="about:blank",
            date=None,
        ))
    request.sources = sources_list

    # 3) Summary
    summary = generate_summary(
        disease=request.disease,
        region=request.region,
        date_from=request.date_from,
        date_to=request.date_to,
        timeseries=request.timeseries or [],
        insights=request.insights.dict() if request.insights else None,
    )

    # 3b) If insights.weekly_increase absent, compute % change first→last (informational)
    if request.timeseries and request.insights is not None and not request.insights.weekly_increase:
        try:
            start_val = float(request.timeseries[0].get("value") or 0.0)
            end_val = float(request.timeseries[-1].get("value") or 0.0)
            pct = 0.0 if start_val == 0 else ((end_val - start_val) / start_val) * 100.0
            request.insights.weekly_increase = f"{pct:.1f}%"
        except Exception:
            pass

    # 4) Chart
    if request.timeseries:
        raw_chart_path = make_line_chart(
            timeseries=request.timeseries,
            disease=request.disease,
            region=request.region,
            date_from=request.date_from,
            date_to=request.date_to,
            charts_dir=CHARTS_DIR,
        )
        chart_public = _public_chart_path_or_fix(
            request.disease, request.region, request.date_from, request.date_to, raw_chart_path
        )
    else:
        chart_public = "/agents/report_generator/artifacts/charts/dummy_chart.png"

    visuals = [{"type": "line", "path": chart_public}]

    # 5) HTML
    slug = f"{slugify(request.disease)}_{slugify(request.region)}_{request.date_from.replace('-','')}_{request.date_to.replace('-','')}"
    period = f"{request.date_from} to {request.date_to}"

    report_rel_url = render_html_report(
        title=f"{request.disease} — {request.region}",
        period=period,
        summary=summary,
        chart_rel_path=chart_public,
        sources=[s.dict() for s in (request.sources or [])],
        disclaimer="Auto-generated demo. Verify with official sources.",
        reports_dir=REPORTS_DIR,
        templates_dir=str((MODULE_DIR / "templates").resolve()),
        filename_slug=slug,
    )

    # 6) PDF
    report_filename = Path(report_rel_url).name
    html_fs_path = Path(REPORTS_DIR) / report_filename
    pdf_filename = report_filename.replace(".html", ".pdf")
    pdf_fs_path = Path(REPORTS_DIR) / pdf_filename

    chart_abs_path = None
    try:
        chart_abs_path = str((Path(CHARTS_DIR) / Path(chart_public).name).resolve())
    except Exception:
        chart_abs_path = None

    pdf_ok = False
    try:
        pdf_ok = html_to_pdf_strict(
            html_path=str(html_fs_path),
            pdf_out_path=str(pdf_fs_path),
            css_abs_path=CSS_PATH,
            chart_abs_path=chart_abs_path,
            wkhtmltopdf_path=WKHTMLTOPDF,
        )
    except Exception:
        pdf_ok = False

    pdf_rel_url = f"/agents/report_generator/artifacts/reports/{pdf_filename}" if pdf_ok else None

    return ReportResponse(
        summary=summary,
        visuals=visuals,
        report_url=report_rel_url,
        pdf_url=pdf_rel_url,
        disclaimer=(
            "If real data was unavailable or flat for the selected window, the chart uses a synthetic, seasonal trend. "
            "Verify with official sources."
        ),
        sources=request.sources or [],
    )


from .services.datasources import fetch_timeseries_if_possible


@app.post("/report_from_text", response_model=ReportResponse)
def report_from_text(body: TextRequest):
    if not body.query or len(body.query.strip()) < 8:
        raise HTTPException(status_code=400, detail="Please provide a longer prompt.")

    data = extract_from_text(body.query)
    # always start with a clean sources list; we’ll add live/local/synthetic markers below
    data["sources"] = []

    # Ensure insights dict exists
    if not isinstance(data.get("insights"), dict):
        data["insights"] = {}

    disease = (data.get("disease") or "").lower()
    region  = data.get("region") or data.get("country") or "World"
    real_series: Optional[List[dict]] = None

    # ---------- 1) Live sources first ----------
    if "covid" in disease:
        real_series = fetch_covid_timeseries(region, data["date_from"], data["date_to"])
        if real_series and len(real_series) >= 2:
            data["timeseries"] = real_series
            s = float(real_series[0].get("value") or 0.0)
            e = float(real_series[-1].get("value") or 0.0)
            pct = 0.0 if s == 0 else ((e - s) / s) * 100.0
            data["insights"]["weekly_increase"] = f"{pct:.1f}%"
            data["sources"].append({
                "name": "disease.sh (JHU)",
                "url": "https://disease.sh/docs/#/COVID-19%3A%20JHUCSSE/get_v3_covid_19_historical__country_",
                "date": None
            })

    elif "malaria" in disease:
        real_series = fetch_malaria_timeseries(region, data["date_from"], data["date_to"])
        if real_series and len(real_series) >= 2:
            data["timeseries"] = real_series
            s = float(real_series[0].get("value") or 0.0)
            e = float(real_series[-1].get("value") or 0.0)
            pct = 0.0 if s == 0 else ((e - s) / s) * 100.0
            data["insights"]["weekly_increase"] = f"{pct:.1f}%"
            data["sources"].append({
                "name": "World Bank — Malaria incidence (SH.MLR.INCD.P3)",
                "url": "https://data.worldbank.org/indicator/SH.MLR.INCD.P3",
                "date": None
            })

    elif "dengue" in disease:
        real_series = fetch_dengue_timeseries(region, data["date_from"], data["date_to"])
        if real_series and len(real_series) >= 2:
            data["timeseries"] = real_series
            s = float(real_series[0].get("value") or 0.0)
            e = float(real_series[-1].get("value") or 0.0)
            pct = 0.0 if s == 0 else ((e - s) / s) * 100.0
            data["insights"]["weekly_increase"] = f"{pct:.1f}%"
            data["sources"].append({
                "name": "WHO GHO — Dengue",
                "url": "https://ghoapi.azureedge.net/ghoapi/api",
                "date": None
            })

    # ---------- 2) Local CSV fallback (only if no live series) ----------
    if not data.get("timeseries") and USE_LOCAL_DATA:
        local_series = fetch_local_timeseries(
            disease=data.get("disease"),
            region=region,
            date_from=data["date_from"],
            date_to=data["date_to"],
        )
        if local_series and len(local_series) >= 2:
            data["timeseries"] = local_series
            # compute simple % change for the text summary
            s = float(local_series[0].get("value") or 0.0)
            e = float(local_series[-1].get("value") or 0.0)
            if s != 0:
                data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"
            data["sources"].append({
                "name": "Local dataset (monthly)",
                "url": "file://data/timeseries_monthly.csv",
                "date": None
            })
        else:
            # be transparent that we **tried** the local dataset
            data["sources"].append({
                "name": "Local dataset (monthly) — no rows in window",
                "url": "file://data/timeseries_monthly.csv",
                "date": None
            })

    # ---------- 3) Synthetic last resort (only if still empty & allowed) ----------
    if not data.get("timeseries") and USE_SYNTHETIC_FALLBACK:
        data["timeseries"] = generate_synthetic_timeseries(
            date_from=data["date_from"],
            date_to=data["date_to"],
            weekly_increase_str=None,
            points=10,
            start_value=100.0,
        )
        ts = data["timeseries"]
        if len(ts) >= 2:
            s = float(ts[0]["value"] or 0.0)
            e = float(ts[-1]["value"] or 0.0)
            if s != 0:
                data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"

        # Mark clearly as illustrative
        data["sources"].append({"name": "Synthetic (no/insufficient live + local data)", "url": "about:blank", "date": None})

        # Optional provenance hints even when synthetic:
        if "malaria" in disease:
            data["sources"].append({
                "name": "World Bank — Malaria incidence (SH.MLR.INCD.P3)",
                "url": "https://data.worldbank.org/indicator/SH.MLR.INCD.P3",
                "date": None
            })
        if "dengue" in disease:
            data["sources"].append({
                "name": "WHO GHO — Dengue",
                "url": "https://ghoapi.azureedge.net/ghoapi/api",
                "date": None
            })
        if "covid" in disease:
            data["sources"].append({
                "name": "disease.sh (JHU)",
                "url": "https://disease.sh/",
                "date": None
            })

    # ---------- 4) Validate & delegate ----------
    try:
        req = ReportRequest(**data)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

    return generate_report(req)


@app.get("/report_options")
def report_options():
    return {
        "diseases": [d.value for d in DiseaseEnum],
        "metrics": [m.value for m in MetricEnum],
        "aggregations": [a.value for a in AggregateEnum],
        "examples": [
            {"disease": "covid", "region": "Sri Lanka", "date_from": "2021-01-10", "date_to": "2021-02-10"},
            {"disease": "dengue", "region": "India", "date_from": "2023-01-01", "date_to": "2023-04-01"},
        ]
    }    

# === Helper: safe date normalization & guard ===
from datetime import datetime

def _parse_date(s: str) -> datetime:
    # Try strict ISO first, then fall back to flexible parser
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        # Accept DD/MM/YYYY or MM/DD/YYYY etc.
        return dtparse.parse(s, dayfirst=True)

def _guard_dates(date_from: str, date_to: str):
    try:
        d0 = _parse_date(date_from)
        d1 = _parse_date(date_to)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid date format. Use YYYY-MM-DD or a common format like 01/10/2021."
        )
    if d1 < d0:
        raise HTTPException(status_code=422, detail="date_to must be after date_from")
    if (d1 - d0).days > 365 * 5:
        raise HTTPException(status_code=422, detail="Date range too large (limit 5 years).")



# === NEW endpoint: POST /report_form ===
@app.post("/report_form", response_model=ReportResponse)
def report_from_form(form: ReportForm = Body(...)):
    """
    Accepts a structured form payload and returns the same ReportResponse
    your /report endpoint returns, but without needing text parsing.
    Fallback order: Live → Local CSV → Synthetic.
    """
    _guard_dates(form.date_from, form.date_to)

    # Build the internal dict your existing pipeline expects
    data = {
        "disease": form.disease.value,   # e.g. "covid", "dengue", "malaria"
        "region": form.region,
        "date_from": form.date_from,
        "date_to": form.date_to,
        "timeseries": [],
        "insights": {},
        "sources": [],
    }

    disease_l = form.disease.value.lower()
    region = form.region

    # ---------------- 1) LIVE data (when available) ----------------
    real_series: Optional[List[dict]] = None
    try:
        if disease_l == "covid":
            real_series = fetch_covid_timeseries(region, form.date_from, form.date_to)
            if real_series and len(real_series) >= 2:
                data["timeseries"] = real_series
                s = float(real_series[0].get("value") or 0.0)
                e = float(real_series[-1].get("value") or 0.0)
                if s != 0:
                    data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"
                data["sources"].append({
                    "name": "disease.sh (JHU)",
                    "url": "https://disease.sh/docs/#/COVID-19%3A%20JHUCSSE/get_v3_covid_19_historical__country_",
                    "date": None
                })

        elif disease_l == "malaria":
            # Yearly WB → resampled in services.timeseries
            real_series = fetch_malaria_timeseries(region, form.date_from, form.date_to)
            if real_series and len(real_series) >= 2:
                data["timeseries"] = real_series
                s = float(real_series[0].get("value") or 0.0)
                e = float(real_series[-1].get("value") or 0.0)
                if s != 0:
                    data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"
                data["sources"].append({
                    "name": "World Bank — Malaria incidence (SH.MLR.INCD.P3)",
                    "url": "https://data.worldbank.org/indicator/SH.MLR.INCD.P3",
                    "date": None
                })

        elif disease_l == "dengue":
            # Yearly WHO GHO → resampled in services.timeseries
            real_series = fetch_dengue_timeseries(region, form.date_from, form.date_to)
            if real_series and len(real_series) >= 2:
                data["timeseries"] = real_series
                s = float(real_series[0].get("value") or 0.0)
                e = float(real_series[-1].get("value") or 0.0)
                if s != 0:
                    data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"
                data["sources"].append({
                    "name": "WHO GHO — Dengue",
                    "url": "https://ghoapi.azureedge.net/ghoapi/api",
                    "date": None
                })
        else:
            # No live source defined for influenza/tb/hiv/measles in form route
            pass

    except Exception:
        # Fail-quiet on live fetch issues; we'll try local next
        real_series = None

    # ---------------- 2) LOCAL CSV fallback (if no live) ----------------
    if not data.get("timeseries") and USE_LOCAL_DATA:
        local_series = fetch_local_timeseries(
            disease=data["disease"],
            region=region,
            date_from=form.date_from,
            date_to=form.date_to,
        )
        if local_series and len(local_series) >= 2:
            data["timeseries"] = local_series
            s = float(local_series[0].get("value") or 0.0)
            e = float(local_series[-1].get("value") or 0.0)
            if s != 0:
                data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"
            data["sources"].append({
                "name": "Local dataset (monthly)",
                "url": "file://data/timeseries_monthly.csv",
                "date": None
            })
        else:
            # Record that we attempted local but had no rows in the requested window
            data["sources"].append({
                "name": "Local dataset (monthly) — no rows in window",
                "url": "file://data/timeseries_monthly.csv",
                "date": None
            })

    # ---------------- 3) SYNTHETIC last resort (optional) ----------------
    if not data.get("timeseries") and USE_SYNTHETIC_FALLBACK:
        data["timeseries"] = generate_synthetic_timeseries(
            date_from=form.date_from,
            date_to=form.date_to,
            weekly_increase_str=None,
            points=12 if form.aggregate in (AggregateEnum.weekly, AggregateEnum.monthly) else 30,
            start_value=100.0,
        )
        ts = data["timeseries"]
        if len(ts) >= 2:
            s = float(ts[0]["value"] or 0.0)
            e = float(ts[-1]["value"] or 0.0)
            if s != 0:
                data["insights"]["weekly_increase"] = f"{((e - s) / s) * 100.0:.1f}%"
        data["sources"].append({"name": "Synthetic (no/insufficient live + local data)", "url": "about:blank", "date": None})

        # Optional provenance hints even when synthetic:
        if disease_l == "malaria":
            data["sources"].append({"name": "World Bank — Malaria incidence (SH.MLR.INCD.P3)", "url": "https://data.worldbank.org/indicator/SH.MLR.INCD.P3", "date": None})
        if disease_l == "dengue":
            data["sources"].append({"name": "WHO GHO — Dengue", "url": "https://ghoapi.azureedge.net/ghoapi/api", "date": None})
        if disease_l == "covid":
            data["sources"].append({"name": "disease.sh (JHU)", "url": "https://disease.sh/", "date": None})

    # ---------------- 4) Validate & delegate ----------------
    req = ReportRequest(**data)
    resp = generate_report(req)

    # If the caller unchecked include_pdf, drop pdf_url
    if not form.include_pdf:
        resp.pdf_url = None

    return resp


@app.post("/forecast", response_model=ForecastResponse)
def make_forecast(form: ForecastForm = Body(...)):
    _guard_dates(form.date_from, form.date_to)
    res = forecast_monthly(
        disease=form.disease.value,
        region=form.region,
        date_from=form.date_from,
        date_to=form.date_to,
        horizon_months=form.horizon_months,
    )

    # --- save as CSV artifact ---
    import csv
    from datetime import datetime
    slug_d = slugify(form.disease.value)
    slug_r = slugify(form.region)
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    fname = f"{slug_d}_{slug_r}_{form.date_to.replace('-','')}_h{form.horizon_months}_{stamp}.csv"
    fpath = Path(FORECASTS_DIR) / fname
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "yhat", "yhat_lower", "yhat_upper"])
        for row in res.forecast:
            w.writerow([row["date"], row.get("yhat") or row.get("value"), row.get("yhat_lower",""), row.get("yhat_upper","")])

    public_url = f"/agents/report_generator/artifacts/forecasts/{fname}"

    return ForecastResponse(
        history=res.history,
        forecast=res.forecast,
        method=res.method,
        provenance=res.provenance,
        warnings=res.warnings,
        forecast_url=public_url,
    )


@app.post("/train_model")
def train_model_endpoint(body: TrainForm):
    """
    Fit and persist a SARIMAX model for (disease, region) using the local merged CSV.
    """
    res, status = train_model(body.disease, body.region)
    ok = (res is not None and status == "ok")
    return {
        "ok": ok,
        "status": status,
        "model_key": f"{body.disease.lower()}__{body.region.lower().replace(' ', '-')}",
    }


@app.post("/predict_month")
def predict_month_endpoint(body: PredictMonthForm):
    """
    Predict a single month (YYYY-MM) using a persisted (or on-the-fly trained) model.
    """
    yhat, status = predict_single_month(body.disease, body.region, body.target_month)
    if yhat is None:
        # bubble the reason clearly
        return {
            "ok": False,
            "status": status,
            "value": None,
            "note": "Model not available or insufficient data to train.",
        }
    return {
        "ok": True,
        "status": status,   # "forecast" or "in_sample"
        "value": yhat,
        "unit": "cases",    # or whatever your metric denotes
        "provenance": [
            "local model: SARIMAX(1,1,1)x(1,0,1,12)",
            "trained on data/timeseries_monthly.csv",
        ],
    }
