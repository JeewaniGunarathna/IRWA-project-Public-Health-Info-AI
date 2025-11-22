import { useMemo, useState } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";

import "../styles/layout.css";
import "../styles/predict.css"; // new styles (section 2 below)

const ORCH_URL = import.meta.env.VITE_ORCH_URL ?? "http://127.0.0.1:8010";

const DISEASES = ["covid", "dengue", "malaria", "tuberculosis", "influenza", "measles", "hiv"];
const COUNTRIES = ["Sri Lanka", "India", "World", "United States", "United Kingdom", "Singapore"];

function fmt(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
function addMonths(d, n) {
  const dt = new Date(d);
  dt.setMonth(dt.getMonth() + n);
  return dt;
}

export default function Predict() {
  const [disease, setDisease] = useState("dengue");
  const [region, setRegion] = useState("Sri Lanka");
  const [horizon, setHorizon] = useState(12);
  const [targetYM, setTargetYM] = useState("2026-10");

  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");

  const [monthResult, setMonthResult] = useState(null); // single month value
  const [forecast, setForecast] = useState(null);       // { forecast, method, provenance, warnings, forecast_url }

  // last 24 months auto window
  const { date_from, date_to } = useMemo(() => {
    const to = new Date();
    const from = addMonths(to, -24);
    return { date_from: fmt(from), date_to: fmt(to) };
  }, []);

  async function callJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async function onPredictMonth(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setToast("");
    setMonthResult(null);
    try {
      const data = await callJSON(`${ORCH_URL}/route/predict_month`, {
        disease,
        region,
        target_month: targetYM,
      });
      if (data?.ok) {
        setMonthResult(data);
        setToast(`Predicted ${Number(data.value).toLocaleString()} ${data.unit || ""} for ${targetYM}`);
      } else {
        setError(data?.note || "Could not predict that month.");
      }
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setBusy(false);
    }
  }

  async function onRunForecast(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setToast("");
    setForecast(null);
    try {
      const data = await callJSON(`${ORCH_URL}/route/forecast`, {
        disease,
        region,
        date_from,
        date_to,
        horizon_months: Number(horizon),
      });
      if (data?.type === "forecast") {
        setForecast(data.payload);
        setToast("Forecast ready.");
      } else {
        setError("Unexpected forecast response.");
      }
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setBusy(false);
    }
  }

  function downloadCSV() {
    if (!forecast?.forecast?.length) return;
    const rows = [["date", "yhat", "yhat_lower", "yhat_upper"]];
    forecast.forecast.forEach(r => rows.push([r.date, r.yhat, r.yhat_lower, r.yhat_upper]));
    const blob = new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), {
      href: url,
      download: `${disease}_${region}_forecast_${date_to}.csv`,
    });
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <Header />

      <main className="app-main">
        <section className="panel panel--elevated">
          <div className="panel__header">
            <div className="panel__title">
              <span className="emoji">📈</span> Disease Prediction
            </div>

            <div className="pillset">
              <span className="pill">Auto window: last 24 months</span>
              <span className="pill pill--muted">{date_from} → {date_to}</span>
            </div>
          </div>

          {/* GLOBAL selectors */}
          <div className="grid grid--2 gap">
            <label className="field">
              <span className="field__label">Disease</span>
              <select className="field__control" value={disease} onChange={(e) => setDisease(e.target.value)}>
                {DISEASES.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>

            <label className="field">
              <span className="field__label">Region / Country</span>
              <input
                list="region-list"
                className="field__control"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                placeholder="Type a country…"
              />
              <datalist id="region-list">
                {COUNTRIES.map(c => <option key={c} value={c} />)}
              </datalist>
            </label>
          </div>

          {/* A) Single month prediction */}
          <div className="card">
            <div className="card__header">
              <div>
                <div className="card__title">Predict a Single Month</div>
                <div className="card__subtitle">Example: <code>2026-10</code>. Model trains automatically if needed.</div>
              </div>
            </div>

            <form className="grid grid--3 gap" onSubmit={onPredictMonth}>
              <label className="field">
                <span className="field__label">Target month (YYYY-MM)</span>
                <input
                  className="field__control"
                  value={targetYM}
                  onChange={(e) => setTargetYM(e.target.value)}
                  placeholder="2026-10"
                />
              </label>

              <div className="field">
                <span className="field__label">&nbsp;</span>
                <button className="btn btn--brand w-100" disabled={busy}>
                  {busy ? "Working…" : "Predict month"}
                </button>
              </div>

              <div className="field">
                <span className="field__label">Result</span>
                <div className="metric">
                  {monthResult?.ok ? (
                    <>
                      <span className="metric__value">
                        {Math.round(monthResult.value).toLocaleString()}
                      </span>
                      <span className="metric__unit">{monthResult.unit || "cases"}</span>
                      <span className={`chip ${monthResult.status === "forecast" ? "chip--brand" : "chip--muted"}`}>
                        {monthResult.status}
                      </span>
                    </>
                  ) : (
                    <span className="metric__placeholder">—</span>
                  )}
                </div>
              </div>
            </form>
          </div>

          {/* B) Horizon forecast */}
          <div className="card">
            <div className="card__header">
              <div>
                <div className="card__title">Forecast Horizon</div>
                <div className="card__subtitle">Uses last 24 months automatically.</div>
              </div>
              <div className="card__actions">
                <label className="field field--compact">
                  <span className="field__label">Horizon (months)</span>
                  <input
                    type="number"
                    min={1}
                    max={36}
                    className="field__control"
                    value={horizon}
                    onChange={(e) => setHorizon(e.target.value)}
                  />
                </label>
                <button className="btn btn--brand" onClick={onRunForecast} disabled={busy}>
                  {busy ? "Working…" : "Run forecast"}
                </button>
              </div>
            </div>

            {/* meta */}
            {forecast && (
              <div className="meta">
                <span className="chip chip--brand">method: {forecast.method}</span>
                {forecast.warnings?.length > 0 && (
                  <span className="chip chip--warn">{forecast.warnings[0]}</span>
                )}
                {forecast.provenance?.[0] && (
                  <span className="chip chip--muted">{forecast.provenance[0]}</span>
                )}
                <div className="meta__spacer" />
                <button className="btn btn--ghost" onClick={downloadCSV}>⬇︎ CSV</button>
              </div>
            )}

            {/* table */}
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Month</th>
                    <th>Prediction</th>
                    <th>Low</th>
                    <th>High</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast?.forecast?.map(row => (
                    <tr key={row.date}>
                      <td>{row.date}</td>
                      <td>{row.yhat ? Math.round(row.yhat).toLocaleString() : "—"}</td>
                      <td>{row.yhat_lower ? Math.round(row.yhat_lower).toLocaleString() : "—"}</td>
                      <td>{row.yhat_upper ? Math.round(row.yhat_upper).toLocaleString() : "—"}</td>
                    </tr>
                  ))}
                  {!forecast && (
                    <tr>
                      <td colSpan={4} className="table__empty">No forecast yet. Choose horizon and click <b>Run forecast</b>.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* banners */}
          {toast && <div className="toast">{toast}</div>}
          {error && <div className="error-banner">{error}</div>}
        </section>
      </main>
    </div>
  );
}
