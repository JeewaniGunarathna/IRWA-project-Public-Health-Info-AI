// src/pages/Generate.jsx
import { useRef, useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import "../styles/layout.css";
import "../styles/generate.css";
import { reportFromText } from "../lib/api";

const API_BASE = import.meta.env.VITE_API_BASE;

// You can expand these safely
const DISEASES = [
  "Select",
  "covid",
  "dengue",
  "malaria",
  "tuberculosis",
  "influenza",
  "measles",
  "hiv",
];

const COUNTRIES = [
  "Sri Lanka",
  "India",
  "World",
  "United States",
  "United Kingdom",
  "Pakistan",
  "Bangladesh",
  "Nepal",
  "China",
  "Japan",
  "Australia",
  "Canada",
  "France",
  "Germany",
  "Spain",
  "Italy",
  "Indonesia",
];

export default function Generate() {
  const [tab, setTab] = useState("form"); // "text" | "form"

  // TEXT PROMPT STATE
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // FORM STATE
  const [fdisease, setFdisease] = useState("COVID-19");
  const [fregion, setFregion] = useState("Sri Lanka");
  const [fFrom, setFFrom] = useState("2021-01-10");
  const [fTo, setFTo] = useState("2021-02-10");
  const [fSource, setFSource] = useState("both"); // who | disease.sh | both | none
  const [formLoading, setFormLoading] = useState(false);
  const [formErr, setFormErr] = useState("");

  const resultRef = useRef(null);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result]);

  // -------------------------
  // TEXT PROMPT FLOW
  // -------------------------
  async function handleGenerate() {
    setError("");
    setResult(null);

    const text = (prompt || "").trim();
    if (text.length < 8) {
      setError("Please describe your request (at least 8 characters).");
      return;
    }

    try {
      setLoading(true);
      const data = await reportFromText(text); // orchestrator /route/report_from_text -> report agent
      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  // -------------------------
  // STRUCTURED FORM FLOW
  // -------------------------
  async function handleFormSubmit(e) {
    e.preventDefault();

    setFormErr("");
    setResult(null);

    // quick validation
    if (!fdisease || !fregion || !fFrom || !fTo) {
      setFormErr("Please fill all required fields.");
      return;
    }
    if (fFrom > fTo) {
      setFormErr("Start date must be before end date.");
      return;
    }

    try {
      setFormLoading(true);

      // Call the orchestrator endpoint you added: POST /route/report_form
      const r = await fetch(`${API_BASE}/route/report_form`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease: fdisease,
          region: fregion,
          date_from: fFrom,
          date_to: fTo,
          prefer_sources: fSource, // "who" | "disease.sh" | "both" | "none"
        }),
      });

      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt || `HTTP ${r.status}`);
      }

      const data = await r.json();
      setResult(data);
    } catch (e) {
      setFormErr(e.message || "Failed to generate report.");
    } finally {
      setFormLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <Header />

      <main className="app-main">
        <section className="generate">
          <h1 className="generate__title">Generate Reports</h1>

          {/* Tabs */}
          <div className="generate-tabs" role="tablist" aria-label="Generate modes">
             <div
                role="tab"
                aria-selected={tab === "form"}
                tabIndex={0}
                onClick={() => setTab("form")}
                onKeyDown={(e) => e.key === "Enter" && setTab("form")}
                className={`generate-tab ${tab === "form" ? "generate-tab--active" : ""}`}
            >
              Structured Form
            </div>
            <div
              role="tab"
              aria-selected={tab === "text"}
              tabIndex={0}
              onClick={() => setTab("text")}
              onKeyDown={(e) => e.key === "Enter" && setTab("text")}
              className={`generate-tab ${tab === "text" ? "generate-tab--active" : ""}`}
            >
              Text Prompts
            </div>
          </div>

          {/* ---------------- TEXT TAB ---------------- */}
          {tab === "text" && (
            <>
              <label htmlFor="gen-input" className="generate__label">
                Enter what you want to generate
              </label>

              <textarea
                id="gen-input"
                className="generate__textarea"
                placeholder='e.g., "Generate a dengue report for Sri Lanka from 2025-07-01 to 2025-08-24 using WHO"'
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />

              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <button
                  className="generate-btn"
                  type="button"
                  onClick={handleGenerate}
                  disabled={loading}
                >
                  {loading ? "Generating…" : "Generate"}
                </button>

                <button
                  type="button"
                  className="btn-link-clear"
                  onClick={() => {
                    setPrompt("");
                    setResult(null);
                    setError("");
                  }}
                  disabled={loading}
                  style={{ background: "#e8eef6", color: "#0b1320" }}
                >
                  Clear
                </button>
              </div>

              {error && (
                <div className="error-banner" role="alert" aria-live="polite" style={{ marginTop: 12 }}>
                  {error}
                </div>
              )}

              {loading && (
                <div className="result-panel" style={{ marginTop: 18 }}>
                  <div className="result-card skeleton">
                    <div className="skeleton-line" style={{ width: "60%" }} />
                    <div className="skeleton-line" style={{ width: "90%" }} />
                    <div className="skeleton-line" style={{ width: "80%" }} />
                  </div>
                  <div className="result-card skeleton" style={{ height: 280 }}>
                    <div className="skeleton-block" style={{ width: "100%", height: "100%" }} />
                  </div>
                </div>
              )}
            </>
          )}

          {/* ---------------- FORM TAB ---------------- */}
          {tab === "form" && (
            <form className="form-panel" onSubmit={handleFormSubmit}>
              <div className="form-row">
                <div className="form-field">
                  <label className="form-label">Disease *</label>
                  <select
                    className="form-input"
                    value={fdisease}
                    onChange={(e) => setFdisease(e.target.value)}
                  >
                    {DISEASES.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-field">
                  <label className="form-label">Country / Region *</label>
                  <input
                    className="form-input"
                    list="regions"
                    value={fregion}
                    onChange={(e) => setFregion(e.target.value)}
                    placeholder="e.g., Sri Lanka"
                  />
                  <datalist id="regions">
                    {COUNTRIES.map((c) => (
                      <option key={c} value={c} />
                    ))}
                  </datalist>
                </div>
              </div>

              <div className="form-row">
                <div className="form-field">
                  <label className="form-label">From *</label>
                  <input
                    type="date"
                    className="form-input"
                    value={fFrom}
                    max={new Date().toISOString().split("T")[0]}
                    onChange={(e) => setFFrom(e.target.value)}
                  />
                </div>

                <div className="form-field">
                  <label className="form-label">To *</label>
                  <input
                    type="date"
                    className="form-input"
                    value={fTo}
                    max={new Date().toISOString().split("T")[0]}
                    onChange={(e) => setFTo(e.target.value)}
                  />
                </div>
              </div>

              <div className="form-row">
                <div className="form-field">
                  <label className="form-label">Preferred Sources</label>
                  <select
                    className="form-input"
                    value={fSource}
                    onChange={(e) => setFSource(e.target.value)}
                  >
                    <option value="both">WHO + disease.sh (recommended)</option>
                    <option value="who">WHO</option>
                    <option value="disease.sh">disease.sh (JHU)</option>
                    <option value="none">None (allow synthetic)</option>
                  </select>
                </div>
              </div>

              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <button type="submit" className="generate-btn" disabled={formLoading}>
                  {formLoading ? "Generating…" : "Generate"}
                </button>

                <button
                  type="button"
                  className="btn-link-clear"
                  onClick={() => {
                    setResult(null);
                    setFormErr("");
                  }}
                  disabled={formLoading}
                  style={{ background: "#e8eef6", color: "#0b1320" }}
                >
                  Clear
                </button>
              </div>

              {formErr && (
                <div className="error-banner" role="alert" aria-live="polite" style={{ marginTop: 12 }}>
                  {formErr}
                </div>
              )}

              {formLoading && (
                <div className="result-panel" style={{ marginTop: 18 }}>
                  <div className="result-card skeleton">
                    <div className="skeleton-line" style={{ width: "60%" }} />
                    <div className="skeleton-line" style={{ width: "90%" }} />
                    <div className="skeleton-line" style={{ width: "80%" }} />
                  </div>
                  <div className="result-card skeleton" style={{ height: 280 }}>
                    <div className="skeleton-block" style={{ width: "100%", height: "100%" }} />
                  </div>
                </div>
              )}
            </form>
          )}

          {/* ---------------- RESULTS (shared) ---------------- */}
          {result && !(loading || formLoading) && (
            <div className="result-panel" ref={resultRef}>
              <div className="result-card">
                <h2 className="result-title">Summary</h2>
                <p className="result-text">{result.summary}</p>
              </div>

              {Array.isArray(result.visuals) &&
                result.visuals.length > 0 &&
                result.visuals[0]?.path && (
                  <div className="result-card">
                    <h3 className="result-subtitle">Trend</h3>
                    <img
                      className="result-chart"
                      src={`${API_BASE}${result.visuals[0].path}`}
                      alt="Trend chart"
                    />
                  </div>
                )}

              {Array.isArray(result.sources) && result.sources.length > 0 && (
                <div className="result-card">
                  <h3 className="result-subtitle">Sources</h3>
                  <ul className="result-sources">
                    {result.sources.map((s, idx) => (
                      <li key={idx}>
                        <a href={s.url} target="_blank" rel="noreferrer">
                          {s.name || s.url}
                        </a>
                        {s.date ? ` (${s.date})` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="result-actions">
                {result.report_url && (
                  <a
                    className="btn-link"
                    href={`${API_BASE}${result.report_url}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    View Report
                  </a>
                )}
                {result.pdf_url ? (
                  <a
                    className="btn-link"
                    href={`${API_BASE}${result.pdf_url}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download PDF
                  </a>
                ) : (
                  <button className="btn-link btn-link--disabled" disabled>
                    PDF not available
                  </button>
                )}
              </div>

              {result.disclaimer && (
                <p className="result-disclaimer">{result.disclaimer}</p>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
