// src/utils/formatters.jsx
import React from "react";

/** Turn assistant JSON payloads into a single friendly summary line. */
export function formatAssistantPayload(payload) {
  if (!payload || typeof payload !== "object") return "";

  // 1) Most of your retrieval responses look like:
  // { type: "retrieval", query: {...}, facts: { summary: "...", type: "covid_cases", data: {...} }, sources:[...] }
  const summaryFromFacts = payload?.facts?.summary;
  if (summaryFromFacts && typeof summaryFromFacts === "string") {
    return summaryFromFacts;
  }

  // 2) Report generator returns { summary: "...", visuals: [...], sources: [...] }
  if (typeof payload?.summary === "string" && payload.summary.trim()) {
    return payload.summary.trim();
  }

  // 3) Sometimes you might get { reply: "..." }
  if (typeof payload?.reply === "string" && payload.reply.trim()) {
    return payload.reply.trim();
  }

  // 4) A couple of specific fact types (optional sugar)
  const t = payload?.facts?.type || payload?.type || "";
  const data = payload?.facts?.data || {};
  const region = payload?.query?.country || payload?.query?.region;

  if (t === "covid_cases") {
    const cases = data?.cases ?? "unknown";
    const today = data?.todayCases ?? "unknown";
    if (region) {
      return `${region} — ${cases} total COVID cases (${today} today).`;
    }
    return `COVID cases — ${cases} total (${today} today).`;
  }

  if (t === "medicine_side_effects") {
    const med = payload?.facts?.data?.medicine || payload?.query?.medicine || "this medicine";
    return `Commonly reported side effects for ${med}.`;
  }

  if (t === "us_flu_ili") {
    const last = Array.isArray(payload?.facts?.series) && payload.facts.series.at(-1);
    const v = last?.value ?? null;
    if (v != null) return `US ILI (flu) most recent value: ${v}%.`;
    return "US ILI (flu) series retrieved.";
  }

  if (t === "worldbank_indicator") {
    const title = payload?.facts?.title || "indicator";
    const start = payload?.facts?.start ?? "n/a";
    const end = payload?.facts?.end ?? "n/a";
    const chg = payload?.facts?.change_pct ?? "n/a";
    const c = payload?.query?.country || payload?.query?.region || "";
    return `${title}${c ? ` — ${c}` : ""}: ${start} → ${end} (${chg}%).`;
  }

  // 5) Last resort: try a generic field or JSON fallback
  if (typeof payload?.message === "string") return payload.message;
  return "";
}

/** Inline sources chips (unchanged) */
export function SourcesInline({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) return null;
  return (
    <div className="sources-inline">
      <span>Sources: </span>
      {sources.map((s, i) => (
        <a
          key={i}
          href={s.url || "#"}
          target="_blank"
          rel="noreferrer"
          className="source-chip"
        >
          {s.name || s.url || "source"}
        </a>
      ))}
    </div>
  );
}

/** Optional small details block (you can keep or remove as you like) */
export function AssistantDetails({ payload }) {
  if (!payload || typeof payload !== "object") return null;

  // Example: show last 3 points of a timeseries if present
  const series = payload?.facts?.series;
  if (Array.isArray(series) && series.length > 0) {
    const last3 = series.slice(-3);
    return (
      <div className="assistant-details">
        <div className="assistant-details__head">Recent values</div>
        <ul className="assistant-details__list">
          {last3.map((p, idx) => (
            <li key={idx}>
              <span className="assistant-details__date">{p.date}</span>
              <span className="assistant-details__value">{String(p.value)}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return null;
}


// src/utils/formatters.js

// Safe number format with thousands separators
const nf = new Intl.NumberFormat();

function fmt(n) {
  if (n === null || n === undefined) return "—";
  if (typeof n === "number") return nf.format(n);
  // sometimes numbers arrive as strings
  const num = Number(n);
  return Number.isFinite(num) ? nf.format(num) : String(n);
}

/**
 * Build a friendly, narrative summary from a retrieval payload.
 * Supports: covid_all (disease.sh), worldbank_* indicators, FDA side effects, USDA nutrition.
 */
export function composeReadableSummary(payload) {
  if (!payload) return "";

  // Some backends wrap result in { type, query, facts, sources }
  const facts = payload.facts || payload;

  // 1) COVID all-country block (disease.sh)
  if (facts.type === "covid_all" && facts.data) {
    const d = facts.data;
    const country = d.country || "this location";
    const cases = fmt(d.cases);
    const deaths = fmt(d.deaths);
    const todayCases = fmt(d.todayCases || 0);
    const todayDeaths = fmt(d.todayDeaths || 0);
    const recovered = fmt(d.recovered);
    const active = fmt(d.active);
    const tests = fmt(d.tests);

    return [
      `Here’s the latest COVID-19 snapshot for ${country}.`,
      `Total confirmed cases stand at ${cases}, with ${deaths} total deaths reported so far.`,
      todayCases !== "0" || todayDeaths !== "0"
        ? `Today’s change shows ${todayCases} new cases and ${todayDeaths} new deaths.`
        : `No new cases or deaths were reported today.`,
      recovered !== "—"
        ? `Recoveries are at ${recovered}, with ${active !== "—" ? `${active} cases currently active` : "active cases reported"}`
        : null,
      tests !== "—" ? `, and ${tests} tests recorded to date.` : ".",
    ]
      .filter(Boolean)
      .join(" ");
  }

  // 2) World Bank indicators (we used type names like worldbank_SP.DYN.LE00.IN)
  if (String(facts.type || "").startsWith("worldbank_") && facts.data) {
    const d = facts.data;
    const title = d.title || "Indicator";
    const country = d.country || "";
    let latestLine = "";
    if (d.latest && d.latest.value !== undefined && d.latest.year) {
      latestLine = `${title} in ${country} is ${fmt(d.latest.value)} as of ${d.latest.year}.`;
    }
    const trend =
      typeof d.change_pct === "number"
        ? d.change_pct > 0
          ? `That’s up by ${d.change_pct}% over the available time series.`
          : d.change_pct < 0
          ? `That’s down by ${Math.abs(d.change_pct)}% over the available time series.`
          : `There’s no overall change across the series.`
        : "";

    return [latestLine || `${title} for ${country} is shown below.`, trend].filter(Boolean).join(" ");
  }

  // 3) FDA medicine side effects
  if (facts.type === "medicine_side_effects" && facts.data) {
    const med = facts.data.medicine || "this medicine";
    const list = (facts.data.side_effects || [])
      .slice(0, 8)
      .map((s) => s[0].toUpperCase() + s.slice(1))
      .join(", ");

    if (list) {
      return `Commonly reported side effects for ${med} include: ${list}. This isn’t a complete list and severity varies — please consult a healthcare professional for guidance tailored to you.`;
    }
    return `No common side effects were available for ${med} in the FDA feed.`;
  }

  // 4) USDA nutrition results (we used 'us_nutrition' wrapper)
  if (facts.type === "us_nutrition" && facts.data) {
    const results = facts.data.results || [];
    if (!results.length) {
      return "We couldn’t find foods that clearly match that query. Try a simpler phrase like “vitamin C” or “high protein”.";
    }
    const top = results.slice(0, 3).map((f) => f.food_name).filter(Boolean);
    if (top.length) {
      return `Here are some of the most relevant foods we found: ${top.join(", ")}. Tap a food to explore its nutrients in detail.`;
    }
    return "We found matching foods — expand the list to see their nutrients.";
  }

  // 5) If your agent already provided a nice one-liner, reuse it
  if (facts.summary) return String(facts.summary);

  return "";
}
