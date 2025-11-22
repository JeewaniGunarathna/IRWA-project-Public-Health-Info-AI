// src/components/Header.jsx
import { useState, useEffect, useRef } from "react";
import "../styles/header.css";

function formatToday() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}.${m}.${day}`;
}

export default function Header() {
  const today = formatToday();

  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [links, setLinks] = useState([]); // last fetched links we show under “Top sources”
  const [showPanel, setShowPanel] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const debounceRef = useRef(null);

  // --- Suggest while typing ---------------------------------------------------
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    const q = query.trim();
    if (!q) {
      setSuggestions([]);
      setShowPanel(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(
          `http://127.0.0.1:8010/route/search_suggest?q=${encodeURIComponent(q)}`
        );
        const data = await res.json();
        setSuggestions(Array.isArray(data?.suggestions) ? data.suggestions : []);
        setShowPanel(true);
      } catch {
        setSuggestions([]);
      }
    }, 250);

    return () => clearTimeout(debounceRef.current);
  }, [query]);

  // --- Fetch links for a query ------------------------------------------------
  async function fetchLinks(q) {
    setLoading(true);
    setError("");
    try {
      const r = await fetch("http://127.0.0.1:8010/route/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, mode: "links" }),
      });
      if (!r.ok) throw new Error("Failed to fetch links");
      const data = await r.json();
      const items = Array.isArray(data?.items) ? data.items : [];
      setLinks(items);
      setShowPanel(true);
      return items;
    } catch (err) {
      setError(err.message || "Something went wrong");
      setLinks([]);
      setShowPanel(true);
      return [];
    } finally {
      setLoading(false);
    }
  }

  // --- Open first link for a query (helper) ----------------------------------
  async function openFirstLinkForQuery(q) {
    const items = await fetchLinks(q);
    if (items.length > 0 && items[0].url) {
      window.open(items[0].url, "_blank");
    } else {
      setError("No sources found for that query.");
    }
  }

  // --- Enter key behavior -----------------------------------------------------
  async function handleKeyDown(e) {
    if (e.key === "Enter") {
      e.preventDefault();
      const q = query.trim();
      if (!q) return;

      // If we already have link results, open the first one immediately.
      if (links.length > 0 && links[0]?.url) {
        window.open(links[0].url, "_blank");
      } else {
        await openFirstLinkForQuery(q);
      }
    }
  }

  // --- Click a suggestion = fetch & open first link ---------------------------
  async function onPickSuggestion(s) {
    setQuery(s);              // put phrase in the box so user sees it
    setSuggestions([]);       // collapse suggestion list
    setShowPanel(true);       // keep panel for feedback/errors
    await openFirstLinkForQuery(s);
  }

  return (
    <header className="header">
      <div className="header__title">Public Health Info AI</div>

      <div className="header__search">
        {/* search icon */}
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M21 21l-4.3-4.3M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15z"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <input
          type="search"
          placeholder="Search health data…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowPanel(true)}
        />

        {query && (
          <button
            className="header__clear"
            onClick={() => {
              setQuery("");
              setSuggestions([]);
              setLinks([]);
              setError("");
              setShowPanel(false);
            }}
            aria-label="Clear search"
            title="Clear"
          >
            ×
          </button>
        )}

        {/* Dropdown panel */}
        {showPanel && (suggestions.length > 0 || links.length > 0 || loading || error) && (
          <div className="header__panel">
            {/* Suggestions */}
            {suggestions.length > 0 && (
              <div className="hdr-section">
                <div className="hdr-section__title">Suggestions</div>
                <ul className="hdr-list">
                  {suggestions.map((s, i) => (
                    <li
                      key={`${s}-${i}`}
                      className="hdr-item hdr-item--phrase"
                      onMouseDown={() => onPickSuggestion(s)} // use mousedown so it fires before input blur
                    >
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Top sources (from last fetch) */}
            {links.length > 0 && (
              <div className="hdr-section">
                <div className="hdr-section__title">Top sources</div>
                <ul className="hdr-list">
                  {links.map((link, i) => (
                    <li key={i} className="hdr-item">
                      <a
                        href={link.url}
                        target="_blank"
                        rel="noreferrer"
                        className="hdr-link"
                      >
                        <div className="hdr-item__title">{link.title}</div>
                        <div className="hdr-item__source">{link.source}</div>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {loading && (
              <div className="hdr-empty">
                Loading <span className="hdr-spinner" />
              </div>
            )}
            {error && <div className="hdr-error">{error}</div>}
          </div>
        )}
      </div>

      <div className="header__date">{today}</div>
      <div className="header__avatar" title="Profile" aria-label="Profile" />
    </header>
  );
}
