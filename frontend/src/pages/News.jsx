import { useState, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import "../styles/layout.css";
import "../styles/news.css";

const ORCH_URL = import.meta.env.VITE_ORCH_URL ?? "http://127.0.0.1:8010";

export default function News() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function fetchNews(limit = 12) {
    try {
      setLoading(true);
      setError("");

      const res = await fetch(`${ORCH_URL}/route/news?limit=${limit}`);
      if (!res.ok) {
        throw new Error(`Failed to fetch news: ${res.status}`);
      }

      const data = await res.json();
      // Accept either {status:'success', news:[...]} or {news:[...]}
      const items = Array.isArray(data?.news) ? data.news : [];
      if (data?.status && data.status !== "success" && items.length === 0) {
        throw new Error(data?.detail || "Failed to load news");
      }

      setNews(items);
    } catch (err) {
      setError(err?.message || "Failed to load news.");
      console.error("News fetch error:", err);
      setNews([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchNews();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const formatDate = (dateString) => {
    if (!dateString) return "Date not available";
    const d = new Date(dateString);
    if (Number.isNaN(d.getTime())) return dateString;
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const getNewsIcon = (source, type) => {
    const icons = {
      WHO: "🏥",
      CDC: "🩺",
      NIH: "🔬",
      "WHO News": "🌍",
      "System Info": "💡",
      "Health Education": "📚",
      "Disease Control": "🦠",
      Research: "🧪",
      Immunization: "💉",
      "Mental Wellness": "🧠",
      "Preventive Care": "🛡️",
      Nutrition: "🍎",
      Exercise: "🏃",
      default: "📰",
    };
    return icons[source] || icons[type] || icons.default;
  };

  return (
    <div className="app-shell">
      <Sidebar />
      <Header />

      <main className="app-main">
        <section className="news">
          <div className="news__header-section">
            <h2 className="news__title">
              <span className="news__icon">📰</span>
              Public Health News
            </h2>
          </div>

          {error && (
            <div className="error-banner">
              {error}
              <button onClick={() => fetchNews()} className="retry-btn">
                Retry
              </button>
            </div>
          )}

          <div className="news__panel">
            {loading && news.length === 0 ? (
              <div className="news__loading">
                <div className="news__loading-spinner">⏳</div>
                <div>Loading latest health news...</div>
              </div>
            ) : news.length === 0 ? (
              <div className="news__empty">
                <div className="news__empty-emoji">📰</div>
                <div className="news__empty-head">No News Available</div>
                <div className="news__empty-body">
                  Check your connection or try refreshing.
                </div>
              </div>
            ) : (
              <div className="news__list">
                {news.map((item, index) => (
                  <article key={index} className="news__card">
                    <div className="news__image-container">
                      {item.image_url ? (
                        <img
                          src={item.image_url}
                          alt={item.title}
                          className="news__image"
                          onError={(e) => {
                            e.currentTarget.style.display = "none";
                            e.currentTarget.nextSibling.style.display = "flex";
                          }}
                        />
                      ) : null}
                      <div className="news__image-fallback">
                        {getNewsIcon(item.source, item.category || item.type)}
                      </div>
                    </div>

                    <div className="news__content">
                      <div className="news__meta">
                        <span className="news__date">
                          {formatDate(item.published)}
                        </span>
                        <span className="news__type">
                          {item.category || item.type || "Health"}
                        </span>
                      </div>

                      <h3 className="news__title-text">{item.title}</h3>

                      <p className="news__summary">
                        {item.summary || "No summary available."}
                      </p>

                      <div className="news__footer">
                        <span className="news__source">
                          {item.source || "Health Source"}
                        </span>
                        {item.link && /^https?:\/\//i.test(item.link) && (
                          <a
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="news__link"
                          >
                            Read More
                          </a>
                        )}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
