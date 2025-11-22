import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import "../styles/layout.css";
import "../styles/dashboard.css";

import health1 from "../assets/health1.jpg";
import health2 from "../assets/health2.jpg";
import health3 from "../assets/health3.jpg";
import health4 from "../assets/health4.jpg";
import icon from "../assets/icon.jpg";
import corona from "../assets/corona.jpg";
import dengue from "../assets/dengue.jpg";
import malaria from "../assets/malaria.jpg";
import hiv from "../assets/hiv.jpg";

export default function Dashboard() {
  // Highlights gallery
  const cards = [
    { src: health1, alt: "Global health insights", caption: "AI insights for public health" },
    { src: health2, alt: "Disease trends dashboard", caption: "Track trends across regions" },
    { src: health3, alt: "Predictive models", caption: "Forecast outbreaks early" },
    { src: health4, alt: "Collaborative decisions", caption: "Better decisions, faster" },
  ];

  // Services
  const services = [
    { to: "/chat",     title: "Chat Assistant",  desc: "Ask health questions with sources.", icon: ChatIcon },
    { to: "/generate", title: "Graph Generator", desc: "Build clean charts from data fast.", icon: ChartIcon },
    { to: "/predict",  title: "Predictions",     desc: "Forecast cases & risk windows.",     icon: PredictIcon },
    { to: "/news",     title: "Health News",     desc: "Track verified global updates.",     icon: NewsIcon },
  ];

  // Common Diseases → WHO links (open in new tab)
  const diseases = [
    {
      name: "COVID-19",
      src: corona,
      alt: "COVID-19",
      link: "https://www.who.int/health-topics/coronavirus",
    },
    {
      name: "Dengue",
      src: dengue,
      alt: "Dengue",
      link: "https://www.who.int/health-topics/dengue-and-severe-dengue",
    },
    {
      name: "Malaria",
      src: malaria,
      alt: "Malaria",
      link: "https://www.who.int/health-topics/malaria",
    },
    {
      name: "HIV",
      src: hiv,
      alt: "HIV",
      link: "https://www.who.int/health-topics/hiv-aids",
    },
  ];

  // News preview (static seed; wire to your API later)
  const newsPreview = [
    {
      id: "n-1",
      title: "WHO updates guidance on vector-borne disease surveillance",
      source: "WHO",
      date: "2025-10-10",
      to: "/news",
      snippet: "Reinforced recommendations for early warning systems and community reporting.",
    },
    {
      id: "n-2",
      title: "Global influenza outlook remains moderate for Q4",
      source: "ECDC",
      date: "2025-10-07",
      to: "/news",
      snippet: "Seasonal activity expected to rise in temperate regions; vaccination urged.",
    },
    {
      id: "n-3",
      title: "AI-supported outbreak analytics improves response time",
      source: "BMJ",
      date: "2025-10-03",
      to: "/news",
      snippet: "Decision support tools show faster detection and targeted interventions.",
    },
  ];

  return (
    <div className="app-shell">
      <Sidebar />
      <Header />

      <main className="app-main">
        <section className="dashboard">
          {/* ===== Highlights (2x2 image grid) ===== */}
          <h2 className="dash-title">Highlights</h2>
          <div className="gallery">
            {cards.map((c, i) => (
              <figure className="gallery-card" key={i}>
                <img src={c.src} alt={c.alt} loading={i < 2 ? "eager" : "lazy"} decoding="async" />
                <figcaption><span className="pill">{c.caption}</span></figcaption>
                <div className="glow" />
              </figure>
            ))}
          </div>

          {/* ===== Our Services ===== */}
          <section className="services">
            <div className="section-head">
              <h3>Our Services</h3>
              <p>Everything you need to explore, visualize, and forecast public-health data.</p>
            </div>

            <div className="services-grid">
              {services.map(({ to, title, desc, icon: Icon }, idx) => (
                <a className="service-card" href={to} key={idx}>
                  <div className="svc-icon"><Icon /></div>
                  <div className="svc-body">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                  </div>
                  <span className="svc-link">Explore →</span>
                </a>
              ))}
            </div>
          </section>

          {/* ===== Common Diseases (WHO) ===== */}
          <section className="diseases">
            <div className="section-head">
              <h3>World Common Diseases</h3>
              <p>Quick access to WHO resources for key diseases.</p>
            </div>

            <div className="diseases-grid">
              {diseases.map((d, i) => (
                <a
                  key={i}
                  className="disease-card"
                  href={d.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`${d.name} — open WHO page in a new tab`}
                >
                  <img src={d.src} alt={d.alt} loading={i < 2 ? "eager" : "lazy"} decoding="async" />
                  <div className="disease-overlay">
                    <span className="disease-name">{d.name}</span>
                  </div>
                </a>
              ))}
            </div>
          </section>

          {/* ===== News Preview ===== */}
          <section className="news">
            <div className="section-head">
              <h3>Latest Health News</h3>
              <a className="see-all" href="/news" aria-label="See all health news">See all →</a>
            </div>

            <div className="news-grid">
              {newsPreview.map((a) => (
                <a className="news-card" href={a.to} key={a.id}>
                  <h4 className="news-title">{a.title}</h4>
                  <div className="news-meta">
                    <span className="news-source">{a.source}</span>
                    <span className="dot">•</span>
                    <time dateTime={a.date}>{a.date}</time>
                  </div>
                  <p className="news-snippet">{a.snippet}</p>
                  <span className="news-link">Read more →</span>
                </a>
              ))}
            </div>
          </section>

          {/* ===== About + Vision ===== */}
          <section className="about">
            <div className="about-wrap">
              <div className="about-media">
                <img src={icon} alt="Public health team collaboration" loading="lazy" />
              </div>
              <div className="about-body">
                <h3>About Health Bridge</h3>
                <p>
                  Health Bridge is a public-health intelligence platform that unifies **chat**, **visual analytics**, **predictions**,
                  and **news** into a single workspace. We help teams make evidence-based decisions faster with transparent methods.
                </p>
                <div className="vision">
                  <h4>Our Vision</h4>
                  <ul>
                    <li>Timely, trustworthy insights for every public-health team.</li>
                    <li>Accessible analytics that bridge experts and communities.</li>
                    <li>Responsible AI with citations, context, and clarity.</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* ===== Footer CTA ===== */}
          <section className="cta">
            <div className="cta-wrap">
              <div className="cta-text">
                <h3>Ready to explore deeper insights?</h3>
                <p>Ask the assistant, visualize trends, forecast risks, or scan today’s headlines.</p>
              </div>
              <div className="cta-actions">
                <a className="btn ghost" href="/chat">Chat</a>
                <a className="btn ghost" href="/generate">Graphs</a>
                <a className="btn solid" href="/predict">Predictions</a>
                <a className="btn ghost" href="/news">News</a>
              </div>
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

/* -------- tiny inline SVG icons (no library needed) -------- */
function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <path fill="currentColor" d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9l-5 3v-3H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>
    </svg>
  );
}
function ChartIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <path fill="currentColor" d="M4 19h16v2H2V5h2v14zM7 17h2V9H7v8zm4 0h2V5h-2v12zm4 0h2v-6h-2v6z"/>
    </svg>
  );
}
function PredictIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <path fill="currentColor" d="M3 12a9 9 0 1 1 18 0c0 4.97-4.03 9-9 9s-9-4.03-9-9zm9-7v7l5 3"/>
    </svg>
  );
}
function NewsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
      <path fill="currentColor" d="M4 5h14a2 2 0 0 1 2 2v11a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V7a2 2 0 0 1 2-2zm2 3v10h11a1 1 0 0 0 1-1V8H6zm2 2h8v2H8v-2zm0 3h8v2H8v-2z"/>
    </svg>
  );
}
