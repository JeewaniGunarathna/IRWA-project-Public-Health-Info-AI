import { NavLink, useNavigate } from "react-router-dom";
import "../styles/sidebar.css";

const NAV_ITEMS = [
  { label: "Dashboard", path: "/", icon: DashboardIcon },
  { label: "Chat", path: "/chat", icon: ChatIcon },
  { label: "Graph", path: "/generate", icon: GraphIcon },
  { label: "News", path: "/news", icon: NewsIcon },
  { label: "Predict", path: "/predict", icon: PredictIcon },
];

export default function Sidebar() {
  const navigate = useNavigate();

  function handleLogout() {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login"); // change to "/" if no login route yet
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__logo">
        <img src="/logo.png" alt="Health Bridge logo" />
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(({ label, path, icon: Icon }) => (
          <NavLink
            key={label}
            to={path}
            className={({ isActive }) =>
              `sidebar__btn ${isActive ? "sidebar__btn--active" : ""}`
            }
          >
            <span className="sidebar__icon"><Icon /></span>
            <span className="sidebar__text">{label}</span>
          </NavLink>

        ))}
      </nav>

      {/* LOGOUT BUTTON */}
      <button className="sidebar__logout" onClick={handleLogout}>
        ⏻ Logout
      </button>
    </aside>
  );
}

/* ===== Inline SVG Icons (lightweight, teal themed) ===== */
function DashboardIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M3 13h8V3H3v10zm10 8h8V3h-8v18zm-10 0h8v-6H3v6z" />
    </svg>
  );
}
function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M4 4h16v9a2 2 0 0 1-2 2H9l-5 3v-3H4z" />
    </svg>
  );
}
function GraphIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M3 17h2v-7H3v7zm4 0h2V3H7v14zm4 0h2v-4h-2v4zm4 0h2V9h-2v8zm4 0h2V5h-2v12z" />
    </svg>
  );
}
function NewsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M4 5h14a2 2 0 0 1 2 2v12H4V5zm2 4h8v2H6V9zm0 3h8v2H6v-2zm0 3h8v2H6v-2z" />
    </svg>
  );
}
function PredictIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
      <path d="M12 2a10 10 0 1 1 0 20A10 10 0 0 1 12 2zm1 5h-2v6l4 2 1-1-3-1V7z" />
    </svg>
  );
}
