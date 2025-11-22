const BASE = import.meta.env.VITE_API_BASE; // http://127.0.0.1:8010

export async function reportFromText(query) {
  const res = await fetch(`${BASE}/route/report_from_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Request failed with ${res.status}`);
  }
  return res.json();
}

// Chat endpoint (Orchestrator)
export async function sendChat(message, history = []) {
  const res = await fetch(`${BASE}/route/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Chat request failed with ${res.status}`);
  }
  return res.json();
}

// --- Search (web) ---
export async function searchWeb(query, filters = {}) {
  const BASE = import.meta.env.VITE_API_BASE; // e.g. http://127.0.0.1:8010
  const res = await fetch(`${BASE}/route/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, filters }),
  });
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Search failed with ${res.status}`);
  }
  return res.json();
}
