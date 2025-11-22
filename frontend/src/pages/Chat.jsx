import { useState, useEffect, useRef } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import "../styles/layout.css";
import "../styles/chat.css";

import { sendChat } from "../lib/api";
// IMPORTANT: this file contains JSX, so ensure it’s .jsx or import with the extension
import {
  formatAssistantPayload,
  SourcesInline,
  AssistantDetails,
} from "../utils/formatters.jsx";

export default function Chat() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]); // [{role:'user'|'assistant', content:string|object}]
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const listEndRef = useRef(null);

  // Auto-scroll on new messages
  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(e) {
    e.preventDefault();
    const text = message.trim();
    if (!text) return;

    // push user message
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setMessage("");
    setError("");
    setLoading(true);

    try {
      // history is optional; pass [] or build from messages if you want context
      const res = await sendChat(text, []);

      // Decide how to store the assistant response:
      // If the orchestrator/agents returned a plain "reply", keep it as string.
      // Otherwise store the whole structured payload object, so we can pretty render it.
      let assistantMsg;
      if (typeof res?.reply === "string") {
        assistantMsg = { role: "assistant", content: res.reply };
      } else {
        assistantMsg = { role: "assistant", content: res }; // keep structured JSON
      }

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <Header />

      <main className="app-main">
        <section className="chat">
          <h2 className="chat__title">
            <span className="chat__bubble">💬</span>
            Health Bridge Chat Assistant
          </h2>

          <div className="chat__panel">
            <div className="chat__list" aria-live="polite">
              {messages.length === 0 && (
                <div className="chat__empty">
                  <div className="chat__empty-emoji">👋</div>
                  <div className="chat__empty-head">Welcome to Health Bridge!</div>
                  <div className="chat__empty-body">
                    Ask me about diseases, symptoms, prevention, or health data.
                  </div>
                </div>
              )}

              {/* Render messages */}
              {messages.map((msg, idx) => {
                const isUser = msg.role === "user";

                // If assistant content is an object, pretty-format it; otherwise show raw string
                const mainText = isUser
                  ? String(msg.content)
                  : typeof msg.content === "object"
                  ? formatAssistantPayload(msg.content)
                  : String(msg.content);

                return (
                  <div
                    key={idx}
                    className={`chat__msg ${
                      isUser ? "chat__msg--user" : "chat__msg--agent"
                    }`}
                  >
                    <div className="chat__msg-text">{mainText}</div>

                    {/* If assistant and structured payload, show extra details & sources */}
                    {!isUser && typeof msg.content === "object" && (
                      <>
                        <AssistantDetails payload={msg.content} />
                        <SourcesInline sources={msg.content?.sources} />
                      </>
                    )}
                  </div>
                );
              })}

              <div ref={listEndRef} />
            </div>

            <form className="chat__composer" onSubmit={handleSend}>
              <input
                type="text"
                className="chat__input"
                placeholder="Ask Health Bridge about your health..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                disabled={loading}
              />
              <button
                type="submit"
                className="chat__send"
                disabled={!message.trim() || loading}
                aria-label="Send"
                title="Send"
              >
                {loading ? "…" : "⤴︎"}
              </button>
            </form>
          </div>

          {error && <div className="error-banner">{error}</div>}
        </section>
      </main>
    </div>
  );
}
