# backend/orchestrator/llm.py
from __future__ import annotations
import os, json
from typing import Any, Dict, Optional

try:
    import ollama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False


MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")


def _run(prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    """
    Call the local Ollama model. If anything fails, return empty string.
    """
    if not (USE_LLM and _HAS_OLLAMA):
        return ""
    try:
        resp = ollama.generate(
            model=MODEL,
            prompt=prompt.strip(),
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return (resp.get("response") or "").strip()
    except Exception:
        return ""


def summarize_retrieval(payload: Dict[str, Any]) -> str:
    """
    Take your retrieval JSON (query, facts, sources) and produce a clean paragraph.
    """
    prompt = f"""
You are a helpful public-health assistant. Rewrite the following retrieval payload
into a clear, friendly 3–5 sentence summary for non-experts. Keep important numbers,
units, country names, and mention sources succinctly at the end (e.g., "Source: X, Y").

JSON:
{json.dumps(payload, ensure_ascii=False)}
"""
    return _run(prompt)


def synthesize_search(payload: Dict[str, Any]) -> str:
    """
    Turn the unified search result into a concise answer + action advice.
    """
    prompt = f"""
You are a health information guide. The user searched for something.
Read the JSON and produce:

1) A short paragraph (3–6 sentences) answering the query in plain language.
2) A 2–4 bullet "Key facts" list with numbers/names if present.
3) One line: "What to do next:" with a concrete suggestion.
Avoid markdown headings; bullets can be simple hyphens.

JSON:
{json.dumps(payload, ensure_ascii=False)}
"""
    return _run(prompt, temperature=0.3, max_tokens=600)


def polish_chat(text: str) -> str:
    """
    Make a short, warm, professional 1–2 sentence reply from a rough draft.
    """
    prompt = f"""
Rewrite the following message to sound friendly, concise, and professional.
Use 1–2 sentences. No emojis.

Text:
{text}
"""
    return _run(prompt, temperature=0.2, max_tokens=120)
