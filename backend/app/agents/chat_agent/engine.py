# agents/chat_agent/engine.py
import os
import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")

SYSTEM_PROMPT = (
    "You are a helpful, careful public-health assistant. "
    "Be concise, cite official sources when you can, avoid speculation, "
    "and highlight when data is uncertain."
)

def _chat(payload: dict) -> str:
    """
    Minimal Ollama chat client using the /api/chat endpoint.
    """
    url = f"{OLLAMA_BASE}/api/chat"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # concatenate streamed parts if present (Ollama may chunk responses)
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    if "response" in data:
        return data["response"].strip()
    return ""

def get_chat_response(user_message: str) -> str:
    """
    Called by the orchestrator when no ‘report’ or ‘retrieval’ route matches.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        # You can tweak these:
        "options": {
            "temperature": 0.5,
            "top_p": 0.9
        }
    }
    try:
        return _chat(payload)
    except Exception as e:
        # Fail open: never crash the app if Ollama is down
        return f"(LLM unavailable) {str(e)}"
