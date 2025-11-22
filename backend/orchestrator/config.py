# backend/orchestrator/config.py
import os
from dotenv import load_dotenv

load_dotenv()

AGENTS = {
    "report": "http://127.0.0.1:8003",
    "retrieval": "http://127.0.0.1:8005",
    "chat":      "http://127.0.0.1:8007",
    "security": "http://127.0.0.1:8090",
}

SEC_USER = os.getenv("SEC_USER", "admin")
SEC_PASS = os.getenv("SEC_PASS", "admin")