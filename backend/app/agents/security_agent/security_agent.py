# backend/security_agent/security_agent.py
from __future__ import annotations

import os
import re
import ipaddress
import hashlib
import logging
from urllib.parse import urlparse
from typing import Tuple
from cryptography.fernet import Fernet

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    filename="security.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------------------------------------------------------
# URL Allowlist configuration (secure defaults + env overrides)
# -----------------------------------------------------------------------------
# Only allow HTTPS by default.
ALLOWED_SCHEMES = {"https"}

# Exact hostnames to allow (no leading dot). Users can extend via env:
#   SEC_ALLOW_DOMAINS="who.int,ourworldindata.org,worldbank.org,disease.sh,cdc.gov,nih.gov"
ALLOW_DOMAINS = {
    "who.int",
    "ourworldindata.org",
    "worldbank.org",
    "disease.sh",
    "cdc.gov",
    "nih.gov",
}

_env_domains = (os.getenv("SEC_ALLOW_DOMAINS") or "").strip()
if _env_domains:
    ALLOW_DOMAINS |= {
        d.strip().lower()
        for d in _env_domains.split(",")
        if d.strip()
    }

# Wildcard suffixes to allow (must start with a dot). Users can extend via env:
#   SEC_ALLOW_WILDCARDS=".who.int,.ourworldindata.org,.worldbank.org,.cdc.gov,.nih.gov"
ALLOW_WILDCARD_SUFFIXES = {
    ".who.int",
    ".ourworldindata.org",
    ".worldbank.org",
    ".cdc.gov",
    ".nih.gov",
}

_env_wild = (os.getenv("SEC_ALLOW_WILDCARDS") or "").strip()
if _env_wild:
    ALLOW_WILDCARD_SUFFIXES |= {
        s.strip().lower()
        for s in _env_wild.split(",")
        if s.strip().startswith(".")
    }

# Private & local networks — always blocked
PRIVATE_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

def _is_local_or_private_host(host: str) -> bool:
    """
    Reject localhost and private/link-local IPs.
    """
    if host in {"localhost"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return any(ip in net for net in PRIVATE_NETS) or ip.is_link_local
    except ValueError:
        # Not an IP literal; treat as domain
        return False

def _host_allowed(host: str) -> bool:
    """
    Subdomain-aware allowlist:
      - exact host match in ALLOW_DOMAINS, OR
      - host endswith any suffix in ALLOW_WILDCARD_SUFFIXES
    """
    if host in ALLOW_DOMAINS:
        return True
    return any(host.endswith(sfx) for sfx in ALLOW_WILDCARD_SUFFIXES)

def allow_outbound_url(url: str) -> Tuple[bool, str]:
    """
    Validate outbound URL against scheme/host/allowlist rules.

    Returns:
      (ok: bool, reason: str)
        reason is one of:
          - "ok"
          - "url-parse-error"
          - "scheme-not-allowed"
          - "no-host"
          - "local-or-private-host"
          - "domain-not-allowlisted"
    """
    try:
        p = urlparse(url)
    except Exception:
        return False, "url-parse-error"

    scheme = (p.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return False, "scheme-not-allowed"

    host = (p.hostname or "").lower().strip()
    if not host:
        return False, "no-host"

    if _is_local_or_private_host(host):
        return False, "local-or-private-host"

    if not _host_allowed(host):
        return False, "domain-not-allowlisted"

    return True, "ok"

# -----------------------------------------------------------------------------
# SecurityAgent (auth, validation, masking, crypto, responsible AI filter)
# -----------------------------------------------------------------------------
class SecurityAgent:
    """
    Simple demo security helper. In production you would:
      - store hashed passwords with salt & a stronger KDF (e.g., bcrypt/argon2),
      - rotate and protect the Fernet key,
      - make block/allow rules configurable via policy management.
    """
    def __init__(self):
        # Demo credentials: "admin" / "admin" (md5). Do NOT use in production.
        self.allowed_users = {
            "admin": "21232f297a57a5a743894a0e4a801fc3"
        }

        # Symmetric encryption key (rotate & store securely in production).
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

        # Basic harmful phrase blocklist
        self._bad_words = [
            "hack", "attack", "drop database", "delete", "shutdown",
            "poison", "make drug", "kill", "bomb"
        ]

        # Safety phrases for responsible AI
        self._unsafe_health_phrases = [
            "suicide", "kill myself", "harm myself", "poison",
            "self medicate", "what medicine should i take",
            "overdose", "illegal drugs"
        ]

    # 1) Authentication
    def authenticate_user(self, username: str, password: str) -> bool:
        hashed_pw = hashlib.md5(password.encode()).hexdigest()
        if username in self.allowed_users and self.allowed_users[username] == hashed_pw:
            logging.info("User %s authenticated successfully.", username)
            return True
        logging.warning("Failed login attempt for user %s.", username)
        return False

    # 2) Input validation (basic harmful phrase blocklist)
    def validate_input(self, user_input: str) -> bool:
        ui = (user_input or "").lower()
        for word in self._bad_words:
            if word in ui:
                logging.warning("Blocked harmful input: %s", user_input)
                return False
        return True

    # 3) Data privacy (mask sensitive sequences like 10-digit numbers, emails)
    def mask_sensitive_data(self, text: str) -> str:
        if not text:
            return text
        # Mask 10+ consecutive digits, common for phone/account numbers
        masked = re.sub(r"\b\d{10,}\b", lambda m: "*" * len(m.group(0)), text)
        # Mask simple emails
        masked = re.sub(r"([A-Za-z0-9._%+-])([A-Za-z0-9._%+-]*)(@[^ \n]+)", r"\1***\3", masked)
        return masked

    # 4) Encrypt data
    def encrypt_data(self, text: str) -> bytes:
        encrypted = self.cipher.encrypt(text.encode("utf-8"))
        logging.info("Data encrypted successfully.")
        return encrypted

    # 5) Decrypt data
    def decrypt_data(self, encrypted_text: bytes) -> str:
        decrypted = self.cipher.decrypt(encrypted_text).decode("utf-8")
        logging.info("Data decrypted successfully.")
        return decrypted

    # 6) Responsible AI filter (route unsafe health queries to professionals)
    def responsible_ai_filter(self, user_input: str) -> Tuple[bool, str]:
        ui = (user_input or "").lower()
        for phrase in self._unsafe_health_phrases:
            if phrase in ui:
                logging.error("Blocked unsafe health query: %s", user_input)
                return (
                    False,
                    "⚠️ This question may be unsafe. Please consult a certified doctor or helpline."
                )
        return True, user_input
