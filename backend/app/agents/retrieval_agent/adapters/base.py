# A tiny common shape for adapters. We'll subclass this.
from __future__ import annotations
from typing import Any, Dict

class FactsAdapter:
    """Base adapter interface. Subclasses should override both methods."""

    def supports(self, query: Dict[str, Any]) -> bool:
        """Return True if this adapter can handle the parsed query."""
        return False

    def fetch(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a normalized payload:
        {
          "query": {...},          # echo of normalized query
          "facts": {...},          # numbers/series
          "sources": [{"name":"...", "url":"..."}]
        }
        """
        raise NotImplementedError
