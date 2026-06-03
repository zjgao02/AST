from __future__ import annotations

from typing import Any, Optional


def load_external_knowledge_provider(path: Optional[str], provider: str = "json", **kwargs: Any):
    """Factory for external KB modules.

    The current providers are JSON/embedding-backed and implemented by
    `astevolve.knowledge.providers`. Keeping this factory lets the inner loop swap to another
    KB implementation later without touching MCTS code.
    """

    selected = str(provider or "json").strip().lower()
    if selected in {"json", "sabdab", "magneton", "default"}:
        from .external import load_provider

        return load_provider(path, **kwargs)
    raise ValueError(f"Unknown external knowledge provider: {provider}")

