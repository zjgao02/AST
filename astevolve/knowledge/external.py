from __future__ import annotations

from typing import Any, Optional


def load_provider(path: Optional[str], **kwargs: Any):
    """Load the configured external knowledge provider.

    The default provider loads ASTevolve prior-cache files and optional
    InterPro/SAbDab-style embedding manifests. Case-specific retrieval filters
    stay in each prior cache so the generic registry remains case-neutral.
    """
    from astevolve.knowledge.providers.sabdab_antibody import load_external_knowledge_provider

    return load_external_knowledge_provider(path, **kwargs)
