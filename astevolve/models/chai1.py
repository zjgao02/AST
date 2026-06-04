from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class Chai1StructureModel:
    """Adapter placeholder for legacy Chai-1 structure scoring."""

    name = "chai1"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Chai-1 confidence adapter is not wired in this project state.")

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError("Chai-1 scalar adapter is not wired in this project state.")

