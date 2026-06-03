from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


class ExternalKnowledgeProvider(Protocol):
    def get_node_prior(self, node_name: str, node_kind: str, chain_id: str) -> Dict[str, Any]:
        ...

    def get_sequence_prior(
        self,
        node_name: str,
        node_kind: str,
        chain_id: str,
        sequence: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...

    def describe(self) -> Dict[str, Any]:
        ...


class InternalMemoryStore(Protocol):
    def load(self) -> Dict[str, Any]:
        ...

    def update_from_round(self, round_output: Dict[str, Any]) -> Dict[str, Any]:
        ...

