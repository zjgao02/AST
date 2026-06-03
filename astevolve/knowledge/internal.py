from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class YamlInternalMemoryStore:
    """Case-local internal memory store.

    This wraps the existing `memory.yaml` format. It intentionally keeps update
    logic in `engine.memory_update` so existing runs stay compatible.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        if not self.path.exists():
            return {}
        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def update_from_round(self, round_output: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        from engine.memory_update import update_internal_memory

        return update_internal_memory(self.path, round_output, **kwargs)


def load_internal_memory_store(path: Optional[str | Path]) -> Optional[YamlInternalMemoryStore]:
    if not path:
        return None
    return YamlInternalMemoryStore(path)

