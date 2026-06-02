from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN_STATE = PROJECT_ROOT / "design_state.json"


def load_design_state(path: str | Path | None = None) -> Dict[str, Any]:
    state_path = Path(path) if path else DEFAULT_DESIGN_STATE
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    data = json.loads(state_path.read_text(encoding="utf-8"))
    validate_design_state(data)
    data["_path"] = str(state_path)
    return data


def validate_design_state(state: Dict[str, Any]) -> None:
    for key in ("task_name", "binder", "target", "mutation_policy"):
        if key not in state:
            raise ValueError(f"design_state missing required key: {key}")

    target = state["target"]
    seq = target.get("sequence", "")
    if not seq:
        raise ValueError("design_state target.sequence is empty")

    for span in target.get("epitope_spans", []):
        if len(span) != 2:
            raise ValueError(f"Invalid epitope span: {span}")
        start, end = int(span[0]), int(span[1])
        if not (0 <= start < end <= len(seq)):
            raise ValueError(f"Epitope span out of range: {span}")


def segment_spans(parts: List[List[str]]) -> Dict[str, Tuple[int, int]]:
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name, _kind, seq in parts:
        spans[name] = (cursor, cursor + len(seq))
        cursor += len(seq)
    return spans


def flatten_binder_parts(state: Dict[str, Any]) -> List[List[str]]:
    binder = state["binder"]
    return (
        list(binder["vh_segments"])
        + list(binder["linker_segments"])
        + list(binder["vl_segments"])
    )


def binder_sequence(state: Dict[str, Any]) -> str:
    return "".join(part[2] for part in flatten_binder_parts(state))

