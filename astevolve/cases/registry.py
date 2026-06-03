from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DesignCase
from astevolve.runtime.paths import artifact_path, project_root, resolve_path


PROJECT_ROOT = project_root()
CASES_ROOT = PROJECT_ROOT / "cases"


def _resolve_path(raw: Optional[str], base: Path, fallback: Path) -> Path:
    return resolve_path(raw, base=base, fallback=fallback)


def _discover_cases() -> List[str]:
    if not CASES_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in CASES_ROOT.iterdir()
        if path.is_dir() and (path / "case.json").exists()
    )


def list_cases() -> List[str]:
    return _discover_cases()


def default_case_id() -> str:
    cases = list_cases()
    if not cases:
        raise FileNotFoundError(f"No case manifests found under {CASES_ROOT}")
    return cases[0]


def _load_manifest(case_id: str) -> tuple[Optional[Path], Dict[str, Any]]:
    path = CASES_ROOT / case_id / "case.json"
    if not path.exists():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        data = {}
    return path, data if isinstance(data, dict) else {}


def resolve_case(case_id: Optional[str] = None) -> DesignCase:
    selected = case_id or os.environ.get("ASTEVOLVE_CASE_ID") or default_case_id()
    manifest_path, manifest = _load_manifest(selected)
    base = manifest_path.parent if manifest_path is not None else CASES_ROOT / selected

    design_state_override = os.environ.get("ASTEVOLVE_DESIGN_STATE_PATH")
    memory_override = os.environ.get("ASTEVOLVE_MEMORY_PATH")
    output_override = os.environ.get("ASTEVOLVE_CASE_OUTPUT_ROOT")

    design_state_path = _resolve_path(
        design_state_override or manifest.get("design_state_path"),
        base,
        base / "design_state.json",
    )
    memory_path = _resolve_path(
        memory_override or manifest.get("memory_path"),
        base,
        base / "memory.yaml",
    )
    output_root = _resolve_path(
        output_override or manifest.get("output_root"),
        base,
        artifact_path(selected),
    )

    metadata = {k: v for k, v in manifest.items() if k not in {
        "design_state_path",
        "memory_path",
        "output_root",
    }}
    return DesignCase(
        case_id=selected,
        root=PROJECT_ROOT,
        manifest_path=manifest_path,
        design_state_path=design_state_path,
        memory_path=memory_path,
        output_root=output_root,
        metadata=metadata,
    )


def current_case() -> DesignCase:
    return resolve_case()
