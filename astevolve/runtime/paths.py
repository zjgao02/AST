from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    raw = os.environ.get("ASTEVOLVE_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _root_from_env(env_name: str, default_name: str) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return project_root() / default_name


def data_root() -> Path:
    return _root_from_env("ASTEVOLVE_DATA_ROOT", "data")


def artifact_root() -> Path:
    return _root_from_env("ASTEVOLVE_ARTIFACT_ROOT", "artifacts")


def model_root() -> Path:
    return _root_from_env("ASTEVOLVE_MODEL_ROOT", "model_weights")


def tmp_root(name: str = "astevolve") -> Path:
    raw = os.environ.get("ASTEVOLVE_TMP_ROOT")
    root = Path(raw).expanduser().resolve() if raw else Path(tempfile.gettempdir())
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: Optional[str | Path], *, base: Optional[Path] = None, fallback: Optional[Path] = None) -> Path:
    if value is None or str(value).strip() == "":
        if fallback is None:
            raise ValueError("resolve_path needs a value or fallback")
        return fallback.expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or project_root()) / path).resolve()


def data_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)


def artifact_path(*parts: str) -> Path:
    return artifact_root().joinpath(*parts)


def model_path(*parts: str) -> Path:
    return model_root().joinpath(*parts)
