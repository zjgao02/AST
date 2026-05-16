from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_SCALAR_CACHE: Dict[str, float] = {}
_CONF_CACHE: Dict[str, Dict[str, Any]] = {}
_DEFAULT_PROTENIX_ROOT = Path(os.environ.get("ASTEVOLVE_PROTENIX_ROOT", r"D:\Downloads\protenix"))


def _default_pred_name(chains: List[Tuple[str, str]]) -> str:
    names = [str(cid) for cid, _ in chains if cid]
    return "__".join(names) if names else "pred"


def _normalise_inputs(
    pred_name: Optional[str],
    chains: Optional[List[Tuple[str, str]]],
) -> Tuple[str, List[Tuple[str, str]]]:
    if chains is None and isinstance(pred_name, list):
        chains = pred_name
        pred_name = None

    norm_chains = [(str(cid), str(seq)) for cid, seq in (chains or [])]
    return str(pred_name or _default_pred_name(norm_chains)), norm_chains


def _tmp_root() -> Path:
    root = os.environ.get("ASTEVOLVE_PROTENIX_TMP")
    if root:
        path = Path(root)
    else:
        path = Path(tempfile.gettempdir()) / "astevolve_protenix"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _protenix_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROTENIX_ROOT_DIR", str(_DEFAULT_PROTENIX_ROOT))
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return env


def _cache_key(
    pred_name: str,
    chains: List[Tuple[str, str]],
    metric: str,
    seed: int,
    model_name: str,
) -> str:
    seq_key = "|".join(f"{cid}:{seq}" for cid, seq in chains)
    return f"{pred_name}|{seq_key}|{metric}|{seed}|{model_name}"


def _write_input_json(path: Path, pred_name: str, chains: List[Tuple[str, str]]) -> None:
    sequences = [
        {"proteinChain": {"sequence": seq, "count": 1}}
        for _, seq in chains
    ]
    data = [{"sequences": sequences, "name": pred_name}]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_protenix(
    pred_name: str,
    chains: List[Tuple[str, str]],
    seed: int,
    model_name: str,
    conda_env: str,
) -> Optional[Path]:
    root = _tmp_root()
    run_dir = root / pred_name
    input_json_path = run_dir / "input.json"
    out_dir = run_dir / "output"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_input_json(input_json_path, pred_name, chains)

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "protenix",
        "predict",
        "--input",
        str(input_json_path),
        "--out_dir",
        str(out_dir),
        "--model_name",
        model_name,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=_protenix_env(),
        )
    except subprocess.TimeoutExpired:
        print("[protenix] prediction timed out (600s)")
        return None
    except FileNotFoundError:
        print("[protenix] conda or protenix not found in PATH")
        return None

    if result.returncode != 0:
        print(f"[protenix] failed (rc={result.returncode})")
        if result.stderr:
            print(f"[protenix] stderr: {result.stderr[:1000]}")
        return None

    return out_dir


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[protenix] failed to parse {path}: {exc}")
        return None


def _first_json(out_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(out_dir.rglob(pattern))
    return files[0] if files else None


def _numeric_list(value: Any) -> Optional[List[float]]:
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], (int, float)):
                out.append(float(item[0]))
            else:
                return None
        return out
    return None


def _walk_values(obj: Any, keys: Iterable[str]) -> Optional[List[float]]:
    key_set = set(keys)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in key_set:
                vals = _numeric_list(value)
                if vals:
                    return vals
        for value in obj.values():
            vals = _walk_values(value, key_set)
            if vals:
                return vals
    elif isinstance(obj, list):
        for value in obj:
            vals = _walk_values(value, key_set)
            if vals:
                return vals
    return None


def _npz_values(path: Path, keys: Iterable[str]) -> Optional[List[float]]:
    try:
        import numpy as np
    except Exception:
        return None

    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None

    for key in keys:
        if key not in data.files:
            continue
        arr = np.asarray(data[key]).reshape(-1)
        if arr.size == 0:
            continue
        try:
            return [float(x) for x in arr.tolist()]
        except (TypeError, ValueError):
            continue
    return None


def _scalar_metrics(summary: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(summary, dict):
        return out
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _split_per_chain(
    values: Optional[List[float]],
    chains: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    if not values:
        return {}

    total_len = sum(len(seq) for _, seq in chains)
    if len(values) != total_len:
        if len(chains) == 1 and len(values) >= len(chains[0][1]):
            cid, seq = chains[0]
            return {cid: values[: len(seq)]}
        return {}

    out: Dict[str, List[float]] = {}
    offset = 0
    for cid, seq in chains:
        n = len(seq)
        out[cid] = values[offset : offset + n]
        offset += n
    return out


def _extract_residue_plddt(
    out_dir: Path,
    chains: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    candidate_keys = (
        "residue_plddt",
        "residue_plddts",
        "token_plddt",
        "token_plddts",
        "plddt",
        "plddts",
    )

    for pattern in ("*confidence*.json", "*full*.json", "*.json"):
        for path in sorted(out_dir.rglob(pattern)):
            data = _load_json(path)
            vals = _walk_values(data, candidate_keys)
            per_chain = _split_per_chain(vals, chains)
            if per_chain:
                return per_chain
    for pattern in ("*confidence*.npz", "*full*.npz", "*.npz"):
        for path in sorted(out_dir.rglob(pattern)):
            vals = _npz_values(path, candidate_keys)
            per_chain = _split_per_chain(vals, chains)
            if per_chain:
                return per_chain
    return {}


def run_protenix_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: str = "protenix_mini",
) -> Dict[str, Any]:
    del device
    pred_name, chains = _normalise_inputs(pred_name, chains)
    if not chains:
        return {"metrics": {}, "residue_plddt": {}, "out_dir": None}

    cache_key = _cache_key(pred_name, chains, metric, seed, model_name)
    if cache_key in _CONF_CACHE:
        return _CONF_CACHE[cache_key]

    out_dir = _run_protenix(
        pred_name=pred_name,
        chains=chains,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
    )
    if out_dir is None:
        result = {"metrics": {}, "residue_plddt": {}, "out_dir": None}
        _CONF_CACHE[cache_key] = result
        return result

    summary_json = _first_json(out_dir, "*summary_confidence*.json")
    summary = _load_json(summary_json) if summary_json else None
    metrics = _scalar_metrics(summary)
    residue_plddt = _extract_residue_plddt(out_dir, chains)

    result = {
        "metrics": metrics,
        "residue_plddt": residue_plddt,
        "out_dir": str(out_dir),
    }
    _CONF_CACHE[cache_key] = result
    return result


def run_protenix_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: str = "protenix_mini",
) -> float:
    pred_name, chains = _normalise_inputs(pred_name, chains)
    if not chains:
        return 0.0

    cache_key = _cache_key(pred_name, chains, metric, seed, model_name)
    if cache_key in _SCALAR_CACHE:
        return _SCALAR_CACHE[cache_key]

    confidence = run_protenix_confidence_multichain(
        pred_name=pred_name,
        chains=chains,
        metric=metric,
        device=device,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
    )
    value = float(confidence.get("metrics", {}).get(metric, 0.0))
    _SCALAR_CACHE[cache_key] = value
    return value
