from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from astevolve.runtime.paths import model_path, tmp_root


_SCALAR_CACHE: Dict[str, float] = {}
_CONF_CACHE: Dict[str, Dict[str, Any]] = {}
_DEFAULT_PROTENIX_ROOT = Path(
    os.environ.get("ASTEVOLVE_PROTENIX_ROOT", str(model_path("protenix")))
)


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
        path = tmp_root("protenix")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _protenix_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PROTENIX_ROOT_DIR", str(_DEFAULT_PROTENIX_ROOT))
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _find_conda_exe() -> str:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
    ]

    prefix = Path(sys.prefix)
    roots = [prefix]
    if prefix.parent.name == "envs":
        roots.append(prefix.parent.parent)
    for root in roots:
        candidates.extend([
            str(root / "Scripts" / "conda.exe"),
            str(root / "bin" / "conda"),
        ])

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return "conda"


def _protenix_num_workers() -> int:
    try:
        return max(0, int(os.environ.get("ASTEVOLVE_PROTENIX_NUM_WORKERS", "1")))
    except (TypeError, ValueError):
        return 1


def _cache_key(
    pred_name: str,
    chains: List[Tuple[str, str]],
    metric: str,
    seed: int,
    model_name: str,
) -> str:
    seq_key = "|".join(f"{cid}:{seq}" for cid, seq in chains)
    return f"{pred_name}|{seq_key}|{metric}|{seed}|{model_name}"


def _json_cache_key(
    pred_name: str,
    entities: List[Dict[str, Any]],
    constraint: Optional[Dict[str, Any]],
    covalent_bonds: Optional[List[Dict[str, Any]]],
    metric: str,
    seed: int,
    model_name: str,
) -> str:
    payload = {
        "pred_name": pred_name,
        "entities": entities,
        "constraint": constraint,
        "covalent_bonds": covalent_bonds,
        "metric": metric,
        "seed": int(seed),
        "model_name": model_name,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _copy_optional(payload: Dict[str, Any], source: Dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        if key in source and source[key] is not None:
            payload[key] = source[key]


def _normalise_count(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _ligand_value(entity: Dict[str, Any]) -> str:
    if entity.get("ligand"):
        return str(entity["ligand"])
    if entity.get("smiles"):
        return str(entity["smiles"])
    if entity.get("ccd"):
        ccd = str(entity["ccd"])
        return ccd if ccd.startswith("CCD_") else f"CCD_{ccd}"
    if entity.get("file"):
        value = str(entity["file"])
        return value if value.startswith("FILE_") else f"FILE_{value}"
    raise ValueError("Ligand entity needs one of: ligand, smiles, ccd, file")


def _normalise_entity(
    entity: Dict[str, Any],
    index: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(entity, dict):
        raise TypeError(f"Entity #{index} must be a dictionary")

    direct_keys = ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
    for key in direct_keys:
        if key in entity and isinstance(entity[key], dict):
            payload = dict(entity[key])
            count = _normalise_count(payload.get("count", 1))
            payload["count"] = count
            label = str(entity.get("id") or entity.get("name") or payload.pop("id", None) or f"entity{index}")
            payload.pop("name", None)
            entry = {key: payload}
            return entry, _entity_info(key, payload, label, count)

    kind = str(entity.get("type") or entity.get("kind") or "").strip().lower()
    label = str(entity.get("id") or entity.get("name") or f"entity{index}")
    count = _normalise_count(entity.get("count", 1))

    if kind in {"protein", "proteinchain", "protein_chain"}:
        sequence = str(entity.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"Protein entity '{label}' needs a sequence")
        payload: Dict[str, Any] = {"sequence": sequence, "count": count}
        _copy_optional(payload, entity, ["modifications", "msa"])
        return {"proteinChain": payload}, _entity_info("proteinChain", payload, label, count)

    if kind in {"dna", "dnasequence", "dna_sequence"}:
        sequence = str(entity.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"DNA entity '{label}' needs a sequence")
        payload = {"sequence": sequence, "count": count}
        _copy_optional(payload, entity, ["modifications"])
        return {"dnaSequence": payload}, _entity_info("dnaSequence", payload, label, count)

    if kind in {"rna", "rnasequence", "rna_sequence"}:
        sequence = str(entity.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"RNA entity '{label}' needs a sequence")
        payload = {"sequence": sequence, "count": count}
        _copy_optional(payload, entity, ["modifications"])
        return {"rnaSequence": payload}, _entity_info("rnaSequence", payload, label, count)

    if kind in {"ligand", "small_molecule", "small-molecule", "molecule"}:
        payload = {"ligand": _ligand_value(entity), "count": count}
        return {"ligand": payload}, _entity_info("ligand", payload, label, count)

    if kind == "ion":
        ion = str(entity.get("ion") or entity.get("ccd") or "").strip().upper()
        if ion.startswith("CCD_"):
            ion = ion[4:]
        if not ion:
            raise ValueError(f"Ion entity '{label}' needs an ion code")
        payload = {"ion": ion, "count": count}
        return {"ion": payload}, _entity_info("ion", payload, label, count)

    raise ValueError(
        f"Unknown Protenix entity type for '{label}'. Use protein, dna, rna, ligand, or ion."
    )


def _entity_info(key: str, payload: Dict[str, Any], label: str, count: int) -> Dict[str, Any]:
    sequence = str(payload.get("sequence") or "")
    token_length = len(sequence) if sequence else 1
    return {
        "kind": key,
        "label": label,
        "count": count,
        "token_length": token_length,
        "sequence": sequence,
    }


def _normalise_entities(
    entities: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    sequences: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    polymer_units: List[Tuple[str, str]] = []
    metric_units: List[Tuple[str, str]] = []

    for index, raw_entity in enumerate(entities, start=1):
        entry, info = _normalise_entity(raw_entity, index)
        sequences.append(entry)
        report.append({
            "entity": index,
            "label": info["label"],
            "kind": info["kind"],
            "count": info["count"],
            "token_length": info["token_length"],
        })

        for copy_idx in range(1, int(info["count"]) + 1):
            label = info["label"] if int(info["count"]) == 1 else f"{info['label']}_{copy_idx}"
            if info["kind"] in {"proteinChain", "dnaSequence", "rnaSequence"}:
                polymer_units.append((label, info["sequence"]))
                metric_units.append((label, info["sequence"]))
            else:
                metric_units.append((label, "X"))

    return sequences, report, polymer_units, metric_units


def _normalise_constraint(constraint: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not constraint:
        return None
    if "constraint" in constraint and isinstance(constraint["constraint"], dict):
        return dict(constraint["constraint"])
    out: Dict[str, Any] = {}
    for key in ("contact", "pocket"):
        if key in constraint and constraint[key]:
            out[key] = constraint[key]
    return out or dict(constraint)


def _write_input_json(path: Path, pred_name: str, chains: List[Tuple[str, str]]) -> None:
    entities = [
        {"type": "protein", "id": cid, "sequence": seq, "count": 1}
        for cid, seq in chains
    ]
    write_protenix_complex_input_json(path, pred_name, entities)


def write_protenix_complex_input_json(
    path: Path | str,
    pred_name: str,
    entities: List[Dict[str, Any]],
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sequences, report, polymer_units, metric_units = _normalise_entities(entities)
    job: Dict[str, Any] = {"sequences": sequences, "name": pred_name}
    if covalent_bonds:
        job["covalent_bonds"] = covalent_bonds
    norm_constraint = _normalise_constraint(constraint)
    if norm_constraint:
        job["constraint"] = norm_constraint
    data = [job]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {
        "input_json": str(out_path),
        "entities": report,
        "polymer_units": polymer_units,
        "metric_units": metric_units,
    }


def _run_protenix(
    pred_name: str,
    chains: List[Tuple[str, str]],
    seed: int,
    model_name: str,
    conda_env: str,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Optional[Path]:
    entities = [
        {"type": "protein", "id": cid, "sequence": seq, "count": 1}
        for cid, seq in chains
    ]
    return _run_protenix_complex(
        pred_name=pred_name,
        entities=entities,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )[0]


def _run_protenix_complex(
    pred_name: str,
    entities: List[Dict[str, Any]],
    seed: int,
    model_name: str,
    conda_env: str,
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    root = _tmp_root()
    run_dir = root / pred_name
    input_json_path = run_dir / "input.json"
    out_dir = run_dir / "output"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = write_protenix_complex_input_json(
        input_json_path,
        pred_name,
        entities,
        constraint=constraint,
        covalent_bonds=covalent_bonds,
    )

    cmd = [
        _find_conda_exe(),
        "run",
        "-n",
        conda_env,
        "python",
        str(Path(__file__).with_name("protenix_predict_worker.py")),
        "--input",
        str(input_json_path),
        "--out-dir",
        str(out_dir),
        "--model-name",
        model_name,
        "--seeds",
        str(seed),
        "--num-workers",
        str(_protenix_num_workers()),
    ]
    if use_msa is not None:
        cmd.extend(["--use-msa", "true" if use_msa else "false"])
    if cycle is not None:
        cmd.extend(["--cycle", str(int(cycle))])
    if step is not None:
        cmd.extend(["--step", str(int(step))])
    if sample is not None:
        cmd.extend(["--sample", str(int(sample))])
    if use_default_params is not None:
        cmd.extend(["--use-default-params", "true" if use_default_params else "false"])

    run_timeout = int(timeout or os.environ.get("ASTEVOLVE_PROTENIX_TIMEOUT", "600"))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=run_timeout,
            env=_protenix_env(),
            cwd=str(run_dir),
        )
    except subprocess.TimeoutExpired as exc:
        print(f"[protenix] prediction timed out ({run_timeout}s)")
        stdout = exc.stdout.decode(errors="ignore") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="ignore") if isinstance(exc.stderr, bytes) else exc.stderr
        if stdout:
            print(f"[protenix] stdout before timeout: {stdout[-1000:]}")
        if stderr:
            print(f"[protenix] stderr before timeout: {stderr[-1000:]}")
        if _prediction_available(out_dir):
            print("[protenix] salvaging completed prediction files after timeout")
            return out_dir, preview
        return None, preview
    except FileNotFoundError:
        print("[protenix] conda or protenix not found in PATH")
        return None, preview

    if result.returncode != 0:
        print(f"[protenix] failed (rc={result.returncode})")
        if result.stdout:
            print(f"[protenix] stdout: {_head_tail_text(result.stdout)}")
        if result.stderr:
            print(f"[protenix] stderr: {_head_tail_text(result.stderr)}")
        if _prediction_available(out_dir):
            print("[protenix] salvaging prediction files despite nonzero return code")
            return out_dir, preview
        return None, preview

    if not _prediction_available(out_dir):
        _report_missing_prediction(out_dir, result)
        return None, preview

    return out_dir, preview


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[protenix] failed to parse {path}: {exc}")
        return None


def _first_json(out_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(out_dir.rglob(pattern))
    return files[0] if files else None


def _first_cif(out_dir: Path) -> Optional[Path]:
    files = sorted(out_dir.rglob("*.cif"))
    return files[0] if files else None


def _prediction_available(out_dir: Path) -> bool:
    return bool(_first_json(out_dir, "*summary_confidence*.json") or _first_cif(out_dir))


def _tail_text(value: Optional[str], limit: int = 12000) -> str:
    text = str(value or "")
    return text[-limit:] if len(text) > limit else text


def _head_tail_text(value: Optional[str], limit: int = 12000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    half = max(1000, limit // 2)
    return f"{text[:half]}\n... <trimmed {len(text) - (2 * half)} chars> ...\n{text[-half:]}"


def _read_error_files(out_dir: Path, limit: int = 12000) -> str:
    err_dir = out_dir / "ERR"
    if not err_dir.exists():
        return ""

    chunks: List[str] = []
    for path in sorted(err_dir.rglob("*.txt"))[:5]:
        try:
            chunks.append(f"{path}:\n{_tail_text(path.read_text(encoding='utf-8', errors='ignore'), limit)}")
        except OSError:
            continue
    return "\n\n".join(chunks)


def _report_missing_prediction(out_dir: Path, result: subprocess.CompletedProcess[str]) -> None:
    print("[protenix] finished without CIF or summary_confidence output")
    errors = _read_error_files(out_dir)
    if errors:
        print(f"[protenix] ERR files:\n{errors}")
    if result.stdout:
        print(f"[protenix] stdout tail: {_tail_text(result.stdout)}")
    if result.stderr:
        print(f"[protenix] stderr tail: {_tail_text(result.stderr)}")


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


def _maybe_scale_metric(metric_name: str, value: float) -> float:
    if "plddt" in metric_name.lower() and abs(value) <= 1.5:
        return float(value) * 100.0
    return float(value)


def _scalar_metrics(summary: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(summary, dict):
        return out
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _chain_metrics(summary: Any, chains: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(summary, dict) or not chains:
        return out

    for key, value in summary.items():
        if not str(key).startswith("chain_") or str(key).startswith("chain_pair_"):
            continue
        vals = _numeric_list(value)
        if not vals or len(vals) != len(chains):
            continue
        metric_name = str(key)[len("chain_"):]
        out[metric_name] = {
            cid: _maybe_scale_metric(metric_name, float(vals[i]))
            for i, (cid, _) in enumerate(chains)
        }
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


def _atom_site_rows_from_cif(path: Path) -> Iterable[Dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return

    headers: List[str] = []
    in_atom_loop = False
    data_started = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            if data_started:
                break
            continue

        if line == "loop_":
            if data_started:
                break
            headers = []
            in_atom_loop = False
            continue

        if line.startswith("_"):
            if data_started:
                break
            if line.startswith("_atom_site."):
                headers.append(line.split()[0])
                in_atom_loop = True
            elif in_atom_loop:
                break
            continue

        if not in_atom_loop or not headers:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        data_started = True
        try:
            tokens = shlex.split(line, posix=False)
        except ValueError:
            tokens = line.split()
        if len(tokens) < len(headers):
            continue
        yield {headers[i]: tokens[i] for i in range(len(headers))}


def _extract_residue_plddt_from_cif(
    out_dir: Path,
    chains: List[Tuple[str, str]],
) -> Dict[str, List[float]]:
    if not chains:
        return {}

    for path in sorted(out_dir.rglob("*.cif")):
        by_asym: Dict[str, Dict[int, List[float]]] = {}
        asym_order: List[str] = []

        for row in _atom_site_rows_from_cif(path):
            asym = row.get("_atom_site.label_asym_id") or row.get("_atom_site.auth_asym_id")
            seq_raw = row.get("_atom_site.label_seq_id") or row.get("_atom_site.auth_seq_id")
            b_raw = row.get("_atom_site.B_iso_or_equiv")
            if not asym or not seq_raw or not b_raw or seq_raw in {".", "?"}:
                continue
            try:
                seq_id = int(float(seq_raw))
                b_val = float(b_raw)
            except ValueError:
                continue
            if seq_id <= 0:
                continue
            if asym not in by_asym:
                by_asym[asym] = {}
                asym_order.append(asym)
            by_asym[asym].setdefault(seq_id, []).append(b_val)

        if not by_asym:
            continue

        per_chain: Dict[str, List[float]] = {}
        for chain_idx, (cid, seq) in enumerate(chains):
            if chain_idx >= len(asym_order):
                break
            residues = by_asym.get(asym_order[chain_idx], {})
            all_vals = [v for vals in residues.values() for v in vals]
            if not all_vals:
                continue
            fallback = float(sum(all_vals) / len(all_vals))
            values: List[float] = []
            for residue_idx in range(1, len(seq) + 1):
                vals = residues.get(residue_idx)
                if vals:
                    values.append(float(sum(vals) / len(vals)))
                else:
                    values.append(fallback)
            per_chain[cid] = values

        if per_chain:
            return per_chain

    return {}


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
    per_chain = _extract_residue_plddt_from_cif(out_dir, chains)
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
    conda_env: str = "pytorch",
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
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
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    if out_dir is None:
        result = {"metrics": {}, "residue_plddt": {}, "out_dir": None}
        _CONF_CACHE[cache_key] = result
        return result

    summary_json = _first_json(out_dir, "*summary_confidence*.json")
    cif_path = _first_cif(out_dir)
    summary = _load_json(summary_json) if summary_json else None
    metrics = _scalar_metrics(summary)
    chain_metrics = _chain_metrics(summary, chains)
    residue_plddt = _extract_residue_plddt(out_dir, chains)

    result = {
        "metrics": metrics,
        "chain_metrics": chain_metrics,
        "residue_plddt": residue_plddt,
        "summary_json": str(summary_json) if summary_json else None,
        "cif_path": str(cif_path) if cif_path else None,
        "out_dir": str(out_dir),
    }
    _CONF_CACHE[cache_key] = result
    return result


def run_protenix_confidence_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    constraint: Optional[Dict[str, Any]] = None,
    covalent_bonds: Optional[List[Dict[str, Any]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: str = "pytorch",
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run Protenix on a general complex.

    Entity examples:

    - {"type": "protein", "id": "A", "sequence": "..."}
    - {"type": "dna", "id": "D1", "sequence": "ATGC"}
    - {"type": "ligand", "id": "L", "ccd": "TCY"}
    - {"type": "ligand", "id": "L", "smiles": "..."}
    - {"type": "ion", "id": "MG", "ion": "MG"}
    """
    del device
    entities = list(entities or [])
    if not entities:
        return {
            "metrics": {},
            "chain_metrics": {},
            "residue_plddt": {},
            "entities": [],
            "input_json": None,
            "out_dir": None,
        }

    pred_name = str(pred_name or "protenix_complex")
    sequences, report, polymer_units, metric_units = _normalise_entities(entities)
    normalised_entities = [
        {"entity": sequence_entry, "report": report_entry}
        for sequence_entry, report_entry in zip(sequences, report)
    ]

    cache_key = _json_cache_key(
        pred_name,
        normalised_entities,
        _normalise_constraint(constraint),
        covalent_bonds,
        metric,
        seed,
        model_name,
    )
    if cache_key in _CONF_CACHE:
        return _CONF_CACHE[cache_key]

    out_dir, preview = _run_protenix_complex(
        pred_name=pred_name,
        entities=entities,
        seed=seed,
        model_name=model_name,
        conda_env=conda_env,
        constraint=constraint,
        covalent_bonds=covalent_bonds,
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    if out_dir is None:
        result = {
            "metrics": {},
            "chain_metrics": {},
            "residue_plddt": {},
            "entities": report,
            "polymer_units": polymer_units,
            "metric_units": metric_units,
            "input_json": preview.get("input_json"),
            "out_dir": None,
        }
        _CONF_CACHE[cache_key] = result
        return result

    summary_json = _first_json(out_dir, "*summary_confidence*.json")
    cif_path = _first_cif(out_dir)
    summary = _load_json(summary_json) if summary_json else None
    metrics = _scalar_metrics(summary)
    chain_metrics = _chain_metrics(summary, metric_units)
    residue_plddt = _extract_residue_plddt(out_dir, polymer_units)

    result = {
        "metrics": metrics,
        "chain_metrics": chain_metrics,
        "residue_plddt": residue_plddt,
        "entities": report,
        "polymer_units": polymer_units,
        "metric_units": metric_units,
        "input_json": preview.get("input_json"),
        "summary_json": str(summary_json) if summary_json else None,
        "cif_path": str(cif_path) if cif_path else None,
        "out_dir": str(out_dir),
    }
    _CONF_CACHE[cache_key] = result
    return result


def run_protenix_plddt_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    metric: str = "plddt",
    **kwargs: Any,
) -> float:
    confidence = run_protenix_confidence_complex(
        pred_name=pred_name,
        entities=entities,
        metric=metric,
        **kwargs,
    )
    return float(confidence.get("metrics", {}).get(metric, 0.0))


def run_protenix_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    metric: str = "plddt",
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: str = "pytorch",
    timeout: Optional[int] = None,
    use_msa: Optional[bool] = None,
    cycle: Optional[int] = None,
    step: Optional[int] = None,
    sample: Optional[int] = None,
    use_default_params: Optional[bool] = None,
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
        timeout=timeout,
        use_msa=use_msa,
        cycle=cycle,
        step=step,
        sample=sample,
        use_default_params=use_default_params,
    )
    value = float(confidence.get("metrics", {}).get(metric, 0.0))
    _SCALAR_CACHE[cache_key] = value
    return value
