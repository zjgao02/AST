from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from astevolve.evaluation.mechanistic_transition_evaluator import evaluate_mechanistic_transition
from astevolve.metrics.structure import _load_atoms, metric_value


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    numeric = _safe_float(value, 0.0) or 0.0
    return float(max(0.0, min(1.0, numeric)))


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in values if _safe_float(x) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _normalize_metric(metric: str, value: Any) -> float:
    val = _safe_float(value, 0.0) or 0.0
    key = str(metric or "").lower()
    if "plddt" in key:
        return _clamp01(val / 100.0 if abs(val) > 1.5 else val)
    if key in {"ptm", "iptm", "ranking_score", "dockq_proxy"}:
        return _clamp01(val)
    if key in {"has_clash", "clash", "clash_count"}:
        return 1.0 - _clamp01(val)
    return _clamp01(val)


def _asym_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"X{index + 1}"


def _expand_entity_units(state_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = state_result.get("entity_units")
    if isinstance(explicit, list) and explicit:
        return [dict(unit) for unit in explicit if isinstance(unit, dict)]

    units: List[Dict[str, Any]] = []
    reports = state_result.get("entities", []) or []
    unit_index = 0
    for entity_index, report in enumerate(reports, start=1):
        if not isinstance(report, dict):
            continue
        label = str(report.get("label") or f"entity{entity_index}")
        count = int(_safe_float(report.get("count"), 1) or 1)
        kind = str(report.get("kind") or "")
        for copy_index in range(1, max(1, count) + 1):
            unit_label = label if count == 1 else f"{label}_{copy_index}"
            units.append(
                {
                    "label": unit_label,
                    "base_label": label,
                    "kind": kind,
                    "copy_index": copy_index,
                    "asym_id": _asym_id(unit_index),
                }
            )
            unit_index += 1
    return units


def _kind_aliases(kind: str) -> List[str]:
    value = str(kind or "").lower()
    aliases = [value]
    if "protein" in value:
        aliases.extend(["protein", "proteinchain"])
    if "dna" in value:
        aliases.extend(["dna", "dnasequence"])
    if "rna" in value:
        aliases.extend(["rna", "rnasequence"])
    if "ligand" in value:
        aliases.extend(["ligand", "small_molecule", "molecule"])
    if "ion" in value:
        aliases.append("ion")
    return aliases


def _selector_asym_ids(state_result: Dict[str, Any], selector: Any) -> List[str]:
    units = _expand_entity_units(state_result)
    if selector is None:
        return [str(unit.get("asym_id")) for unit in units if unit.get("asym_id")]
    if isinstance(selector, (list, tuple, set)):
        out: List[str] = []
        for item in selector:
            for asym in _selector_asym_ids(state_result, item):
                if asym not in out:
                    out.append(asym)
        return out

    raw = str(selector).strip()
    if not raw:
        return []
    low = raw.lower()
    out: List[str] = []
    for unit in units:
        asym = str(unit.get("asym_id") or "")
        candidates = {
            asym.lower(),
            str(unit.get("label") or "").lower(),
            str(unit.get("base_label") or "").lower(),
            str(unit.get("source_chain") or "").lower(),
        }
        candidates.update(_kind_aliases(str(unit.get("kind") or "")))
        label_text = " ".join(candidates)
        if low in candidates or (len(low) >= 3 and low in label_text):
            if asym and asym not in out:
                out.append(asym)
    if out:
        return out
    return [raw] if len(raw) <= 3 else []


def _pair_key(left: str, right: str, pairs: Dict[str, Any]) -> Optional[str]:
    direct = f"{left}:{right}"
    reverse = f"{right}:{left}"
    if direct in pairs:
        return direct
    if reverse in pairs:
        return reverse
    return None


def _sum_pair_metrics(
    state_result: Dict[str, Any],
    left_selector: Any,
    right_selector: Any,
) -> Tuple[Dict[str, Any], List[str]]:
    summary = state_result.get("structure_metrics", {}) or {}
    interface = summary.get("interface", {}) or {}
    pairs = interface.get("pairs", {}) or {}
    warnings: List[str] = []

    if left_selector is None and right_selector is None:
        return {
            "available": bool(interface.get("available", False)),
            "contact_count": int(interface.get("total_contact_count") or 0),
            "residue_pair_count": int(interface.get("total_residue_pair_count") or 0),
            "clash_count": int(interface.get("clash_count") or 0),
            "interface_plddt_mean": interface.get("interface_plddt_mean"),
        }, warnings

    left_ids = _selector_asym_ids(state_result, left_selector)
    right_ids = _selector_asym_ids(state_result, right_selector)
    if not left_ids or not right_ids:
        warnings.append(f"could not resolve interface selectors: {left_selector}, {right_selector}")
        return {"available": False}, warnings

    selected: List[Dict[str, Any]] = []
    for left in left_ids:
        for right in right_ids:
            if left == right:
                continue
            key = _pair_key(left, right, pairs)
            if key is not None and isinstance(pairs.get(key), dict):
                selected.append(pairs[key])

    if not selected:
        warnings.append(f"no interface pair metrics for selectors: {left_selector}, {right_selector}")
        return {"available": False}, warnings

    plddt_values = [
        float(item["interface_plddt_mean"])
        for item in selected
        if _safe_float(item.get("interface_plddt_mean")) is not None
    ]
    return {
        "available": True,
        "contact_count": int(sum(int(item.get("contact_count") or 0) for item in selected)),
        "residue_pair_count": int(sum(int(item.get("residue_pair_count") or 0) for item in selected)),
        "clash_count": int(sum(int(item.get("clash_count") or 0) for item in selected)),
        "interface_plddt_mean": _mean(plddt_values),
    }, warnings


def _interface_selectors(spec: Dict[str, Any]) -> Tuple[Any, Any]:
    pair = spec.get("pair")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return pair[0], pair[1]
    return (
        spec.get("left", spec.get("protein", spec.get("binder"))),
        spec.get("right", spec.get("target", spec.get("ligand"))),
    )


def _interface_strength(state_result: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    left, right = _interface_selectors(spec)
    metrics, warnings = _sum_pair_metrics(state_result, left, right)
    if not metrics.get("available"):
        return 0.0, metrics, warnings

    kind_hint = f"{left} {right}".lower()
    default_target = 8.0 if "ligand" in kind_hint or "dopamine" in kind_hint else 30.0
    contact_target = max(1.0, float(spec.get("contact_target", default_target)))
    residue_target = max(1.0, float(spec.get("residue_pair_target", max(3.0, contact_target / 5.0))))
    clash_target = max(1.0, float(spec.get("clash_tolerance", 3.0)))

    contact_score = _clamp01(math.log1p(float(metrics.get("contact_count") or 0)) / math.log1p(contact_target))
    residue_score = _clamp01(math.log1p(float(metrics.get("residue_pair_count") or 0)) / math.log1p(residue_target))
    plddt_score = _normalize_metric("plddt", metrics.get("interface_plddt_mean"))
    clash_score = 1.0 - _clamp01(float(metrics.get("clash_count") or 0) / clash_target)
    score = (0.35 * contact_score) + (0.20 * residue_score) + (0.30 * plddt_score) + (0.15 * clash_score)
    details = dict(metrics)
    details.update(
        {
            "contact_score": contact_score,
            "residue_pair_score": residue_score,
            "interface_plddt_score": plddt_score,
            "clash_score": clash_score,
        }
    )
    return _clamp01(score), details, warnings


def _state_names(spec: Dict[str, Any]) -> List[str]:
    if spec.get("states") is not None:
        states = spec.get("states")
        return [str(s) for s in states] if isinstance(states, list) else [str(states)]
    if spec.get("state") is not None:
        return [str(spec.get("state"))]
    return []


def _score_confidence(by_state: Dict[str, Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
    names = _state_names(spec) or list(by_state)
    metric = str(spec.get("metric") or "plddt")
    values: Dict[str, float] = {}
    warnings: List[str] = []
    for name in names:
        state = by_state.get(name)
        if not state:
            warnings.append(f"unknown state: {name}")
            continue
        summary = state.get("structure_metrics", {}) or {}
        raw = metric_value(summary, metric, default=0.0)
        values[name] = _normalize_metric(metric, raw)
    if not values:
        return 0.0, {"metric": metric, "states": names, "values": values}, warnings
    return float(sum(values.values()) / len(values)), {"metric": metric, "states": names, "values": values}, warnings


def _region_names(spec: Dict[str, Any], design_state: Dict[str, Any]) -> Optional[List[str]]:
    region = spec.get("region", spec.get("regions"))
    if region is None:
        return None
    aliases = design_state.get("multistate_regions", {}) if isinstance(design_state, dict) else {}
    if isinstance(region, str):
        value = aliases.get(region, region)
    else:
        value = region
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def _region_indices_by_chain(compiled: Optional[Dict[str, Any]], names: Optional[List[str]]) -> Dict[str, Optional[set[int]]]:
    if not compiled or not names:
        return {}
    wanted = {str(name) for name in names}
    out: Dict[str, set[int]] = {}
    for seg in compiled.get("segments", []) or []:
        if getattr(seg, "name", None) not in wanted:
            continue
        chain_id = str(getattr(seg, "chain_id", ""))
        out.setdefault(chain_id, set()).update(int(i) for i in seg.indices())
    return out


def _expand_region_names(value: Any, design_state: Dict[str, Any]) -> List[str]:
    aliases = design_state.get("multistate_regions", {}) if isinstance(design_state, dict) else {}
    if value is None:
        return []
    if isinstance(value, str):
        expanded = aliases.get(value, value)
        if isinstance(expanded, list):
            return [str(item) for item in expanded]
        return [str(expanded)]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.extend(_expand_region_names(item, design_state))
        return out
    return []


def _ast_region_specs(
    value: Any,
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        if any(key in value for key in ("indices", "residues", "spans", "ranges", "residue_ranges")):
            return [dict(value)]
        if value.get("region") or value.get("regions"):
            return _ast_region_specs(value.get("region", value.get("regions")), compiled, design_state)
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.extend(_ast_region_specs(item, compiled, design_state))
            elif isinstance(item, str):
                out.extend(_ast_region_specs(item, compiled, design_state))
        return out

    names = _expand_region_names(value, design_state)
    if not names:
        return []
    if not compiled:
        return [{"name": name} for name in names]
    wanted = set(names)
    out: List[Dict[str, Any]] = []
    for seg in compiled.get("segments", []) or []:
        if getattr(seg, "name", None) not in wanted:
            continue
        out.append(
            {
                "name": getattr(seg, "name", "region"),
                "chain_id": str(getattr(seg, "chain_id", "A")),
                "indices": [int(i) for i in seg.indices()],
                "zero_based": True,
            }
        )
    return out


def _unit_key(unit: Dict[str, Any]) -> Tuple[str, int]:
    return (str(unit.get("source_chain") or unit.get("base_label") or unit.get("kind") or ""), int(unit.get("copy_index") or 1))


def _protein_ca_by_seq(cif_path: Optional[str], asym_id: str) -> Dict[int, np.ndarray]:
    if not cif_path:
        return {}
    coords: Dict[int, np.ndarray] = {}
    for atom in _load_atoms(str(cif_path)):
        if str(atom.get("asym") or "") != str(asym_id):
            continue
        atom_name = str(atom.get("atom") or "").strip("'\"")
        if atom_name != "CA":
            continue
        coords[int(atom.get("seq_id") or 1) - 1] = np.asarray(atom.get("xyz"), dtype=float)
    return coords


def _kabsch_rmsd(ref: np.ndarray, mob: np.ndarray, eval_ref: np.ndarray, eval_mob: np.ndarray) -> Optional[float]:
    if len(ref) < 3 or len(mob) < 3 or len(eval_ref) == 0 or len(eval_mob) == 0:
        return None
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    ref0 = ref - ref_center
    mob0 = mob - mob_center
    cov = mob0.T @ ref0
    try:
        u, _, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    det = np.linalg.det(u @ vt)
    corr = np.eye(3)
    corr[2, 2] = 1.0 if det >= 0 else -1.0
    rot = u @ corr @ vt
    moved = (eval_mob - mob_center) @ rot + ref_center
    diff = moved - eval_ref
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _region_rmsd(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
    spec: Dict[str, Any],
) -> Optional[float]:
    region_by_chain = _region_indices_by_chain(compiled, _region_names(spec, design_state))
    units_a = {_unit_key(unit): unit for unit in _expand_entity_units(state_a) if unit.get("source_chain")}
    units_b = {_unit_key(unit): unit for unit in _expand_entity_units(state_b) if unit.get("source_chain")}
    values: List[float] = []
    for key in sorted(set(units_a).intersection(units_b)):
        source_chain = key[0]
        asym_a = units_a[key].get("asym_id")
        asym_b = units_b[key].get("asym_id")
        if not asym_a or not asym_b:
            continue
        coords_a = _protein_ca_by_seq((state_a.get("structure_metrics") or {}).get("cif_path"), str(asym_a))
        coords_b = _protein_ca_by_seq((state_b.get("structure_metrics") or {}).get("cif_path"), str(asym_b))
        common = sorted(set(coords_a).intersection(coords_b))
        if len(common) < 3:
            continue
        region_indices = region_by_chain.get(source_chain)
        eval_indices = [idx for idx in common if region_indices is None or idx in region_indices]
        if not eval_indices:
            continue
        ref = np.asarray([coords_a[idx] for idx in common], dtype=float)
        mob = np.asarray([coords_b[idx] for idx in common], dtype=float)
        eval_ref = np.asarray([coords_a[idx] for idx in eval_indices], dtype=float)
        eval_mob = np.asarray([coords_b[idx] for idx in eval_indices], dtype=float)
        rmsd = _kabsch_rmsd(ref, mob, eval_ref, eval_mob)
        if rmsd is not None:
            values.append(float(rmsd))
    return _mean(values)


def _score_conf_change(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    states = _state_names(spec)
    if len(states) < 2:
        states = [str(spec.get("state_a") or ""), str(spec.get("state_b") or "")]
    if len(states) < 2 or not states[0] or not states[1]:
        return 0.0, {}, ["conf_change needs states, or state_a/state_b"]
    state_a = by_state.get(states[0])
    state_b = by_state.get(states[1])
    if not state_a or not state_b:
        return 0.0, {"states": states}, [f"unknown conf_change states: {states}"]

    rmsd = _region_rmsd(state_a, state_b, compiled, design_state, spec)
    if rmsd is not None:
        min_rmsd = float(spec.get("min_rmsd", 0.5))
        target_rmsd = max(min_rmsd + 1e-6, float(spec.get("target_rmsd", 3.0)))
        score = _clamp01((float(rmsd) - min_rmsd) / (target_rmsd - min_rmsd))
        return score, {"states": states, "region_rmsd": float(rmsd), "min_rmsd": min_rmsd, "target_rmsd": target_rmsd}, []

    interface_a = (state_a.get("structure_metrics", {}) or {}).get("interface", {}) or {}
    interface_b = (state_b.get("structure_metrics", {}) or {}).get("interface", {}) or {}
    delta = abs(float(interface_a.get("total_contact_count") or 0) - float(interface_b.get("total_contact_count") or 0))
    score = _clamp01(math.log1p(delta) / math.log1p(float(spec.get("contact_delta_target", 50.0))))
    return score, {"states": states, "contact_count_delta": delta, "fallback": "interface_contact_delta"}, [
        "conf_change used contact-delta fallback because region RMSD was unavailable"
    ]


def _state_cif_path(state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not state:
        return None
    summary = state.get("structure_metrics", {}) or {}
    return summary.get("cif_path") or state.get("cif_path")


def _score_mechanistic_transition(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    states = _state_names(spec)
    if len(states) < 2:
        states = [str(spec.get("state_a") or ""), str(spec.get("state_b") or "")]
    if len(states) < 2 or not states[0] or not states[1]:
        return 0.0, {}, ["mechanistic_transition needs states, or state_a/state_b"]
    apo_state = by_state.get(states[0])
    holo_state = by_state.get(states[1])
    apo_path = _state_cif_path(apo_state)
    holo_path = _state_cif_path(holo_state)
    if not apo_path or not holo_path:
        return 0.0, {"states": states, "apo_path": apo_path, "holo_path": holo_path}, [
            "mechanistic_transition needs CIF/PDB paths for both states"
        ]

    moving_value = spec.get("moving_regions", spec.get("moving_region", spec.get("region")))
    preserved_value = spec.get("preserved_regions", spec.get("preserved_region"))
    moving_regions = _ast_region_specs(moving_value, compiled, design_state)
    preserved_regions = _ast_region_specs(preserved_value, compiled, design_state)
    report = evaluate_mechanistic_transition(
        apo_path,
        holo_path,
        moving_regions=moving_regions,
        preserved_regions=preserved_regions,
        forbid=spec.get("forbid", ["chain_break", "severe_clash", "complete_unfolding"]),
        intermediate_states=spec.get("intermediate_states"),
        config=spec.get("config", spec),
    )
    details = {
        "states": states,
        "moving_region_count": len(moving_regions),
        "preserved_region_count": len(preserved_regions),
        "kinetic_path_score": report.get("kinetic_path_score", 0.0),
        "interpretability_report": report.get("interpretability_report", {}),
        "region_scores": report.get("region_scores", {}),
    }
    warnings = list((details["interpretability_report"] or {}).get("warnings", []) or [])
    return _clamp01(report.get("kinetic_path_score", 0.0)), details, warnings


def _score_objective(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    kind = str(spec.get("type") or spec.get("kind") or "").strip().lower()
    if kind in {"confidence", "structural_confidence"}:
        return _score_confidence(by_state, spec)
    if kind in {"interface_on", "preserve_interface", "bind", "binding"}:
        state_name = (_state_names(spec) or [""])[0]
        score, details, warnings = _interface_strength(by_state.get(state_name, {}), spec)
        return score, {"state": state_name, **details}, warnings
    if kind in {"interface_off", "disrupt_interface", "anti_bind", "anti-binding", "anti_binding"}:
        state_name = (_state_names(spec) or [""])[0]
        score, details, warnings = _interface_strength(by_state.get(state_name, {}), spec)
        if any("could not resolve" in warning for warning in warnings):
            return 0.0, {"state": state_name, "interface_strength": score, **details}, warnings
        return 1.0 - score, {"state": state_name, "interface_strength": score, **details}, warnings
    if kind in {"conf_change", "conformational_change"}:
        return _score_conf_change(by_state, spec, compiled, design_state)
    if kind in {"mechanistic_transition", "kinetic_path", "transition_path"}:
        return _score_mechanistic_transition(by_state, spec, compiled, design_state)
    if kind in {"preserve_motif", "motif_on"}:
        motif_spec = dict(spec)
        motif_spec.setdefault("metric", "node_plddt_mean")
        return _score_confidence(by_state, motif_spec)
    if kind in {"disrupt_motif", "motif_off"}:
        motif_spec = dict(spec)
        motif_spec.setdefault("metric", "node_plddt_mean")
        score, details, warnings = _score_confidence(by_state, motif_spec)
        return 1.0 - score, {"motif_confidence": score, **details}, warnings
    return 0.0, {}, [f"unsupported objective type: {kind}"]


def evaluate_multistate_objectives(
    complex_summary: Dict[str, Any],
    objective_specs: Optional[Sequence[Dict[str, Any]]],
    compiled: Optional[Dict[str, Any]] = None,
    design_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    specs = [dict(spec) for spec in (objective_specs or []) if isinstance(spec, dict)]
    by_state = {
        str(state.get("name")): state
        for state in (complex_summary.get("states", []) or [])
        if isinstance(state, dict) and state.get("name")
    }
    if not specs:
        return {
            "enabled": False,
            "weighted_score": 0.0,
            "weight_sum": 0.0,
            "normalized_score": 0.0,
            "loss": 0.0,
            "objectives": {},
            "warnings": [],
        }

    design_state = design_state or ((compiled or {}).get("_design_state", {}) if compiled else {})
    objectives: Dict[str, Any] = {}
    warnings: List[str] = []
    weighted = 0.0
    weight_sum = 0.0

    for index, spec in enumerate(specs, start=1):
        name = str(spec.get("name") or f"{spec.get('type', 'objective')}_{index}")
        weight = float(spec.get("weight", 1.0))
        score, details, obj_warnings = _score_objective(by_state, spec, compiled, design_state if isinstance(design_state, dict) else {})
        score = _clamp01(score)
        weighted += weight * score
        weight_sum += abs(weight)
        objectives[name] = {
            "type": spec.get("type", spec.get("kind")),
            "weight": weight,
            "score": score,
            "details": details,
            "warnings": obj_warnings,
        }
        warnings.extend(f"{name}: {warning}" for warning in obj_warnings)

    normalized = _clamp01(weighted / weight_sum) if weight_sum > 0 else 0.0
    return {
        "enabled": True,
        "weighted_score": float(weighted),
        "weight_sum": float(weight_sum),
        "normalized_score": float(normalized),
        "loss": float(1.0 - normalized),
        "objectives": objectives,
        "warnings": warnings,
    }
