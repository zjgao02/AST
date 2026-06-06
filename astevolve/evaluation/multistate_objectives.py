from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _one_letter_residue(value: Any) -> str:
    text = str(value or "").upper().strip()
    if len(text) == 1:
        return text if text in "ACDEFGHIKLMNPQRSTVWY" else ""
    return _THREE_TO_ONE.get(text, "")


def _aa_class_members(class_name: str) -> str:
    table = {
        "aromatic": "YWHF",
        "polar": "STNQY",
        "polar_uncharged": "STNQY",
        "contextual_charge": "RKHDE",
        "flexible_small": "GSA",
        "positive": "RKH",
        "negative": "DE",
        "acidic": "DE",
        "basic": "RKH",
        "hydrophobic": "AILMFWVY",
        "charged": "RKHDE",
        "small": "GAS",
        "turn_loop": "GSPNDT",
        "amine_near": "DENQ",
        "catechol_near": "YHSTNQDE",
        "aromatic_support": "FYWILV",
        "calcium_ligand": "DEQNST",
    }
    return table.get(str(class_name or "").lower(), "")


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


def _region_indices_from_spec(spec: Dict[str, Any]) -> Optional[set[int]]:
    values = spec.get("indices")
    if values is None:
        values = spec.get("residues")
    if values is not None:
        try:
            out = {int(v) for v in values}
        except (TypeError, ValueError):
            out = set()
        if not bool(spec.get("zero_based", False)) and bool(spec.get("one_based", False)):
            out = {idx - 1 for idx in out if idx > 0}
        return out

    spans = spec.get("spans", spec.get("ranges", spec.get("residue_ranges")))
    if spans is None:
        return None
    out: set[int] = set()
    for raw in spans:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            start = int(raw[0])
            end = int(raw[1])
        except (TypeError, ValueError):
            continue
        if bool(spec.get("one_based", False)) and not bool(spec.get("zero_based", False)):
            start -= 1
            end -= 1
        for idx in range(max(0, start), max(0, end)):
            out.add(idx)
    return out


def _unit_region_candidates(unit: Dict[str, Any]) -> set[str]:
    candidates = {
        str(unit.get("asym_id") or "").lower(),
        str(unit.get("label") or "").lower(),
        str(unit.get("base_label") or "").lower(),
        str(unit.get("source_chain") or "").lower(),
        str(unit.get("kind") or "").lower(),
    }
    candidates.update(_kind_aliases(str(unit.get("kind") or "")))
    return {item for item in candidates if item}


def _region_filters_by_asym(
    state_result: Dict[str, Any],
    region_specs: Optional[List[Dict[str, Any]]],
) -> Dict[str, Optional[set[int]]]:
    if not region_specs:
        return {}
    units = _expand_entity_units(state_result)
    filters: Dict[str, Optional[set[int]]] = {}
    for spec in region_specs:
        if not isinstance(spec, dict):
            continue
        chain_id = str(
            spec.get("chain_id")
            or spec.get("source_chain")
            or spec.get("chain")
            or spec.get("asym_id")
            or spec.get("entity")
            or ""
        ).strip()
        wanted = chain_id.lower()
        indices = _region_indices_from_spec(spec)
        matched = False
        for unit in units:
            asym = str(unit.get("asym_id") or "")
            if not asym:
                continue
            if wanted and wanted not in _unit_region_candidates(unit):
                continue
            matched = True
            if indices is None:
                filters[asym] = None
            elif asym not in filters:
                filters[asym] = set(indices)
            elif filters.get(asym) is not None:
                filters.setdefault(asym, set()).update(indices)
        if not matched and chain_id and len(chain_id) <= 3:
            if indices is None:
                filters[chain_id] = None
            elif chain_id not in filters:
                filters[chain_id] = set(indices)
            elif filters.get(chain_id) is not None:
                filters.setdefault(chain_id, set()).update(indices)
    return filters


def _residue_allowed(residue: Dict[str, Any], filters: Dict[str, Optional[set[int]]]) -> bool:
    if not filters:
        return True
    chain = str(residue.get("chain") or "")
    if chain not in filters:
        return False
    allowed = filters[chain]
    if allowed is None:
        return True
    try:
        idx = int(residue.get("residue") or 0) - 1
    except (TypeError, ValueError):
        return False
    return idx in allowed


def _coverage(filters: Dict[str, Optional[set[int]]], contacted: set[Tuple[str, int]]) -> Optional[float]:
    if not filters:
        return None
    total = 0
    for residues in filters.values():
        if residues is not None:
            total += len(residues)
    if total <= 0:
        return None
    covered = 0
    for chain, zero_idx in contacted:
        allowed = filters.get(chain)
        if allowed is not None and zero_idx in allowed:
            covered += 1
    return _clamp01(float(covered) / float(total))


def _filtered_pair_item(
    item: Dict[str, Any],
    left_ids: List[str],
    right_ids: List[str],
    left_filters: Dict[str, Optional[set[int]]],
    right_filters: Dict[str, Optional[set[int]]],
) -> Optional[Dict[str, Any]]:
    residue_pairs = item.get("residue_pairs")
    if not residue_pairs:
        if left_filters or right_filters:
            return None
        return dict(item)

    contact_count = 0
    clash_count = 0
    residue_pair_count = 0
    plddt_values: List[float] = []
    left_contacted: set[Tuple[str, int]] = set()
    right_contacted: set[Tuple[str, int]] = set()
    examples: List[Dict[str, Any]] = []

    for pair in residue_pairs:
        if not isinstance(pair, dict):
            continue
        raw_left = pair.get("left", {}) or {}
        raw_right = pair.get("right", {}) or {}
        left_chain = str(raw_left.get("chain") or "")
        right_chain = str(raw_right.get("chain") or "")
        if left_chain in left_ids and right_chain in right_ids:
            oriented_left, oriented_right = raw_left, raw_right
        elif left_chain in right_ids and right_chain in left_ids:
            oriented_left, oriented_right = raw_right, raw_left
        else:
            continue
        if not _residue_allowed(oriented_left, left_filters):
            continue
        if not _residue_allowed(oriented_right, right_filters):
            continue

        contact_count += int(pair.get("contact_count") or 0)
        clash_count += int(pair.get("clash_count") or 0)
        residue_pair_count += 1
        for residue, contacted in ((oriented_left, left_contacted), (oriented_right, right_contacted)):
            plddt = _safe_float(residue.get("plddt"))
            if plddt is not None:
                plddt_values.append(float(plddt))
            try:
                contacted.add((str(residue.get("chain") or ""), int(residue.get("residue") or 0) - 1))
            except (TypeError, ValueError):
                pass
        if len(examples) < 10:
            examples.append(
                {
                    "left": oriented_left,
                    "right": oriented_right,
                    "contact_count": pair.get("contact_count"),
                    "clash_count": pair.get("clash_count"),
                    "min_distance": pair.get("min_distance"),
                }
            )

    if contact_count <= 0 and clash_count <= 0:
        return None
    return {
        "contact_count": int(contact_count),
        "residue_pair_count": int(residue_pair_count),
        "clash_count": int(clash_count),
        "interface_plddt_mean": _mean(plddt_values),
        "interface_plddt_min": min(plddt_values) if plddt_values else None,
        "left_region_coverage": _coverage(left_filters, left_contacted),
        "right_region_coverage": _coverage(right_filters, right_contacted),
        "contact_examples": examples,
    }


def _sum_pair_metrics(
    state_result: Dict[str, Any],
    left_selector: Any,
    right_selector: Any,
    left_region_specs: Optional[List[Dict[str, Any]]] = None,
    right_region_specs: Optional[List[Dict[str, Any]]] = None,
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

    left_filters = _region_filters_by_asym(state_result, left_region_specs)
    right_filters = _region_filters_by_asym(state_result, right_region_specs)
    if left_region_specs and not left_filters:
        warnings.append(f"could not resolve left region for selector: {left_selector}")
    if right_region_specs and not right_filters:
        warnings.append(f"could not resolve right region for selector: {right_selector}")

    if left_filters or right_filters:
        filtered: List[Dict[str, Any]] = []
        for item in selected:
            filtered_item = _filtered_pair_item(item, left_ids, right_ids, left_filters, right_filters)
            if filtered_item is not None:
                filtered.append(filtered_item)
        selected = filtered
        if not selected:
            warnings.append("no contacts remained after interface region filters")
            return {
                "available": False,
                "reason": "no_region_filtered_contacts",
                "left_region_required": bool(left_region_specs),
                "right_region_required": bool(right_region_specs),
                "left_region_resolved": (not left_region_specs) or bool(left_filters),
                "right_region_resolved": (not right_region_specs) or bool(right_filters),
            }, warnings

    plddt_values = [
        float(item["interface_plddt_mean"])
        for item in selected
        if _safe_float(item.get("interface_plddt_mean")) is not None
    ]
    left_coverages = [
        float(item["left_region_coverage"])
        for item in selected
        if _safe_float(item.get("left_region_coverage")) is not None
    ]
    right_coverages = [
        float(item["right_region_coverage"])
        for item in selected
        if _safe_float(item.get("right_region_coverage")) is not None
    ]
    return {
        "available": True,
        "contact_count": int(sum(int(item.get("contact_count") or 0) for item in selected)),
        "residue_pair_count": int(sum(int(item.get("residue_pair_count") or 0) for item in selected)),
        "clash_count": int(sum(int(item.get("clash_count") or 0) for item in selected)),
        "interface_plddt_mean": _mean(plddt_values),
        "left_region_coverage": _mean(left_coverages),
        "right_region_coverage": _mean(right_coverages),
        "region_filtered": bool(left_filters or right_filters),
    }, warnings


def _interface_selectors(spec: Dict[str, Any]) -> Tuple[Any, Any]:
    pair = spec.get("pair")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return pair[0], pair[1]
    return (
        spec.get("left", spec.get("protein", spec.get("binder"))),
        spec.get("right", spec.get("target", spec.get("ligand"))),
    )


def _interface_region_specs(
    spec: Dict[str, Any],
    side: str,
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    regions = spec.get("regions")
    value = None
    if isinstance(regions, dict):
        value = regions.get(side)
    if value is None:
        if side == "left":
            for key in ("left_region", "binder_region", "protein_region"):
                if spec.get(key) is not None:
                    value = spec.get(key)
                    break
        else:
            for key in ("right_region", "target_region", "ligand_region"):
                if spec.get(key) is not None:
                    value = spec.get(key)
                    break
    return _ast_region_specs(value, compiled, design_state)


def _interface_strength(
    state_result: Dict[str, Any],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]] = None,
    design_state: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any], List[str]]:
    left, right = _interface_selectors(spec)
    design_state = design_state or {}
    left_region_specs = _interface_region_specs(spec, "left", compiled, design_state)
    right_region_specs = _interface_region_specs(spec, "right", compiled, design_state)
    metrics, warnings = _sum_pair_metrics(
        state_result,
        left,
        right,
        left_region_specs=left_region_specs,
        right_region_specs=right_region_specs,
    )
    full_metrics, full_warnings = _sum_pair_metrics(state_result, left, right)
    full_warnings = [warning for warning in full_warnings if warning not in warnings]
    warnings.extend(full_warnings)
    if not metrics.get("available"):
        details = dict(metrics)
        details["full_interface"] = full_metrics
        return 0.0, details, warnings

    kind_hint = f"{left} {right}".lower()
    default_target = 8.0 if "ligand" in kind_hint or "dopamine" in kind_hint else 30.0
    contact_target = max(1.0, float(spec.get("contact_target", default_target)))
    residue_target = max(1.0, float(spec.get("residue_pair_target", max(3.0, contact_target / 5.0))))
    clash_target = max(1.0, float(spec.get("clash_tolerance", 3.0)))

    contact_score = _clamp01(math.log1p(float(metrics.get("contact_count") or 0)) / math.log1p(contact_target))
    residue_score = _clamp01(math.log1p(float(metrics.get("residue_pair_count") or 0)) / math.log1p(residue_target))
    plddt_score = _normalize_metric("plddt", metrics.get("interface_plddt_mean"))
    clash_score = 1.0 - _clamp01(float(metrics.get("clash_count") or 0) / clash_target)
    coverage_values = [
        float(value)
        for value in (metrics.get("left_region_coverage"), metrics.get("right_region_coverage"))
        if _safe_float(value) is not None
    ]
    coverage = _mean(coverage_values)
    coverage_target = _safe_float(spec.get("coverage_target"))
    coverage_score = None
    if coverage is not None and coverage_target is not None:
        coverage_score = _clamp01(float(coverage) / max(1e-6, float(coverage_target)))

    if coverage_score is None:
        score = (0.35 * contact_score) + (0.20 * residue_score) + (0.30 * plddt_score) + (0.15 * clash_score)
    else:
        score = (
            (0.25 * contact_score)
            + (0.15 * residue_score)
            + (0.25 * plddt_score)
            + (0.20 * coverage_score)
            + (0.15 * clash_score)
        )

    off_target_contact_count = max(0.0, float(full_metrics.get("contact_count") or 0) - float(metrics.get("contact_count") or 0))
    off_target_residue_pair_count = max(
        0.0,
        float(full_metrics.get("residue_pair_count") or 0) - float(metrics.get("residue_pair_count") or 0),
    )
    off_target_penalty_weight = float(spec.get("off_target_penalty_weight", 0.0) or 0.0)
    off_target_contact_tolerance = max(1.0, float(spec.get("off_target_contact_tolerance", contact_target) or contact_target))
    off_target_score = _clamp01(math.log1p(off_target_contact_count) / math.log1p(off_target_contact_tolerance))
    if off_target_penalty_weight > 0.0:
        score -= off_target_penalty_weight * off_target_score

    details = dict(metrics)
    details.update(
        {
            "contact_score": contact_score,
            "residue_pair_score": residue_score,
            "interface_plddt_score": plddt_score,
            "coverage": coverage,
            "coverage_target": coverage_target,
            "coverage_score": coverage_score,
            "clash_score": clash_score,
            "full_contact_count": full_metrics.get("contact_count"),
            "full_residue_pair_count": full_metrics.get("residue_pair_count"),
            "off_target_contact_count": off_target_contact_count,
            "off_target_residue_pair_count": off_target_residue_pair_count,
            "off_target_score": off_target_score,
            "off_target_penalty_weight": off_target_penalty_weight,
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


def _node_metric_value(item: Dict[str, Any], metric: str) -> Optional[float]:
    metric = str(metric or "plddt_mean")
    aliases = {
        "plddt": "plddt_mean",
        "mean": "plddt_mean",
        "min": "plddt_min",
        "max": "plddt_max",
    }
    key = aliases.get(metric, metric)
    return _safe_float(item.get(key))


def _score_region_confidence(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    names = _state_names(spec) or list(by_state)
    metric = str(spec.get("metric") or "plddt_mean")
    region_value = spec.get("region", spec.get("regions", spec.get("nodes", spec.get("node"))))
    nodes = _expand_region_names(region_value, design_state)
    if not nodes:
        return 0.0, {"metric": metric, "states": names, "nodes": []}, ["region_confidence needs region, regions, node, or nodes"]

    target_raw = _safe_float(spec.get("target"), None)
    target = _normalize_metric(metric, target_raw) if target_raw is not None else None
    values: Dict[str, Dict[str, Dict[str, float]]] = {}
    scores: List[float] = []
    warnings: List[str] = []

    for name in names:
        state = by_state.get(name)
        if not state:
            warnings.append(f"unknown state: {name}")
            continue
        summary = state.get("structure_metrics", {}) or {}
        node_plddt = summary.get("node_plddt", {}) or state.get("node_plddt", {}) or {}
        state_values: Dict[str, Dict[str, float]] = {}
        for node in nodes:
            item = node_plddt.get(node)
            if not isinstance(item, dict):
                continue
            raw = _node_metric_value(item, metric)
            if raw is None:
                continue
            normalized = _normalize_metric(metric, raw)
            score = _clamp01(normalized / target) if target and target > 0 else normalized
            state_values[str(node)] = {"raw": float(raw), "normalized": float(normalized), "score": float(score)}
            scores.append(float(score))
        if not state_values:
            warnings.append(f"{name}: no node confidence values for region/nodes {nodes}")
        values[name] = state_values

    if not scores:
        return 0.0, {"metric": metric, "states": names, "nodes": nodes, "values": values}, warnings
    return float(sum(scores) / len(scores)), {
        "metric": metric,
        "states": names,
        "nodes": nodes,
        "target": target_raw,
        "values": values,
    }, warnings


def _score_region_confidence_floor(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    names = _state_names(spec) or list(by_state)
    metric = str(spec.get("metric") or "plddt_min")
    region_value = spec.get("region", spec.get("regions", spec.get("nodes", spec.get("node"))))
    nodes = _expand_region_names(region_value, design_state)
    if not nodes:
        return 0.0, {"metric": metric, "states": names, "nodes": []}, [
            "region_confidence_floor needs region, regions, node, or nodes"
        ]

    target_raw = _safe_float(spec.get("target"), 70.0) or 70.0
    hard_floor_raw = _safe_float(spec.get("hard_floor"), None)
    if hard_floor_raw is None:
        hard_floor_raw = min(target_raw, _safe_float(spec.get("floor"), 50.0) or 50.0)
    target = _normalize_metric(metric, target_raw)
    hard_floor = _normalize_metric(metric, hard_floor_raw)

    values: Dict[str, Dict[str, Dict[str, float]]] = {}
    scores: List[float] = []
    failing: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for name in names:
        state = by_state.get(name)
        if not state:
            warnings.append(f"unknown state: {name}")
            continue
        summary = state.get("structure_metrics", {}) or {}
        node_plddt = summary.get("node_plddt", {}) or state.get("node_plddt", {}) or {}
        state_values: Dict[str, Dict[str, float]] = {}
        for node in nodes:
            item = node_plddt.get(node)
            if not isinstance(item, dict):
                continue
            raw = _node_metric_value(item, metric)
            if raw is None:
                continue
            normalized = _normalize_metric(metric, raw)
            if normalized < hard_floor:
                score = 0.25 * _clamp01(normalized / max(1e-6, hard_floor))
                failing.append({"state": name, "node": node, "raw": float(raw), "hard_floor": hard_floor_raw})
            else:
                score = _clamp01(normalized / max(1e-6, target))
            state_values[str(node)] = {"raw": float(raw), "normalized": float(normalized), "score": float(score)}
            scores.append(float(score))
        if not state_values:
            warnings.append(f"{name}: no node confidence values for region/nodes {nodes}")
        values[name] = state_values

    if not scores:
        return 0.0, {
            "metric": metric,
            "states": names,
            "nodes": nodes,
            "target": target_raw,
            "hard_floor": hard_floor_raw,
            "values": values,
        }, warnings

    aggregate = str(spec.get("aggregate", "min")).lower()
    score = min(scores) if aggregate in {"min", "floor", "strict"} else float(sum(scores) / len(scores))
    return _clamp01(score), {
        "metric": metric,
        "states": names,
        "nodes": nodes,
        "target": target_raw,
        "hard_floor": hard_floor_raw,
        "aggregate": aggregate,
        "values": values,
        "failing_nodes": failing,
    }, warnings


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
    if isinstance(design_state, dict):
        target = design_state.get("target", {}) or {}
        target_name = str(target.get("epitope_name") or "")
        existing_names = {str(spec.get("name") or "") for spec in out if isinstance(spec, dict)}
        if target_name and target_name in wanted and target_name not in existing_names:
            indices: List[int] = []
            for span in target.get("epitope_spans", []) or []:
                if not isinstance(span, (list, tuple)) or len(span) < 2:
                    continue
                try:
                    start = int(span[0])
                    end = int(span[1])
                except (TypeError, ValueError):
                    continue
                indices.extend(range(max(0, start), max(0, end)))
            out.append(
                {
                    "name": target_name,
                    "chain_id": str(target.get("chain_id", "T")),
                    "indices": sorted(set(indices)),
                    "zero_based": True,
                }
            )
    if isinstance(design_state, dict):
        constraints = design_state.get("design_constraints", {})
        region_tables: List[Any] = [design_state.get("entity_regions")]
        if isinstance(constraints, dict):
            region_tables.extend([constraints.get("target_epitopes"), constraints.get("decoy_epitopes")])
        named: Dict[str, Dict[str, Any]] = {}
        for table in region_tables:
            if isinstance(table, dict):
                for name, spec in table.items():
                    if isinstance(spec, dict):
                        named[str(name)] = dict(spec)
            elif isinstance(table, list):
                for item in table:
                    if isinstance(item, dict) and item.get("name"):
                        named[str(item["name"])] = dict(item)
        existing_names = {str(spec.get("name") or "") for spec in out if isinstance(spec, dict)}
        for name in names:
            if str(name) in existing_names:
                continue
            spec = named.get(str(name))
            if not isinstance(spec, dict):
                continue
            resolved = dict(spec)
            resolved.setdefault("name", str(name))
            if "entity" not in resolved and resolved.get("id"):
                resolved["entity"] = resolved.get("id")
            if "chain_id" not in resolved and resolved.get("source_chain"):
                resolved["chain_id"] = resolved.get("source_chain")
            resolved.setdefault("zero_based", True)
            out.append(resolved)
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


def _canonical_source_chain(unit: Dict[str, Any]) -> Optional[str]:
    source_chain = str(unit.get("source_chain") or "").strip()
    asym_id = str(unit.get("asym_id") or "").strip()
    if not source_chain or not asym_id:
        return None
    try:
        copy_index = int(unit.get("copy_index") or 1)
    except (TypeError, ValueError):
        copy_index = 1
    return f"{source_chain}:{max(1, copy_index)}"


def _transition_chain_map(state: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not state:
        return out
    for unit in _expand_entity_units(state):
        if not isinstance(unit, dict):
            continue
        asym_id = str(unit.get("asym_id") or "").strip()
        canonical = _canonical_source_chain(unit)
        if asym_id and canonical:
            out[asym_id] = canonical
    return out


def _transition_structure_input(state: Dict[str, Any], path: str) -> Any:
    chain_map = _transition_chain_map(state)
    if not chain_map:
        return path
    return {
        "path": path,
        "chain_map": chain_map,
        "source": str(state.get("name") or path),
    }


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
    apo_structure = _transition_structure_input(apo_state, apo_path)
    holo_structure = _transition_structure_input(holo_state, holo_path)
    report = evaluate_mechanistic_transition(
        apo_structure,
        holo_structure,
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
        "apo_chain_map": _transition_chain_map(apo_state),
        "holo_chain_map": _transition_chain_map(holo_state),
        "kinetic_path_score": report.get("kinetic_path_score", 0.0),
        "interpretability_report": report.get("interpretability_report", {}),
        "region_scores": report.get("region_scores", {}),
    }
    warnings = list((details["interpretability_report"] or {}).get("warnings", []) or [])
    return _clamp01(report.get("kinetic_path_score", 0.0)), details, warnings


def _strength_for_state(
    by_state: Dict[str, Dict[str, Any]],
    state_name: str,
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _interface_strength(by_state.get(state_name, {}), spec, compiled, design_state)


def _state_pair_names(spec: Dict[str, Any]) -> Tuple[str, str]:
    positive = str(
        spec.get("positive_state")
        or spec.get("on_state")
        or spec.get("state_a")
        or ""
    )
    negative = str(
        spec.get("negative_state")
        or spec.get("off_state")
        or spec.get("state_b")
        or ""
    )
    states = _state_names(spec)
    if (not positive or not negative) and len(states) >= 2:
        positive = positive or states[0]
        negative = negative or states[1]
    return positive, negative


ObjectiveScorer = Callable[
    [Dict[str, Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]],
    Tuple[float, Dict[str, Any], List[str]],
]
OBJECTIVE_REGISTRY: Dict[str, ObjectiveScorer] = {}


def _register_objective(*names: str) -> Callable[[ObjectiveScorer], ObjectiveScorer]:
    def decorator(func: ObjectiveScorer) -> ObjectiveScorer:
        for name in names:
            key = str(name).strip().lower()
            if key:
                OBJECTIVE_REGISTRY[key] = func
        return func
    return decorator


@_register_objective("interface_delta", "delta_interface", "state_interface_delta")
def _objective_interface_delta(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    positive_state, negative_state = _state_pair_names(spec)
    if not positive_state or not negative_state:
        return 0.0, {}, ["interface_delta needs positive_state/negative_state or a two-state states list"]

    positive_spec = dict(spec)
    negative_spec = dict(spec)
    positive_pair = spec.get("positive_pair", spec.get("on_pair"))
    negative_pair = spec.get("negative_pair", spec.get("off_pair"))
    if positive_pair is not None:
        positive_spec["pair"] = positive_pair
    if negative_pair is not None:
        negative_spec["pair"] = negative_pair
    for source_key, target_key in [
        ("positive_left_region", "left_region"),
        ("positive_right_region", "right_region"),
        ("positive_contact_target", "contact_target"),
        ("positive_residue_pair_target", "residue_pair_target"),
        ("positive_coverage_target", "coverage_target"),
        ("negative_left_region", "left_region"),
        ("negative_right_region", "right_region"),
        ("negative_contact_target", "contact_target"),
        ("negative_residue_pair_target", "residue_pair_target"),
        ("negative_coverage_target", "coverage_target"),
    ]:
        if source_key not in spec:
            continue
        if source_key.startswith("positive_"):
            positive_spec[target_key] = spec[source_key]
        else:
            negative_spec[target_key] = spec[source_key]

    positive_strength, positive_details, warnings_a = _strength_for_state(
        by_state, positive_state, positive_spec, compiled, design_state
    )
    negative_strength, negative_details, warnings_b = _strength_for_state(
        by_state, negative_state, negative_spec, compiled, design_state
    )
    direction = str(spec.get("direction", "decrease")).lower()
    if direction in {"increase", "up", "higher"}:
        delta = negative_strength - positive_strength
        favorable_low = positive_strength
        favorable_high = negative_strength
    else:
        delta = positive_strength - negative_strength
        favorable_low = negative_strength
        favorable_high = positive_strength

    min_delta = float(spec.get("min_delta", 0.05))
    target_delta = max(min_delta + 1e-6, float(spec.get("target_delta", 0.35)))
    delta_score = _clamp01((delta - min_delta) / (target_delta - min_delta))
    high_score = _clamp01(favorable_high)
    low_score = 1.0 - _clamp01(favorable_low)
    score = (0.55 * delta_score) + (0.25 * high_score) + (0.20 * low_score)

    contact_a = _safe_float(positive_details.get("contact_count"), 0.0) or 0.0
    contact_b = _safe_float(negative_details.get("contact_count"), 0.0) or 0.0
    contact_delta = contact_b - contact_a if direction in {"increase", "up", "higher"} else contact_a - contact_b
    contact_delta_target = _safe_float(spec.get("contact_delta_target"), None)
    contact_delta_score = None
    if contact_delta_target is not None:
        contact_delta_score = _clamp01(contact_delta / max(1e-6, float(contact_delta_target)))
        score = 0.75 * score + 0.25 * contact_delta_score

    return _clamp01(score), {
        "states": [positive_state, negative_state],
        "direction": direction,
        "positive_state": positive_state,
        "negative_state": negative_state,
        "positive_strength": positive_strength,
        "negative_strength": negative_strength,
        "strength_delta": delta,
        "min_delta": min_delta,
        "target_delta": target_delta,
        "delta_score": delta_score,
        "contact_count_delta": contact_delta,
        "contact_delta_target": contact_delta_target,
        "contact_delta_score": contact_delta_score,
        "positive_details": positive_details,
        "negative_details": negative_details,
    }, warnings_a + warnings_b


@_register_objective("competition_interface", "interface_competition", "competitive_binding")
def _objective_competition_interface(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    if not state_name:
        return 0.0, {}, ["competition_interface needs state"]
    winner_spec = dict(spec.get("winner") or spec.get("on") or {})
    loser_spec = dict(spec.get("loser") or spec.get("off") or {})
    if not winner_spec:
        winner_spec = {
            "pair": spec.get("winner_pair", spec.get("on_pair")),
            "left_region": spec.get("winner_left_region", spec.get("on_left_region")),
            "right_region": spec.get("winner_right_region", spec.get("on_right_region")),
            "contact_target": spec.get("winner_contact_target", spec.get("contact_target")),
            "residue_pair_target": spec.get("winner_residue_pair_target", spec.get("residue_pair_target")),
            "coverage_target": spec.get("winner_coverage_target", spec.get("coverage_target")),
        }
    if not loser_spec:
        loser_spec = {
            "pair": spec.get("loser_pair", spec.get("off_pair")),
            "left_region": spec.get("loser_left_region", spec.get("off_left_region")),
            "right_region": spec.get("loser_right_region", spec.get("off_right_region")),
            "contact_target": spec.get("loser_contact_target", spec.get("contact_target")),
            "residue_pair_target": spec.get("loser_residue_pair_target", spec.get("residue_pair_target")),
            "coverage_target": spec.get("loser_coverage_target", spec.get("coverage_target")),
        }
    winner_spec = {**spec, **{k: v for k, v in winner_spec.items() if v is not None}}
    loser_spec = {**spec, **{k: v for k, v in loser_spec.items() if v is not None}}
    winner_strength, winner_details, warnings_a = _interface_strength(
        by_state.get(state_name, {}), winner_spec, compiled, design_state
    )
    loser_strength, loser_details, warnings_b = _interface_strength(
        by_state.get(state_name, {}), loser_spec, compiled, design_state
    )
    score = (0.55 * _clamp01(winner_strength)) + (0.45 * (1.0 - _clamp01(loser_strength)))
    return _clamp01(score), {
        "state": state_name,
        "winner_strength": winner_strength,
        "loser_strength": loser_strength,
        "winner_details": winner_details,
        "loser_details": loser_details,
    }, warnings_a + warnings_b


@_register_objective("epitope_specificity", "hotspot_specificity", "interface_specificity")
def _objective_epitope_specificity(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    if not state_name:
        return 0.0, {}, ["epitope_specificity needs state"]

    target_spec = dict(spec)
    if spec.get("target_region") is not None:
        target_spec["right_region"] = spec.get("target_region")
    target_strength, target_details, warnings = _interface_strength(
        by_state.get(state_name, {}), target_spec, compiled, design_state
    )
    full_contact = max(0.0, _safe_float(target_details.get("full_contact_count"), 0.0) or 0.0)
    target_contact = max(0.0, _safe_float(target_details.get("contact_count"), 0.0) or 0.0)
    off_contact = max(0.0, _safe_float(target_details.get("off_target_contact_count"), full_contact - target_contact) or 0.0)
    specificity_ratio = target_contact / max(1.0, full_contact)
    target_fraction_goal = float(spec.get("target_fraction_goal", spec.get("specificity_target", 0.45)))
    specificity_score = _clamp01(specificity_ratio / max(1e-6, target_fraction_goal))

    avoid_score = None
    avoid_details: Dict[str, Any] = {}
    avoid_region = spec.get("avoid_region", spec.get("avoid_regions"))
    if avoid_region is not None:
        avoid_spec = dict(spec)
        avoid_spec["right_region"] = avoid_region
        avoid_strength, avoid_details, avoid_warnings = _interface_strength(
            by_state.get(state_name, {}), avoid_spec, compiled, design_state
        )
        avoid_score = 1.0 - _clamp01(avoid_strength)
        warnings.extend(avoid_warnings)

    score = (0.55 * _clamp01(target_strength)) + (0.35 * specificity_score)
    score += 0.10 * (avoid_score if avoid_score is not None else (1.0 - _clamp01(off_contact / max(1.0, float(spec.get("off_target_contact_tolerance", 120.0))))))
    return _clamp01(score), {
        "state": state_name,
        "target_region": spec.get("target_region", spec.get("right_region")),
        "avoid_region": avoid_region,
        "target_strength": target_strength,
        "target_contact_count": target_contact,
        "full_contact_count": full_contact,
        "off_target_contact_count": off_contact,
        "specificity_ratio": specificity_ratio,
        "target_fraction_goal": target_fraction_goal,
        "specificity_score": specificity_score,
        "avoid_score": avoid_score,
        "target_details": target_details,
        "avoid_details": avoid_details,
    }, warnings


def _interface_left_residue_names(
    state_result: Dict[str, Any],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any], List[str]]:
    left, right = _interface_selectors(spec)
    left_region_specs = _interface_region_specs(spec, "left", compiled, design_state)
    right_region_specs = _interface_region_specs(spec, "right", compiled, design_state)
    metrics, warnings = _sum_pair_metrics(
        state_result,
        left,
        right,
        left_region_specs=left_region_specs,
        right_region_specs=right_region_specs,
    )
    if not metrics.get("available"):
        return [], metrics, warnings

    pairs = ((state_result.get("structure_metrics", {}) or {}).get("interface", {}) or {}).get("pairs", {}) or {}
    left_ids = _selector_asym_ids(state_result, left)
    right_ids = _selector_asym_ids(state_result, right)
    left_filters = _region_filters_by_asym(state_result, left_region_specs)
    right_filters = _region_filters_by_asym(state_result, right_region_specs)
    residues: List[str] = []
    for left_id in left_ids:
        for right_id in right_ids:
            key = _pair_key(left_id, right_id, pairs)
            if key is None or not isinstance(pairs.get(key), dict):
                continue
            for pair in pairs[key].get("residue_pairs", []) or []:
                raw_left = pair.get("left", {}) or {}
                raw_right = pair.get("right", {}) or {}
                left_chain = str(raw_left.get("chain") or "")
                right_chain = str(raw_right.get("chain") or "")
                if left_chain in left_ids and right_chain in right_ids:
                    oriented_left, oriented_right = raw_left, raw_right
                elif left_chain in right_ids and right_chain in left_ids:
                    oriented_left, oriented_right = raw_right, raw_left
                else:
                    continue
                if not _residue_allowed(oriented_left, left_filters):
                    continue
                if not _residue_allowed(oriented_right, right_filters):
                    continue
                resname = _one_letter_residue(oriented_left.get("resname"))
                if resname:
                    residues.append(resname)
    return residues, metrics, warnings


@_register_objective("ligand_pocket_pharmacophore", "pocket_pharmacophore", "ion_pocket_pharmacophore")
def _objective_ligand_pocket_pharmacophore(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    if not state_name:
        return 0.0, {}, ["ligand_pocket_pharmacophore needs state"]
    state_result = by_state.get(state_name, {})
    strength, strength_details, warnings = _interface_strength(state_result, spec, compiled, design_state)
    residues, residue_metrics, residue_warnings = _interface_left_residue_names(state_result, spec, compiled, design_state)
    warnings.extend(residue_warnings)

    classes = spec.get("desired_residue_classes", {}) or {}
    if not isinstance(classes, dict):
        classes = {}
    class_scores: Dict[str, Any] = {}
    for class_name, members in classes.items():
        allowed = set("".join(_name_list(members)).upper())
        if not allowed:
            allowed = set(_aa_class_members(str(class_name)))
        hits = sum(1 for aa in residues if aa[:1] in allowed or aa in allowed)
        min_hits = int(_safe_float((spec.get("min_class_hits", {}) or {}).get(class_name), 1) or 1) if isinstance(spec.get("min_class_hits"), dict) else 1
        class_scores[str(class_name)] = {
            "allowed": sorted(allowed),
            "hits": hits,
            "min_hits": min_hits,
            "score": _clamp01(hits / max(1, min_hits)),
        }
    chemistry_score = (
        float(sum(item["score"] for item in class_scores.values()) / len(class_scores))
        if class_scores
        else 0.5
    )
    score = (0.60 * _clamp01(strength)) + (0.40 * chemistry_score)
    return _clamp01(score), {
        "state": state_name,
        "interface_strength": strength,
        "contact_residue_count": len(residues),
        "contact_residues": residues[:40],
        "chemistry_score": chemistry_score,
        "class_scores": class_scores,
        "strength_details": strength_details,
        "residue_metrics": residue_metrics,
    }, warnings


@_register_objective("confidence", "structural_confidence")
def _objective_confidence(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_confidence(by_state, spec)


@_register_objective("region_confidence", "node_confidence", "motif_confidence")
def _objective_region_confidence(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_region_confidence(by_state, spec, design_state)


@_register_objective("region_confidence_floor", "node_confidence_floor", "motif_confidence_floor")
def _objective_region_confidence_floor(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_region_confidence_floor(by_state, spec, design_state)


@_register_objective("interface_on", "preserve_interface", "bind", "binding")
def _objective_interface_on(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    score, details, warnings = _interface_strength(by_state.get(state_name, {}), spec, compiled, design_state)
    return score, {"state": state_name, **details}, warnings


@_register_objective("interface_off", "disrupt_interface", "anti_bind", "anti-binding", "anti_binding")
def _objective_interface_off(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    state_name = (_state_names(spec) or [""])[0]
    score, details, warnings = _interface_strength(by_state.get(state_name, {}), spec, compiled, design_state)
    if any("could not resolve" in warning for warning in warnings):
        return 0.0, {"state": state_name, "interface_strength": score, **details}, warnings
    return 1.0 - score, {"state": state_name, "interface_strength": score, **details}, warnings


@_register_objective("conf_change", "conformational_change")
def _objective_conf_change(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_conf_change(by_state, spec, compiled, design_state)


@_register_objective("mechanistic_transition", "kinetic_path", "transition_path")
def _objective_mechanistic_transition(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    return _score_mechanistic_transition(by_state, spec, compiled, design_state)


@_register_objective("preserve_motif", "motif_on")
def _objective_motif_on(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    motif_spec = dict(spec)
    if motif_spec.get("region") or motif_spec.get("regions") or motif_spec.get("node") or motif_spec.get("nodes"):
        motif_spec.setdefault("metric", "plddt_mean")
        return _score_region_confidence(by_state, motif_spec, design_state)
    motif_spec.setdefault("metric", "node_plddt_mean")
    return _score_confidence(by_state, motif_spec)


@_register_objective("disrupt_motif", "motif_off")
def _objective_motif_off(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    score, details, warnings = _objective_motif_on(by_state, spec, compiled, design_state)
    return 1.0 - score, {"motif_confidence": score, **details}, warnings


def supported_objective_types() -> List[str]:
    return sorted(OBJECTIVE_REGISTRY)


def _score_objective(
    by_state: Dict[str, Dict[str, Any]],
    spec: Dict[str, Any],
    compiled: Optional[Dict[str, Any]],
    design_state: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], List[str]]:
    kind = str(spec.get("type") or spec.get("kind") or "").strip().lower()
    scorer = OBJECTIVE_REGISTRY.get(kind)
    if scorer is not None:
        return scorer(by_state, spec, compiled, design_state)
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
            "supported_objective_types": supported_objective_types(),
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
        "supported_objective_types": supported_objective_types(),
    }
