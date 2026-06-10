from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astevolve.core.protein_lang import Blueprint, Node
from astevolve.search.inner_opt import SAConfig, optimize_multichain
from astevolve.knowledge.registry import load_external_knowledge_provider
from astevolve.runtime.conda import resolve_protenix_conda_env
from astevolve.semantic_graph import build_semantic_graph_summary
from .memory_update import update_internal_memory

from .design_state import (
    PROJECT_ROOT,
    binder_domain_order,
    binder_sequence,
    flatten_binder_parts,
    load_design_state,
    segment_spans,
    state_domain_aliases,
    state_domain_segment_keys,
)
AA_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
AA_NO_CYS = set("ADEFGHIKLMNPQRSTVWY")
MUTATION_OPS = {
    "point",
    "block",
    "segment_resample",
    "swap",
    "site_resample",
    "region_shuffle",
    "motif_graft",
    "segment_mutagenesis",
    "cdr_resample",
    "pocket_motif_swap",
    "linker_length_perturb",
    "domain_length_perturb",
}
MAX_DESIGN_REGIONS = 8
MAX_REGION_TARGETS = 6
MAX_REGION_RESIDUES = 16

KIND_LENGTH_RANGES: Dict[str, Tuple[int, int]] = {
    "cdr": (4, 24),
    "linker": (1, 20),
    "framework": (1, 80),
}


def _safe_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def load_memory_yaml(memory_path: Optional[str] = None) -> Dict[str, Any]:
    yaml = _safe_import_yaml()
    if yaml is None:
        return {}

    candidates: List[Path] = []
    if memory_path:
        p = Path(memory_path)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
    candidates.append(PROJECT_ROOT / "memory.yaml")

    for path in candidates:
        try:
            if path.exists():
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def resolve_memory_path(memory_path: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if memory_path:
        p = Path(memory_path)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
    candidates.append(PROJECT_ROOT / "memory.yaml")

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_case_path(raw_path: Any, design_state_path: Optional[str] = None) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if design_state_path:
        base = Path(design_state_path)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base.parent / path
    return PROJECT_ROOT / path


def load_case_sheet(state: Dict[str, Any], design_state_path: Optional[str] = None) -> Dict[str, Any]:
    candidates: List[Path] = []
    configured = _resolve_case_path(state.get("case_sheet_path"), design_state_path)
    if configured:
        candidates.append(configured)
    default_sheet = _resolve_case_path("case_sheet.json", design_state_path)
    if default_sheet:
        candidates.append(default_sheet)

    for path in candidates:
        try:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".yaml", ".yml"}:
                yaml = _safe_import_yaml()
                if yaml is None:
                    continue
                data = yaml.safe_load(text)
            else:
                data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def compact_case_sheet(case_sheet: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(case_sheet, dict) or not case_sheet:
        return {}
    return {
        "schema_version": case_sheet.get("schema_version"),
        "case_name": case_sheet.get("case_name"),
        "readiness": case_sheet.get("readiness", {}),
        "design_goal": case_sheet.get("design_goal", {}),
        "state_success_criteria": case_sheet.get("state_success_criteria", {}),
        "residue_level_constraints": case_sheet.get("residue_level_constraints", {}),
        "objective_thresholds": case_sheet.get("objective_thresholds", {}),
        "information_gaps": case_sheet.get("information_gaps", []),
        "provisional_assumptions": case_sheet.get("provisional_assumptions", []),
    }


def extract_memory_bias(memory: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    policy = state["mutation_policy"]
    linker_parts = state.get("binder", {}).get("linker_segments", [])
    out: Dict[str, Any] = {
        "preferred_edit_order": list(policy.get("preferred_edit_order", [])),
        "linker_default_sequence": "".join(p[2] for p in linker_parts if len(p) >= 3),
        "linker_gs_min": 0.60,
        "linker_hydrophobic_max": 0.15,
        "linker_charged_max": 0.20,
        "cdr_favored_sparse": ["Y", "W", "H", "N", "Q", "S", "T", "R", "D", "E"],
    }

    try:
        stable = memory.get("stable_priors", {})
        scfv_prior = stable.get("scfv_prior", {})
        aa_prior = stable.get("aa_prior", {})
        linker_prior = stable.get("linker_prior", {})

        general = scfv_prior.get("general", {})
        if isinstance(general.get("preferred_edit_order"), list):
            out["preferred_edit_order"] = general["preferred_edit_order"]

        if isinstance(linker_prior.get("default_sequence"), str):
            out["linker_default_sequence"] = linker_prior["default_sequence"]

        comp_targets = linker_prior.get("composition_targets", {})
        if "gly_ser_fraction_preferred_min" in comp_targets:
            out["linker_gs_min"] = float(comp_targets["gly_ser_fraction_preferred_min"])
        if "hydrophobic_fraction_preferred_max" in comp_targets:
            out["linker_hydrophobic_max"] = float(comp_targets["hydrophobic_fraction_preferred_max"])
        if "charged_fraction_preferred_max" in comp_targets:
            out["linker_charged_max"] = float(comp_targets["charged_fraction_preferred_max"])

        cdr_design = scfv_prior.get("cdr_design_bias", {})
        if isinstance(cdr_design.get("favored_residues_sparse"), list):
            out["cdr_favored_sparse"] = cdr_design["favored_residues_sparse"]

        subprefs = aa_prior.get("substitution_preferences", {})
        if isinstance(subprefs.get("globally_disfavored_new_positions"), list):
            out["globally_disfavored_new_positions"] = subprefs["globally_disfavored_new_positions"]
    except Exception:
        pass

    return out


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _strategy_tree(strategy: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("strategy_tree", "design_tree", "node_tree"):
        tree = strategy.get(key)
        if isinstance(tree, dict):
            return tree
    return {}


def _layout_plan(strategy: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("layout_plan", "domain_layout", "node_layout"):
        plan = strategy.get(key)
        if isinstance(plan, dict):
            return plan
    return {}


def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _safe_float(value, default)))


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, _safe_int(value, default)))


def _canonical_residue_list(value: Any, *, allow_cys: bool = False) -> List[str]:
    allowed = AA_CANONICAL if allow_cys else AA_NO_CYS
    out: List[str] = []
    for aa in _name_list(value):
        aa = aa.upper()
        if len(aa) == 1 and aa in allowed and aa not in out:
            out.append(aa)
        if len(out) >= MAX_REGION_RESIDUES:
            break
    return out


def _canonical_position_list(value: Any, limit: int = 64) -> List[int]:
    out: List[int] = []
    raw = value if isinstance(value, list) else []
    for item in raw:
        try:
            pos = int(round(float(item)))
        except (TypeError, ValueError):
            continue
        if pos >= 0 and pos not in out:
            out.append(pos)
        if len(out) >= limit:
            break
    return out


def _canonical_motif_list(value: Any, limit: int = 12) -> List[str]:
    raw = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    out: List[str] = []
    for item in raw:
        motif = "".join(ch for ch in str(item).upper() if ch in AA_CANONICAL)
        if 2 <= len(motif) <= 48 and motif not in out:
            out.append(motif)
        if len(out) >= limit:
            break
    return out


def _canonical_domain_order(value: Any, state: Dict[str, Any]) -> List[str]:
    allowed = list(state_domain_segment_keys(state).keys())
    fallback = binder_domain_order(state)
    aliases = state_domain_aliases(state)
    if not isinstance(value, list):
        return fallback

    out: List[str] = []
    for item in value:
        text = str(item)
        canonical = aliases.get(text, text)
        if canonical in allowed and canonical not in out:
            out.append(canonical)
    for item in fallback:
        if item not in out:
            out.append(item)
    return out or fallback


def _node_length_range(node_name: str, kind: str) -> Tuple[int, int]:
    return KIND_LENGTH_RANGES.get(kind, (1, 80))


def _sanitize_length_range(value: Any, node_name: str, kind: str) -> Optional[List[int]]:
    if not (isinstance(value, list) and len(value) == 2):
        return None
    lo_bound, hi_bound = _node_length_range(node_name, kind)
    lo = _clamp_int(value[0], lo_bound, lo_bound, hi_bound)
    hi = _clamp_int(value[1], hi_bound, lo_bound, hi_bound)
    if lo > hi:
        lo, hi = hi, lo
    return [lo, hi]


def _sanitize_mutation_ops(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    out: Dict[str, float] = {}
    for op, weight in value.items():
        op = str(op)
        if op in MUTATION_OPS:
            out[op] = _clamp_float(weight, 0.0, 0.0, 1.0)
    if not out or sum(out.values()) <= 0:
        return None
    return out


def _ss_code(value: Any) -> Optional[str]:
    text = str(value).strip().lower()
    if text in {"h", "helix", "alpha", "alpha_helix"}:
        return "H"
    if text in {"e", "beta", "strand", "sheet", "beta_strand"}:
        return "E"
    if text in {"l", "loop", "coil", "turn", "flexible_loop"}:
        return "L"
    return None


def _sanitize_site_anchor(value: Any, node_name: str, kind: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    rel_positions = []
    for item in value.get("relative_positions", value.get("positions", [])) or []:
        pos = _safe_int(item, -1)
        if 0 <= pos < _node_length_range(node_name, kind)[1] and pos not in rel_positions:
            rel_positions.append(pos)
    if rel_positions:
        out["relative_positions"] = rel_positions[:8]

    if isinstance(value.get("relative_ranges"), list):
        ranges = []
        for raw in value["relative_ranges"]:
            if isinstance(raw, list) and len(raw) == 2:
                start = _clamp_int(raw[0], 0, 0, _node_length_range(node_name, kind)[1])
                end = _clamp_int(raw[1], start + 1, start + 1, _node_length_range(node_name, kind)[1])
                ranges.append([start, end])
        if ranges:
            out["relative_ranges"] = ranges[:4]

    out["weight"] = _clamp_float(value.get("weight", value.get("priority_boost", 2.0)), 2.0, 0.25, 6.0)
    residues = _canonical_residue_list(value.get("favored_residues", []))
    if residues:
        out["favored_residues"] = residues
    classes = _name_list(value.get("favored_residue_classes", []))[:8]
    if classes:
        out["favored_residue_classes"] = classes
    return out


def sanitize_strategy_for_ast(state: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Constrain LLM output to ASTevolve's protein-design strategy language."""
    if not isinstance(strategy, dict):
        strategy = {}
    cleaned = deepcopy(strategy)
    plan = dict(_layout_plan(cleaned))
    segment_meta = _segment_metadata(state)

    plan["binder_domain_order"] = _canonical_domain_order(plan.get("binder_domain_order"), state)
    regions = plan.get("design_regions", plan.get("regions", []))
    if not isinstance(regions, list):
        regions = []

    clean_regions: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for idx, region in enumerate(regions[:MAX_DESIGN_REGIONS], start=1):
        if not isinstance(region, dict):
            continue

        targets = _region_targets(state, region, segment_meta)[:MAX_REGION_TARGETS]
        if not targets:
            rejected.append({"name": str(region.get("name", f"region_{idx}")), "reason": "no valid target nodes"})
            continue

        out: Dict[str, Any] = {
            "name": str(region.get("name") or f"design_region_{idx}")[:80],
            "role": str(region.get("role") or region.get("intent") or "")[:240],
            "position": _clamp_int(region.get("position"), idx, 1, MAX_DESIGN_REGIONS),
            "bind_to": targets,
            "enabled": _as_bool(region.get("enabled"), True),
            "mutable": _as_bool(region.get("mutable"), True),
            "priority_boost": _clamp_float(region.get("priority_boost", 1.0), 1.0, 0.01, 5.0),
            "mutation_rate": _clamp_float(region.get("mutation_rate", 0.05), 0.05, 0.0, 0.30),
            "max_mutations_per_step": _clamp_int(region.get("max_mutations_per_step"), 2, 1, 16),
            "policy_weight": _clamp_float(region.get("policy_weight", 0.7), 0.7, 0.0, 1.0),
        }

        mut_ops = _sanitize_mutation_ops(region.get("mutation_ops"))
        if mut_ops is not None:
            out["mutation_ops"] = mut_ops

        for field in ("hotspot_positions", "anchor_positions", "mutable_positions", "protected_positions"):
            positions = _canonical_position_list(region.get(field))
            if positions:
                out[field] = positions

        for field in ("graft_motifs", "motif_candidates"):
            motifs = _canonical_motif_list(region.get(field))
            if motifs:
                out[field] = motifs

        phase = str(region.get("operator_phase") or "").strip().lower()
        if phase in {"explore", "refine", "stabilize"}:
            out["operator_phase"] = phase
        if "large_jump" in region:
            out["large_jump"] = _as_bool(region.get("large_jump"), False)
        if isinstance(region.get("design_points"), dict):
            out["design_points"] = dict(region["design_points"])

        favored = _canonical_residue_list(region.get("favored_residues", []))
        if favored:
            out["favored_residues"] = favored
        disfavored = _canonical_residue_list(region.get("disfavored_residues", []), allow_cys=True)
        if "C" not in disfavored:
            disfavored.append("C")
        out["disfavored_residues"] = disfavored[:MAX_REGION_RESIDUES]

        for field in ("favored_residue_classes", "disfavored_residue_classes"):
            values = _name_list(region.get(field, []))[:8]
            if values:
                out[field] = values

        length_budget = region.get("length_budget")
        if length_budget is not None:
            out["length_budget"] = _clamp_int(length_budget, 0, 1, 72)

        for field in ("target_lengths", "length_deltas", "length_ranges", "node_weights"):
            raw = region.get(field)
            if not isinstance(raw, dict):
                continue
            vals: Dict[str, Any] = {}
            for node_name in targets:
                if node_name not in raw:
                    continue
                kind = segment_meta[node_name]["kind"]
                if field == "length_ranges":
                    sanitized = _sanitize_length_range(raw[node_name], node_name, kind)
                    if sanitized is not None:
                        vals[node_name] = sanitized
                elif field == "target_lengths":
                    lo, hi = _node_length_range(node_name, kind)
                    vals[node_name] = _clamp_int(raw[node_name], segment_meta[node_name]["length"], lo, hi)
                elif field == "length_deltas":
                    vals[node_name] = _clamp_int(raw[node_name], 0, -8, 8)
                elif field == "node_weights":
                    vals[node_name] = _clamp_float(raw[node_name], 1.0, 0.05, 5.0)
            if vals:
                out[field] = vals

        if "length_range" in region and len(targets) == 1:
            sanitized = _sanitize_length_range(region["length_range"], targets[0], segment_meta[targets[0]]["kind"])
            if sanitized is not None:
                out["length_range"] = sanitized
        if "target_length" in region and len(targets) == 1:
            lo, hi = _node_length_range(targets[0], segment_meta[targets[0]]["kind"])
            out["target_length"] = _clamp_int(region["target_length"], segment_meta[targets[0]]["length"], lo, hi)
        if "length_mutable" in region:
            out["length_mutable"] = _as_bool(region.get("length_mutable"), False)
        if "allow_framework_length_change" in region:
            out["allow_framework_length_change"] = False

        ss = _ss_code(region.get("secondary_structure", region.get("ss", None)))
        if ss is not None:
            out["secondary_structure"] = ss

        site_anchors = region.get("site_anchors", {})
        if isinstance(site_anchors, dict):
            clean_anchors = {}
            for node_name in targets:
                anchor = _sanitize_site_anchor(site_anchors.get(node_name), node_name, segment_meta[node_name]["kind"])
                if anchor:
                    clean_anchors[node_name] = anchor
            if clean_anchors:
                out["site_anchors"] = clean_anchors

        clean_regions.append(out)

    plan["design_regions"] = clean_regions
    plan.pop("regions", None)

    ss_priors = plan.get("secondary_structure_priors", {})
    if isinstance(ss_priors, dict):
        clean_ss = {}
        for node_name, ss_value in ss_priors.items():
            if node_name in segment_meta:
                code = _ss_code(ss_value)
                if code:
                    clean_ss[node_name] = code
        plan["secondary_structure_priors"] = clean_ss
    else:
        plan["secondary_structure_priors"] = {}

    cleaned["layout_plan"] = plan
    cleaned["strategy_schema_report"] = {
        "active_region_count": len(clean_regions),
        "rejected_regions": rejected,
        "allowed_nodes": list(segment_meta.keys()),
        "domain_order": plan["binder_domain_order"],
    }
    return cleaned


def _iter_strategy_nodes(
    node: Dict[str, Any],
    inherited_mutable: bool = True,
    path: Tuple[str, ...] = (),
):
    order = 0

    def walk(cur: Dict[str, Any], parent_mutable: bool, cur_path: Tuple[str, ...]):
        nonlocal order
        if not isinstance(cur, dict):
            return

        name = str(cur.get("name") or cur.get("node") or "")
        if "mutable" in cur:
            mutable = _as_bool(cur.get("mutable"), parent_mutable)
        elif "enabled" in cur:
            mutable = _as_bool(cur.get("enabled"), parent_mutable)
        else:
            mutable = parent_mutable

        order += 1
        next_path = cur_path + ((name,) if name else ())
        yield cur, mutable, order, next_path

        for child in cur.get("children", []) or []:
            if isinstance(child, dict):
                yield from walk(child, mutable, next_path)

    yield from walk(node, inherited_mutable, path)


def _segment_metadata(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for name, kind, seq in flatten_binder_parts(state):
        meta[str(name)] = {"kind": str(kind), "length": len(seq)}
    return meta


def _copy_node_field(policy: Dict[str, Any], node: Dict[str, Any], field: str) -> None:
    if field in node and field not in policy:
        policy[field] = node[field]


def _name_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    return []


def _segment_names_by_domain(state: Dict[str, Any], domain_name: str) -> List[str]:
    key = state_domain_segment_keys(state).get(str(domain_name), "")
    if not key:
        return []
    return [str(x[0]) for x in state["binder"].get(key, []) if len(x) >= 3]


def _region_targets(
    state: Dict[str, Any],
    region: Dict[str, Any],
    segment_meta: Dict[str, Dict[str, Any]],
) -> List[str]:
    explicit: List[str] = []
    for key in ("bind_to", "nodes", "segments", "target_nodes", "covers"):
        explicit.extend(_name_list(region.get(key)))

    targets: List[str] = []
    for name in explicit:
        if name in segment_meta and name not in targets:
            targets.append(name)

    if not targets:
        domain = region.get("domain") or region.get("parent_domain")
        kind_filter = region.get("kind_filter") or region.get("node_kind") or region.get("kind")
        domain_names = _segment_names_by_domain(state, str(domain)) if domain else list(segment_meta)
        for name in domain_names:
            if name not in segment_meta:
                continue
            if kind_filter and segment_meta[name]["kind"] != str(kind_filter):
                continue
            targets.append(name)

    max_nodes = _safe_int(region.get("max_nodes"), 0)
    if max_nodes > 0:
        targets = targets[:max_nodes]
    return targets


def _combine_unique(old: Any, new: Any) -> List[str]:
    out: List[str] = []
    for value in _name_list(old) + _name_list(new):
        if value not in out:
            out.append(value)
    return out


def _augment_large_step_ops_for_policy(policy: Dict[str, Any], kind: str, node_name: str) -> None:
    if not _as_bool(policy.get("large_jump"), False):
        return
    ops = policy.get("mutation_ops")
    if not isinstance(ops, dict):
        ops = {}
    kind_text = str(kind or "").lower()
    name_text = str(node_name or "").lower()
    if kind_text == "cdr" or "cdr" in name_text:
        additions = {"cdr_resample": 0.12, "motif_graft": 0.10, "segment_mutagenesis": 0.10}
    elif kind_text == "linker" or "linker" in name_text:
        additions = {"linker_length_perturb": 0.12, "segment_resample": 0.08, "region_shuffle": 0.04}
    elif kind_text in {"pocket", "ligand", "dna_contact"} or any(token in name_text for token in ("pocket", "groove", "efhand", "loop")):
        additions = {"pocket_motif_swap": 0.12, "motif_graft": 0.08, "segment_mutagenesis": 0.08}
    elif kind_text in {"hinge", "relay", "framework", "helix"}:
        additions = {"domain_length_perturb": 0.08, "segment_mutagenesis": 0.06}
    else:
        additions = {"segment_mutagenesis": 0.08, "motif_graft": 0.04}
    for op, weight in additions.items():
        ops.setdefault(op, weight)
    policy["mutation_ops"] = ops


def _augment_large_step_ops(
    policies: Dict[str, Dict[str, Any]],
    segment_meta: Dict[str, Dict[str, Any]],
) -> None:
    for node_name, policy in policies.items():
        if not isinstance(policy, dict):
            continue
        kind = segment_meta.get(node_name, {}).get("kind", policy.get("kind", ""))
        _augment_large_step_ops_for_policy(policy, str(kind), str(node_name))


def _merge_region_policy(
    base: Dict[str, Any],
    region: Dict[str, Any],
    node_name: str,
    node_weight: float,
    region_order: int,
) -> None:
    region_name = str(region.get("name") or f"layout_region_{region_order}")
    role = region.get("role") or region.get("edit_intent") or region.get("intent")
    base["enabled"] = _as_bool(region.get("enabled"), _as_bool(base.get("enabled"), True))
    base["mutable"] = _as_bool(region.get("mutable"), _as_bool(base.get("mutable"), True))
    base["layout_position"] = min(_safe_int(base.get("layout_position"), region_order), region_order)

    old_priority = _safe_float(base.get("priority_boost", base.get("priority", 1.0)), 1.0)
    region_priority = _safe_float(region.get("priority_boost", region.get("priority", 1.0)), 1.0)
    base["priority_boost"] = max(0.01, old_priority * region_priority * max(0.05, node_weight))

    if role and "edit_intent" not in base:
        base["edit_intent"] = str(role)

    for field in (
        "target_length",
        "length",
        "length_delta",
        "length_range",
        "min_length",
        "max_length",
        "length_mutable",
        "mutation_rate",
        "mutation_ops",
        "max_mutations_per_step",
        "aa_weights",
        "fill_residues",
        "policy_weight",
        "confidence",
        "allow_framework_length_change",
        "secondary_structure",
        "position_weights",
        "hotspot_positions",
        "anchor_positions",
        "mutable_positions",
        "protected_positions",
        "graft_motifs",
        "motif_candidates",
        "operator_phase",
        "large_jump",
        "design_points",
    ):
        if field in region:
            base[field] = region[field]

    for field in (
        "favored_residues",
        "disfavored_residues",
        "favored_residue_classes",
        "disfavored_residue_classes",
    ):
        if field in region:
            base[field] = _combine_unique(base.get(field, []), region.get(field, []))

    target_lengths = region.get("target_lengths")
    if isinstance(target_lengths, dict) and node_name in target_lengths:
        base["target_length"] = target_lengths[node_name]
        base["length_mutable"] = True

    length_deltas = region.get("length_deltas")
    if isinstance(length_deltas, dict) and node_name in length_deltas:
        base["length_delta"] = length_deltas[node_name]
        base["length_mutable"] = True

    node_ranges = region.get("length_ranges")
    if isinstance(node_ranges, dict) and node_name in node_ranges:
        base["length_range"] = node_ranges[node_name]
        base["length_mutable"] = True

    if "length_bias" in region:
        base["length_bias"] = region["length_bias"]

    site_anchors = region.get("site_anchors")
    if isinstance(site_anchors, dict):
        node_anchor = site_anchors.get(node_name)
        if isinstance(node_anchor, dict):
            base["site_anchors"] = node_anchor
            base["favored_residues"] = _combine_unique(
                base.get("favored_residues", []),
                node_anchor.get("favored_residues", []),
            )
            base["favored_residue_classes"] = _combine_unique(
                base.get("favored_residue_classes", []),
                node_anchor.get("favored_residue_classes", []),
            )

    regions = list(base.get("layout_regions", []))
    if region_name not in regions:
        regions.append(region_name)
    base["layout_regions"] = regions


def _apply_region_length_budget(
    region: Dict[str, Any],
    targets: List[str],
    policies: Dict[str, Dict[str, Any]],
    segment_meta: Dict[str, Dict[str, Any]],
) -> None:
    if not targets or "length_budget" not in region:
        return
    budget = _safe_int(region.get("length_budget"), 0)
    if budget <= 0:
        return

    node_weights = region.get("node_weights", {})
    if not isinstance(node_weights, dict):
        node_weights = {}
    raw_weights = []
    for name in targets:
        raw_weights.append(max(0.01, _safe_float(node_weights.get(name), segment_meta[name]["length"])))
    total_weight = sum(raw_weights) or 1.0

    assigned: Dict[str, int] = {}
    remaining = int(budget)
    for idx, name in enumerate(targets):
        if idx == len(targets) - 1:
            target_len = max(1, remaining)
        else:
            target_len = max(1, int(round(budget * raw_weights[idx] / total_weight)))
            remaining -= target_len
        assigned[name] = target_len

    node_ranges = region.get("length_ranges", {})
    if not isinstance(node_ranges, dict):
        node_ranges = {}

    for name, target_len in assigned.items():
        kind = segment_meta[name]["kind"]
        lo, hi = _node_length_range(name, kind)
        if name in node_ranges:
            sanitized_range = _sanitize_length_range(node_ranges[name], name, kind)
            if sanitized_range is not None:
                lo, hi = sanitized_range
        target_len = max(lo, min(hi, int(target_len)))
        policies.setdefault(name, {})["target_length"] = target_len
        policies[name]["length_mutable"] = True


def _apply_layout_plan_to_policies(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
    policies: Dict[str, Dict[str, Any]],
    memory_bias: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    plan = _layout_plan(strategy)
    regions = plan.get("design_regions", plan.get("regions", [])) if plan else []
    if not isinstance(regions, list):
        regions = []

    segment_meta = _segment_metadata(state)
    active_regions: List[Dict[str, Any]] = []

    for order, region in enumerate(regions, start=1):
        if not isinstance(region, dict) or not _as_bool(region.get("enabled"), True):
            continue
        targets = _region_targets(state, region, segment_meta)
        if not targets:
            continue

        node_weights = region.get("node_weights", {})
        if not isinstance(node_weights, dict):
            node_weights = {}

        _apply_region_length_budget(region, targets, policies, segment_meta)
        for name in targets:
            base = policies.setdefault(
                name,
                {
                    "node_name": name,
                    "kind": segment_meta[name]["kind"],
                    "current_length": segment_meta[name]["length"],
                    "mutable": True,
                    "enabled": True,
                    "priority_boost": 1.0,
                },
            )
            _merge_region_policy(
                base,
                region,
                name,
                _safe_float(node_weights.get(name), 1.0),
                _safe_int(region.get("position"), order),
            )

        active_regions.append(
            {
                "name": str(region.get("name") or f"layout_region_{order}"),
                "role": region.get("role", ""),
                "targets": targets,
                "position": _safe_int(region.get("position"), order),
            }
        )

    if policies:
        edit_entries = []
        for name, policy in policies.items():
            if not _as_bool(policy.get("enabled"), _as_bool(policy.get("mutable"), False)):
                continue
            edit_entries.append(
                (
                    _safe_int(policy.get("layout_position", policy.get("tree_order", 9999)), 9999),
                    -_safe_float(policy.get("priority_boost", 1.0), 1.0),
                    name,
                )
            )
        edit_entries.sort()
        edit_order = [name for _pos, _priority, name in edit_entries]
    else:
        edit_order = list(memory_bias.get("preferred_edit_order", []))

    return {"active_regions": active_regions}, edit_order


def _collect_strategy_tree_policies(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
    memory_bias: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    tree = _strategy_tree(strategy)
    if not tree:
        return {}, []

    segment_meta = _segment_metadata(state)
    policies: Dict[str, Dict[str, Any]] = {}
    edit_entries: List[Tuple[float, int, str]] = []

    for node, inherited_mutable, order, path in _iter_strategy_nodes(tree):
        name = str(node.get("name") or node.get("node") or "")
        if name not in segment_meta:
            continue

        policy = dict(node.get("edit_policy") or {})
        for field in (
            "target_length",
            "length",
            "length_delta",
            "length_range",
            "min_length",
            "max_length",
            "length_mutable",
            "mutable",
            "enabled",
            "priority",
            "priority_boost",
            "mutation_rate",
            "mutation_ops",
            "max_mutations_per_step",
            "favored_residues",
            "disfavored_residues",
            "favored_residue_classes",
            "disfavored_residue_classes",
            "aa_weights",
            "fill_residues",
            "seed_sequence",
            "template_sequence",
            "policy_weight",
            "confidence",
            "edit_intent",
            "allow_framework_length_change",
            "secondary_structure",
            "position_weights",
            "hotspot_positions",
            "anchor_positions",
            "mutable_positions",
            "protected_positions",
            "graft_motifs",
            "motif_candidates",
            "operator_phase",
            "large_jump",
            "design_points",
            "site_anchors",
        ):
            _copy_node_field(policy, node, field)

        mutable = _as_bool(policy.get("mutable"), inherited_mutable)
        enabled = _as_bool(policy.get("enabled"), mutable)
        priority = _safe_float(
            policy.get("priority_boost", policy.get("priority", 1.0)),
            1.0,
        )

        policy.update(
            {
                "node_name": name,
                "kind": segment_meta[name]["kind"],
                "current_length": segment_meta[name]["length"],
                "mutable": mutable,
                "enabled": enabled,
                "priority_boost": priority,
                "tree_order": order,
                "tree_path": list(path),
            }
        )
        policies[name] = policy

        if enabled and mutable:
            edit_entries.append((priority, order, name))

    if edit_entries:
        edit_entries.sort(key=lambda x: (-x[0], x[1]))
        edit_order = [name for _priority, _order, name in edit_entries]
    else:
        edit_order = list(memory_bias.get("preferred_edit_order", []))

    return policies, edit_order


def normalize_strategy_tree(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
    memory_bias: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(strategy)
    policies, edit_order = _collect_strategy_tree_policies(state, strategy, memory_bias)
    layout_summary, layout_edit_order = _apply_layout_plan_to_policies(
        state,
        strategy,
        policies,
        memory_bias,
    )
    _augment_large_step_ops(policies, _segment_metadata(state))
    if layout_edit_order:
        edit_order = layout_edit_order
    if not policies:
        normalized.setdefault("node_edit_policies", {})
        return normalized

    normalized["_tree_policy_active"] = True
    normalized["_layout_plan_active"] = bool(layout_summary.get("active_regions"))
    normalized["layout_summary"] = layout_summary
    normalized["node_edit_policies"] = policies
    if not normalized.get("preferred_edit_order"):
        normalized["preferred_edit_order"] = edit_order
    return normalized


def _sanitize_aa_sequence(seq: Any) -> str:
    return "".join(aa for aa in str(seq).upper() if aa in AA_CANONICAL)


def _fill_residues(kind: str, policy: Dict[str, Any]) -> str:
    explicit = _sanitize_aa_sequence(policy.get("fill_residues", ""))
    if explicit:
        return explicit
    if kind == "linker":
        return "GGGGS"
    if kind == "cdr":
        return "YSGNQ"
    return "S"


def _repeat_to_length(seed: str, length: int) -> str:
    if length <= 0:
        return ""
    seed = seed or "S"
    repeats = (length + len(seed) - 1) // len(seed)
    return (seed * repeats)[:length]


def _resize_segment_sequence(seq: str, target_len: int, kind: str, policy: Dict[str, Any]) -> str:
    seed = _sanitize_aa_sequence(policy.get("seed_sequence") or policy.get("template_sequence") or "")
    if seed:
        seq = seed
    seq = _sanitize_aa_sequence(seq)
    if len(seq) == target_len:
        return seq
    if target_len <= 0:
        return ""

    if len(seq) > target_len:
        if kind == "cdr" and target_len >= 2:
            left = target_len // 2
            right = target_len - left
            return seq[:left] + seq[-right:]
        return seq[:target_len]

    insert = _repeat_to_length(_fill_residues(kind, policy), target_len - len(seq))
    if kind == "cdr" and len(seq) >= 2:
        mid = len(seq) // 2
        return seq[:mid] + insert + seq[mid:]
    return seq + insert


def _length_bounds(kind: str, current_len: int, policy: Dict[str, Any]) -> Tuple[int, int]:
    if isinstance(policy.get("length_range"), list) and len(policy["length_range"]) == 2:
        lo = _safe_int(policy["length_range"][0], current_len)
        hi = _safe_int(policy["length_range"][1], current_len)
    else:
        if kind == "cdr":
            lo, hi = 4, 24
        elif kind == "linker":
            lo, hi = 1, 20
        else:
            lo, hi = current_len, current_len
        lo = _safe_int(policy.get("min_length"), lo)
        hi = _safe_int(policy.get("max_length"), hi)
    lo = max(1, min(lo, hi))
    hi = max(lo, hi)
    return lo, hi


def _target_length(kind: str, current_len: int, policy: Dict[str, Any]) -> int:
    if "target_length" in policy:
        raw_target = _safe_int(policy.get("target_length"), current_len)
    elif "length" in policy:
        raw_target = _safe_int(policy.get("length"), current_len)
    elif "length_delta" in policy:
        raw_target = current_len + _safe_int(policy.get("length_delta"), 0)
    elif str(policy.get("length_bias", "")).lower() in {"extend", "longer", "expand"}:
        raw_target = current_len + 1
    elif str(policy.get("length_bias", "")).lower() in {"compact", "shorter", "shrink"}:
        raw_target = current_len - 1
    else:
        raw_target = current_len
    lo, hi = _length_bounds(kind, current_len, policy)
    return max(lo, min(hi, raw_target))


def _apply_layout_domain_order(state: Dict[str, Any], strategy: Dict[str, Any]) -> None:
    plan = _layout_plan(strategy)
    order = None
    for key in ("binder_domain_order", "domain_order", "binder_order"):
        if isinstance(plan.get(key), list):
            order = plan[key]
            break
        if isinstance(strategy.get(key), list):
            order = strategy[key]
            break
    if order is not None:
        state["binder"]["domain_order"] = order


def apply_strategy_tree_to_state(state: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    policies = strategy.get("node_edit_policies", {})
    updated = deepcopy(state)
    _apply_layout_domain_order(updated, strategy)

    if not isinstance(policies, dict) or not policies:
        return updated

    fixed_linker = set(updated["mutation_policy"].get("fixed_linker_segments", []))
    for group in state_domain_segment_keys(updated).values():
        for segment in updated["binder"].get(group, []):
            if len(segment) < 3:
                continue
            name, kind, seq = str(segment[0]), str(segment[1]), str(segment[2])
            policy = policies.get(name)
            if not isinstance(policy, dict):
                continue
            if not _as_bool(policy.get("enabled"), _as_bool(policy.get("mutable"), True)):
                continue
            if not _as_bool(policy.get("length_mutable"), False):
                continue
            if kind == "framework" and not _as_bool(policy.get("allow_framework_length_change"), False):
                continue
            if name in fixed_linker:
                continue

            target_len = _target_length(kind, len(seq), policy)
            if target_len != len(seq):
                segment[2] = _resize_segment_sequence(seq, target_len, kind, policy)

    return updated


def _make_group_node(name: str, children: List[List[str]]) -> Node:
    child_kinds = {str(segment[1]) for segment in children if len(segment) >= 2}
    group_kind = "linker" if child_kinds == {"linker"} or "linker" in name.lower() else "domain"
    return Node(
        kind=group_kind,
        name=name,
        children=[
            Node(kind=kind, name=segment_name, length=len(seq))
            for segment_name, kind, seq in children
        ],
    )


def make_binder_chain(state: Dict[str, Any]) -> Node:
    binder = state["binder"]
    children: List[Node] = []
    segment_keys = state_domain_segment_keys(state)
    for domain_name in binder_domain_order(state):
        key = segment_keys.get(domain_name)
        if key:
            children.append(_make_group_node(domain_name, binder[key]))

    return Node(
        kind="chain",
        name=binder.get("name", "Binder"),
        props={"chain_id": binder.get("chain_id", "BB")},
        children=children,
    )


def make_target_chain(state: Dict[str, Any]) -> Node:
    target = state["target"]
    return Node(
        kind="chain",
        name=target.get("name", "Target"),
        length=len(target["sequence"]),
        props={"chain_id": target.get("chain_id", "T")},
        children=[
            Node(
                kind=target.get("feature_kind", "epitope"),
                name=target.get("epitope_name", "target_feature"),
                residue_spans=[tuple(x) for x in target.get("epitope_spans", [])],
                props={
                    "source": target.get("epitope_source", ""),
                    "hotspot_motif": target.get("hotspot_motif", ""),
                },
            )
        ],
    )


def build_blueprint(state: Dict[str, Any]) -> Blueprint:
    return Blueprint(
        root=Node(
            kind="complex",
            name=state.get("task_name", "ASTevolve_Task"),
            children=[make_binder_chain(state), make_target_chain(state)],
        )
    )


def _mask_from_spans(length: int, allowed_spans: List[Tuple[int, int]]) -> List[bool]:
    mask = [False] * length
    for start, end in allowed_spans:
        for idx in range(max(0, start), min(length, end)):
            mask[idx] = True
    return mask


def _spans_from_constraint_entries(entries: Any, default_chain: str) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        chain_id = str(entry.get("chain_id") or entry.get("chain") or default_chain)
        raw_spans = entry.get("spans", entry.get("ranges", entry.get("residue_ranges", [])))
        spans: List[Tuple[int, int]] = []
        if isinstance(raw_spans, list):
            for raw in raw_spans:
                if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                    continue
                try:
                    start, end = int(raw[0]), int(raw[1])
                except (TypeError, ValueError):
                    continue
                spans.append((start, end))
        raw_residues = entry.get("residues", entry.get("indices", []))
        if isinstance(raw_residues, list):
            for raw in raw_residues:
                try:
                    idx = int(raw)
                except (TypeError, ValueError):
                    continue
                spans.append((idx, idx + 1))
        if spans:
            out.setdefault(chain_id, []).extend(spans)
    return out


def _constraint_spans_by_chain(
    state: Dict[str, Any],
    span_key: str,
    node_key: str,
) -> Dict[str, List[Tuple[int, int]]]:
    constraints = state.get("design_constraints", {})
    if not isinstance(constraints, dict):
        return {}
    binder_chain = state["binder"].get("chain_id", "BB")
    spans = segment_spans(flatten_binder_parts(state))
    out: Dict[str, List[Tuple[int, int]]] = {}
    for node_name in _name_list(constraints.get(node_key, [])):
        if node_name in spans:
            out.setdefault(binder_chain, []).append(spans[node_name])
    explicit = _spans_from_constraint_entries(constraints.get(span_key, []), binder_chain)
    for chain_id, chain_spans in explicit.items():
        out.setdefault(chain_id, []).extend(chain_spans)
    return out


def _apply_closed_spans(mask: List[bool], closed_spans: List[Tuple[int, int]]) -> None:
    for start, end in closed_spans:
        for idx in range(max(0, start), min(len(mask), end)):
            mask[idx] = False


def build_masks(state: Dict[str, Any], memory_bias: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, List[bool]]:
    parts = flatten_binder_parts(state)
    spans = segment_spans(parts)
    policy = state["mutation_policy"]
    always_open = set(policy.get("always_open_segments", []))
    conditionally_open = set(policy.get("conditionally_open_segments", []))
    edit_order = list(strategy.get("preferred_edit_order") or memory_bias.get("preferred_edit_order", []))
    design_points = state.get("design_points", {}) if isinstance(state.get("design_points"), dict) else {}
    for key in ("primary_design_nodes", "secondary_design_nodes", "default_open_nodes"):
        for name in _name_list(design_points.get(key, [])):
            if name not in edit_order:
                edit_order.append(name)
    tree_policy_active = bool(strategy.get("_tree_policy_active"))
    node_policies = strategy.get("node_edit_policies", {})

    selected: List[Tuple[int, int]] = []
    for name in edit_order:
        if tree_policy_active:
            node_policy = node_policies.get(name, {}) if isinstance(node_policies, dict) else {}
            if not _as_bool(node_policy.get("enabled"), _as_bool(node_policy.get("mutable"), False)):
                continue
        if name in always_open or name in conditionally_open:
            if name in spans and spans[name] not in selected:
                selected.append(spans[name])
        elif name in spans and name in set(_name_list(design_points.get("default_open_nodes", []))):
            selected.append(spans[name])
    if not tree_policy_active:
        for name in sorted(always_open):
            if name in spans and spans[name] not in selected:
                selected.append(spans[name])

    binder_chain = state["binder"].get("chain_id", "BB")
    target_chain = state["target"].get("chain_id", "T")
    for span in _constraint_spans_by_chain(state, "mutable_residue_spans", "mutable_nodes").get(binder_chain, []):
        if span not in selected:
            selected.append(span)
    binder_mask = _mask_from_spans(len(binder_sequence(state)), selected)
    frozen = _constraint_spans_by_chain(state, "frozen_residue_spans", "frozen_nodes")
    _apply_closed_spans(binder_mask, frozen.get(binder_chain, []))

    return {
        binder_chain: binder_mask,
        target_chain: [False] * len(state["target"]["sequence"]),
    }


def build_fixed_residues(state: Dict[str, Any], memory_bias: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    parts = flatten_binder_parts(state)
    spans = segment_spans(parts)
    fixed: Dict[str, Dict[int, str]] = {
        state["binder"].get("chain_id", "BB"): {},
        state["target"].get("chain_id", "T"): {},
    }

    linker_seq = str(memory_bias.get("linker_default_sequence", ""))
    for segment_name in state["mutation_policy"].get("fixed_linker_segments", []):
        if segment_name not in spans:
            continue
        start, end = spans[segment_name]
        offset = start - spans.get("Linker_head", (start, start))[0]
        for i, pos in enumerate(range(start, end)):
            if 0 <= offset + i < len(linker_seq):
                fixed[state["binder"].get("chain_id", "BB")][pos] = linker_seq[offset + i]

    binder_chain = state["binder"].get("chain_id", "BB")
    binder_seq = binder_sequence(state)
    for start, end in _constraint_spans_by_chain(state, "frozen_residue_spans", "frozen_nodes").get(binder_chain, []):
        for pos in range(max(0, start), min(len(binder_seq), end)):
            fixed[binder_chain][pos] = binder_seq[pos]

    for i, aa in enumerate(state["target"]["sequence"]):
        fixed[state["target"].get("chain_id", "T")][i] = aa

    return fixed


def _secondary_structure_constraint_specs(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    plan = _layout_plan(strategy)
    binder_chain = state["binder"].get("chain_id", "BB")
    segment_meta = _segment_metadata(state)
    target_map: Dict[str, str] = {}

    ss_priors = plan.get("secondary_structure_priors", {})
    if isinstance(ss_priors, dict):
        for node_name, ss_value in ss_priors.items():
            if node_name in segment_meta:
                code = _ss_code(ss_value)
                if code:
                    target_map[f"{binder_chain}:{node_name}"] = code

    regions = plan.get("design_regions", [])
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict):
                continue
            code = _ss_code(region.get("secondary_structure", region.get("ss", None)))
            if not code:
                continue
            for node_name in _region_targets(state, region, segment_meta):
                target_map[f"{binder_chain}:{node_name}"] = code

    if not target_map:
        return []
    return [
        {
            "kind": "ss_proxy",
            "weight": float(strategy.get("secondary_structure_weight", 0.35)),
            "params": {"target_map": target_map},
        }
    ]


def _site_anchor_constraint_specs(
    state: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    plan = _layout_plan(strategy)
    binder_chain = state["binder"].get("chain_id", "BB")
    segment_meta = _segment_metadata(state)
    specs: List[Dict[str, Any]] = []

    regions = plan.get("design_regions", [])
    if not isinstance(regions, list):
        return specs

    for region in regions:
        if not isinstance(region, dict):
            continue
        site_anchors = region.get("site_anchors", {})
        if not isinstance(site_anchors, dict):
            continue
        for node_name in _region_targets(state, region, segment_meta):
            anchor = site_anchors.get(node_name)
            if not isinstance(anchor, dict):
                continue
            favored = _canonical_residue_list(anchor.get("favored_residues", []))
            if not favored:
                continue
            specs.append(
                {
                    "kind": "segment_composition",
                    "weight": _clamp_float(anchor.get("constraint_weight", 0.25), 0.25, 0.0, 1.0),
                    "params": {
                        "aa_set": "".join(favored),
                        "min_frac": 0.05,
                        "max_frac": 0.90,
                        "segment_filter": {
                            "chain_id": binder_chain,
                            "name": node_name,
                        },
                    },
                }
            )
    return specs


def build_constraint_specs(
    state: Dict[str, Any],
    memory_bias: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    target_chain = state["target"].get("chain_id", "T")
    binder_chain = state["binder"].get("chain_id", "BB")
    task_type = str(state.get("task_type") or state.get("binder", {}).get("architecture", "")).lower()
    epitope_name = state["target"].get("epitope_name", "target_feature")
    no_cys = set("ADEFGHIKLMNPQRSTVWY")

    if "scfv" not in task_type and "antibody" not in task_type and "cd25" not in task_type:
        specs: List[Dict[str, Any]] = [
            {"kind": "alphabet", "weight": 1.0, "params": {"allowed": no_cys, "chain_ids": [binder_chain]}},
            {"kind": "fixed_chain_sequence", "weight": 1.0, "params": {"chain_id": target_chain, "sequence": state["target"]["sequence"]}},
            {"kind": "hydrophobic_pattern", "weight": 0.8, "params": {"domain_min_hydro": 0.20, "linker_max_hydro": 0.35}},
            {"kind": "max_run", "weight": 1.2, "params": {"aa_set": "AILMFWVY", "max_run": int(strategy.get("max_hydrophobic_run", 3)), "segment_filter": {"chain_id": binder_chain}}},
            {"kind": "max_run", "weight": 0.8, "params": {"aa_set": "KRDE", "max_run": int(strategy.get("max_charged_run", 3)), "segment_filter": {"chain_id": binder_chain}}},
            {"kind": "segment_composition", "weight": 0.7, "params": {"aa_set": "C", "min_frac": 0.0, "max_frac": 0.0, "segment_filter": {"chain_id": binder_chain}}},
            {"kind": "segment_composition", "weight": 0.7, "params": {"aa_set": "AILMFWVY", "min_frac": 0.05, "max_frac": float(strategy.get("pocket_hydrophobic_max", 0.55)), "segment_filter": {"chain_id": binder_chain, "kind": "pocket"}}},
            {"kind": "segment_composition", "weight": 0.6, "params": {"aa_set": "YHSTNQDEKR", "min_frac": 0.15, "max_frac": 0.90, "segment_filter": {"chain_id": binder_chain, "kind": "pocket"}}},
            {"kind": "segment_composition", "weight": 0.5, "params": {"aa_set": "GSTNQAP", "min_frac": 0.12, "max_frac": 0.85, "segment_filter": {"chain_id": binder_chain, "kind": "hinge"}}},
        ]
        specs.extend(_secondary_structure_constraint_specs(state, strategy))
        specs.extend(_site_anchor_constraint_specs(state, strategy))
        return specs

    cdr_favored = set(strategy.get("cdr_favored_residues") or memory_bias.get("cdr_favored_sparse", []))
    if not cdr_favored:
        cdr_favored = set("YWHNQSTRDE")

    linker_gs_min = float(strategy.get("linker_gs_min", memory_bias.get("linker_gs_min", 0.60)))
    linker_hydrophobic_max = float(strategy.get("linker_hydrophobic_max", memory_bias.get("linker_hydrophobic_max", 0.15)))
    linker_charged_max = float(strategy.get("linker_charged_max", memory_bias.get("linker_charged_max", 0.20)))
    desired_cdr3_hydro = float(strategy.get("desired_cdr3_hydro", 0.30))

    specs = [
        {"kind": "alphabet", "weight": 1.0, "params": {"allowed": no_cys, "chain_ids": [binder_chain]}},
        {"kind": "fixed_chain_sequence", "weight": 1.0, "params": {"chain_id": target_chain, "sequence": state["target"]["sequence"]}},
        {"kind": "hydrophobic_pattern", "weight": 1.2, "params": {"domain_min_hydro": 0.22, "linker_max_hydro": linker_hydrophobic_max}},
        {"kind": "segment_composition", "weight": 2.0, "params": {"aa_set": "GSAT", "min_frac": linker_gs_min, "max_frac": 1.0, "segment_filter": {"chain_id": binder_chain, "kind": "linker"}}},
        {"kind": "segment_composition", "weight": 1.3, "params": {"aa_set": "AILMFWVY", "min_frac": 0.0, "max_frac": linker_hydrophobic_max, "segment_filter": {"chain_id": binder_chain, "kind": "linker"}}},
        {"kind": "segment_composition", "weight": 1.0, "params": {"aa_set": "KRDE", "min_frac": 0.0, "max_frac": linker_charged_max, "segment_filter": {"chain_id": binder_chain, "kind": "linker"}}},
        {"kind": "max_run", "weight": 1.2, "params": {"aa_set": "AILMFWVY", "max_run": int(strategy.get("max_hydrophobic_run", 2)), "segment_filter": {"chain_id": binder_chain}}},
        {"kind": "max_run", "weight": 1.2, "params": {"aa_set": "KRDE", "max_run": int(strategy.get("max_charged_run", 2)), "segment_filter": {"chain_id": binder_chain}}},
        {"kind": "segment_composition", "weight": 1.2, "params": {"aa_set": "".join(sorted(cdr_favored)), "min_frac": 0.20, "max_frac": 0.85, "segment_filter": {"chain_id": binder_chain, "kind": "cdr"}}},
        {"kind": "segment_composition", "weight": 1.0, "params": {"aa_set": "AILMFWVY", "min_frac": 0.05, "max_frac": float(strategy.get("cdr_hydrophobic_max", 0.40)), "segment_filter": {"chain_id": binder_chain, "kind": "cdr"}}},
        {"kind": "segment_composition", "weight": 0.8, "params": {"aa_set": "KRDE", "min_frac": 0.05, "max_frac": float(strategy.get("cdr_charged_max", 0.45)), "segment_filter": {"chain_id": binder_chain, "kind": "cdr"}}},
        {"kind": "segment_composition", "weight": 1.0, "params": {"aa_set": "AILMFWVY", "min_frac": 0.10, "max_frac": 0.38, "segment_filter": {"chain_id": binder_chain, "kind": "framework"}}},
        {"kind": "segment_composition", "weight": 0.8, "params": {"aa_set": "KRDE", "min_frac": 0.05, "max_frac": 0.35, "segment_filter": {"chain_id": binder_chain, "kind": "framework"}}},
        {"kind": "interface_proxy", "weight": 0.8, "params": {"binder_chain": binder_chain, "binder_segment": "VH_CDR3", "target_chain": target_chain, "target_segment": epitope_name, "desired_binder_hydro": desired_cdr3_hydro}},
    ]
    specs.extend(_secondary_structure_constraint_specs(state, strategy))
    specs.extend(_site_anchor_constraint_specs(state, strategy))
    return specs


def build_sa_config(strategy: Dict[str, Any]) -> Dict[str, Any]:
    mcts_output_dir = Path(str(strategy.get("mcts_output_dir", "inner_loop")))
    if not mcts_output_dir.is_absolute():
        mcts_output_dir = PROJECT_ROOT / mcts_output_dir

    def resolve_optional_project_path(value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)

    external_kb_path = resolve_optional_project_path(strategy.get("external_kb_path"))
    external_kb_embedding_manifest_value = resolve_optional_project_path(strategy.get("external_kb_embedding_manifest"))
    mutation_ops = dict(strategy.get("mutation_ops", {
        "point": 0.42,
        "block": 0.12,
        "segment_resample": 0.08,
        "site_resample": 0.07,
        "segment_mutagenesis": 0.08,
        "motif_graft": 0.05,
        "region_shuffle": 0.04,
        "swap": 0.04,
    }))
    for op, weight in {
        "cdr_resample": 0.05,
        "pocket_motif_swap": 0.05,
        "linker_length_perturb": 0.02,
        "domain_length_perturb": 0.02,
    }.items():
        mutation_ops.setdefault(op, weight)

    return {
        "iterations": int(strategy.get("iterations", 1200)),
        "init_temp": float(strategy.get("init_temp", 2.0)),
        "cooling": float(strategy.get("cooling", 0.995)),
        "mutation_rate": float(strategy.get("mutation_rate", 0.06)),
        "resample_segment_prob": float(strategy.get("resample_segment_prob", 0.08)),
        "progen_weight": float(strategy.get("progen_weight", 1.0)),
        "progen_chains": ["BB"],
        "progen_reduce": "length_weighted",
        "sequence_prior_model": str(strategy.get("sequence_prior_model", "progen")),
        "chai1_enabled": bool(strategy.get("chai1_enabled", True)),
        "chai1_top_frac": float(strategy.get("chai1_top_frac", 0.01)),
        "chai1_min_candidates": int(strategy.get("chai1_min_candidates", 1)),
        "chai1_max_candidates": int(strategy.get("chai1_max_candidates", 3)),
        "chai1_num_trunk_recycles": 3,
        "chai1_num_diffn_timesteps": 50,
        "protenix_model_name": str(strategy.get("protenix_model_name", "protenix_mini_esm_v0.5.0")),
        "protenix_conda_env": resolve_protenix_conda_env(strategy.get("protenix_conda_env")),
        "protenix_seed": int(strategy.get("protenix_seed", 101)),
        "protenix_complex_use_msa": strategy.get("protenix_complex_use_msa"),
        "protenix_complex_cycle": strategy.get("protenix_complex_cycle"),
        "protenix_complex_step": strategy.get("protenix_complex_step"),
        "protenix_complex_sample": strategy.get("protenix_complex_sample"),
        "protenix_complex_use_default_params": strategy.get("protenix_complex_use_default_params"),
        "protenix_complex_timeout": strategy.get("protenix_complex_timeout"),
        "structure_model": str(strategy.get("structure_model", "protenix")),
        "structure_model_name": strategy.get("structure_model_name", strategy.get("esmfold2_model_name")),
        "esmfold2_mode": str(strategy.get("esmfold2_mode", "local")),
        "esmfold2_conda_env": strategy.get("esmfold2_conda_env"),
        "esmfold2_num_loops": int(strategy.get("esmfold2_num_loops", 3)),
        "esmfold2_num_sampling_steps": int(strategy.get("esmfold2_num_sampling_steps", 32)),
        "esmfold2_num_diffusion_samples": int(strategy.get("esmfold2_num_diffusion_samples", 1)),
        "multistate_objectives_enabled": bool(strategy.get("multistate_objectives_enabled", True)),
        "multistate_objective_weight": float(strategy.get("multistate_objective_weight", 1.0)),
        "mutation_ops": mutation_ops,
        "history_size": int(strategy.get("history_size", 50)),
        "search_method": str(strategy.get("search_method", "mcts")),
        "mcts_c_puct": float(strategy.get("mcts_c_puct", 1.4)),
        "mcts_max_depth": int(strategy.get("mcts_max_depth", 4)),
        "mcts_reward_scale": float(strategy.get("mcts_reward_scale", 1.0)),
        "mcts_output_dir": str(mcts_output_dir),
        "mcts_save_tree": bool(strategy.get("mcts_save_tree", True)),
        "mcts_save_variants": bool(strategy.get("mcts_save_variants", True)),
        "mcts_memory_enabled": bool(strategy.get("mcts_memory_enabled", True)),
        "external_kb_enabled": bool(strategy.get("external_kb_enabled", True)),
        "external_kb_path": external_kb_path,
        "external_kb_weight": float(strategy.get("external_kb_weight", 0.7)),
        "external_kb_embedding_manifest": external_kb_embedding_manifest_value,
        "external_kb_retrieval_enabled": bool(strategy.get("external_kb_retrieval_enabled", True)),
        "external_kb_retrieval_top_k": int(strategy.get("external_kb_retrieval_top_k", 20)),
        "external_kb_retrieval_weight": float(strategy.get("external_kb_retrieval_weight", 0.6)),
        "external_kb_device": str(strategy.get("external_kb_device", "auto")),
        "external_kb_max_length": int(strategy.get("external_kb_max_length", 128)),
        "node_edit_policies": dict(strategy.get("node_edit_policies", {})),
    }


def build_score_config(strategy: Dict[str, Any]) -> Dict[str, Any]:
    score = dict(strategy.get("score_config", {}))
    return {
        "weight_fast": float(score.get("weight_fast", 1.0)),
        "weight_plddt": float(score.get("weight_plddt", 5.0)),
        "weight_iptm": float(score.get("weight_iptm", 1.0)),
        "weight_ptm": float(score.get("weight_ptm", 0.5)),
        "weight_ranking_score": float(score.get("weight_ranking_score", 0.0)),
        "weight_interface_plddt": float(score.get("weight_interface_plddt", 1.0)),
        "weight_node_plddt_min": float(score.get("weight_node_plddt_min", 0.5)),
        "weight_clash": float(score.get("weight_clash", 1.0)),
        "weight_multistate": float(score.get("weight_multistate", 1.0)),
        "plddt_scale": float(score.get("plddt_scale", 100.0)),
        "clash_scale": float(score.get("clash_scale", 10.0)),
        "fast_loss_nonneg": bool(score.get("fast_loss_nonneg", True)),
    }


def build_case_inputs(
    strategy: Dict[str, Any],
    design_state_path: Optional[str] = None,
    memory_path: Optional[str] = None,
) -> Tuple[Blueprint, List[Dict[str, Any]], Dict[str, Any], Dict[str, List[bool]], Dict[str, str], Dict[str, Dict[int, str]], Dict[str, Any], Dict[str, Any]]:
    state = load_design_state(design_state_path)
    case_sheet = load_case_sheet(state, design_state_path)
    memory = load_memory_yaml(memory_path or state.get("memory_path"))
    memory_bias = extract_memory_bias(memory, state)
    strategy = sanitize_strategy_for_ast(state, strategy)
    strategy = normalize_strategy_tree(state, strategy, memory_bias)
    state = apply_strategy_tree_to_state(state, strategy)
    state["_case_sheet"] = case_sheet
    state["_layout_summary"] = strategy.get("layout_summary", {})
    state["_node_edit_policies"] = strategy.get("node_edit_policies", {})
    state["_strategy_schema_report"] = strategy.get("strategy_schema_report", {})

    bp = build_blueprint(state)
    masks = build_masks(state, memory_bias, strategy)
    templates = {
        state["binder"].get("chain_id", "BB"): binder_sequence(state),
        state["target"].get("chain_id", "T"): state["target"]["sequence"],
    }
    fixed_residues = build_fixed_residues(state, memory_bias)
    constraint_specs = build_constraint_specs(state, memory_bias, strategy)
    sa_config = build_sa_config(strategy)
    score_config = build_score_config(strategy)
    return bp, constraint_specs, sa_config, masks, templates, fixed_residues, score_config, state


def run_design_search(
    strategy: Dict[str, Any],
    seed: Optional[int] = None,
    design_state_path: Optional[str] = None,
    memory_path: Optional[str] = None,
) -> Dict[str, Any]:
    bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg, state = build_case_inputs(
        strategy,
        design_state_path=design_state_path,
        memory_path=memory_path,
    )
    compiled = bp.compile()
    compiled["_design_state"] = state
    masks_np = {k: np.array(v, dtype=bool) for k, v in masks.items()}
    cfg = SAConfig(**sa_cfg, seed=seed)
    memory = load_memory_yaml(memory_path or state.get("memory_path"))
    external_kb = load_external_knowledge_provider(
        cfg.external_kb_path if cfg.external_kb_enabled else None,
        embedding_manifest=cfg.external_kb_embedding_manifest
        if cfg.external_kb_enabled and cfg.external_kb_retrieval_enabled
        else None,
        retrieval_top_k=cfg.external_kb_retrieval_top_k,
        retrieval_weight=cfg.external_kb_retrieval_weight,
        device=cfg.external_kb_device,
        max_length=cfg.external_kb_max_length,
    )

    out = optimize_multichain(
        compiled,
        constraint_specs,
        cfg,
        masks=masks_np,
        template_seqs=templates,
        fixed_residues=fixed_residues,
        internal_memory=memory,
        external_kb=external_kb,
    )
    out["chain_lengths"] = compiled["chain_lengths"]
    out["segments"] = [
        {
            "chain_id": s.chain_id,
            "kind": s.kind,
            "name": s.name,
            "spans": s.spans,
            "total_length": s.total_length,
            "is_contiguous": s.is_contiguous,
            "start": s.start,
            "end": s.end,
            "props": s.props,
        }
        for s in compiled["segments"]
    ]
    out["blueprint_summary"] = {
        "task_name": state.get("task_name", "ASTevolve_Task"),
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "binder_architecture": state["binder"].get("architecture", "VH-Linker-VL"),
        "binder_domain_order": binder_domain_order(state),
        "target_name": state["target"].get("name", state["target"].get("chain_id", "target")),
        "epitope_name": state["target"].get("epitope_name"),
        "epitope_spans": state["target"].get("epitope_spans", []),
    }
    design_points = state.get("design_points", {}) if isinstance(state.get("design_points"), dict) else {}
    out["case_design_points"] = {
        "design_intent": design_points.get("design_intent"),
        "primary_design_nodes": design_points.get("primary_design_nodes", []),
        "secondary_design_nodes": design_points.get("secondary_design_nodes", []),
        "preserved_nodes": design_points.get("preserved_nodes", []),
        "operator_policy": design_points.get("operator_policy", {}),
        "known_gap": (
            (design_points.get("epitope_focus", {}) or {}).get("known_gap")
            if isinstance(design_points.get("epitope_focus"), dict)
            else (design_points.get("state_logic", {}) or {}).get("known_gap")
            if isinstance(design_points.get("state_logic"), dict)
            else None
        ),
        "case_information_needed": state.get("case_information_needed", []),
    }
    out["case_sheet_summary"] = compact_case_sheet(state.get("_case_sheet", {}))
    out["semantic_graph_summary"] = build_semantic_graph_summary(
        state,
        compiled,
        out.get("node_plddt") or (out.get("structure_metrics", {}) or {}).get("node_plddt"),
    )
    out["layout_summary"] = state.get("_layout_summary", {})
    out["strategy_schema_report"] = state.get("_strategy_schema_report", {})
    out["score_config"] = score_cfg
    out["design_state_version"] = state.get("version")
    if bool(strategy.get("memory_auto_update_enabled", True)) and cfg.mcts_memory_enabled:
        memory_update = update_internal_memory(
            resolve_memory_path(memory_path or state.get("memory_path")),
            out,
            max_recent_runs=int(strategy.get("memory_update_max_recent_runs", 10)),
            max_residues_per_node=int(strategy.get("memory_update_max_residues_per_node", 8)),
        )
        out["memory_update"] = {k: v for k, v in memory_update.items() if k != "memory"}
    else:
        out["memory_update"] = {
            "updated": False,
            "reason": "disabled by strategy or mcts_memory_enabled=False",
        }
    return out
