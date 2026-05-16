from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from protein_lang import Blueprint, Node
from inner_opt import SAConfig, optimize_multichain
from external_kb import load_external_knowledge_provider

from .design_state import (
    PROJECT_ROOT,
    binder_sequence,
    flatten_binder_parts,
    load_design_state,
    segment_spans,
)

AA_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


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


def extract_memory_bias(memory: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    policy = state["mutation_policy"]
    out: Dict[str, Any] = {
        "preferred_edit_order": list(policy.get("preferred_edit_order", [])),
        "linker_default_sequence": "".join(p[2] for p in state["binder"]["linker_segments"]),
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
    if not policies:
        normalized.setdefault("node_edit_policies", {})
        return normalized

    normalized["_tree_policy_active"] = True
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
    else:
        raw_target = current_len
    lo, hi = _length_bounds(kind, current_len, policy)
    return max(lo, min(hi, raw_target))


def apply_strategy_tree_to_state(state: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    policies = strategy.get("node_edit_policies", {})
    if not isinstance(policies, dict) or not policies:
        return state

    updated = deepcopy(state)
    fixed_linker = set(updated["mutation_policy"].get("fixed_linker_segments", []))
    for group in ("vh_segments", "linker_segments", "vl_segments"):
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


def _make_domain_node(name: str, children: List[List[str]]) -> Node:
    return Node(
        kind="domain",
        name=name,
        children=[
            Node(kind=kind, name=segment_name, length=len(seq))
            for segment_name, kind, seq in children
        ],
    )


def make_binder_chain(state: Dict[str, Any]) -> Node:
    binder = state["binder"]
    return Node(
        kind="chain",
        name="Binder_scFv",
        props={"chain_id": binder.get("chain_id", "BB")},
        children=[
            _make_domain_node("VH_domain", binder["vh_segments"]),
            Node(
                kind="linker",
                name="Linker_module",
                children=[
                    Node(kind=kind, name=segment_name, length=len(seq))
                    for segment_name, kind, seq in binder["linker_segments"]
                ],
            ),
            _make_domain_node("VL_domain", binder["vl_segments"]),
        ],
    )


def make_target_chain(state: Dict[str, Any]) -> Node:
    target = state["target"]
    return Node(
        kind="chain",
        name="Target_CD25",
        length=len(target["sequence"]),
        props={"chain_id": target.get("chain_id", "T")},
        children=[
            Node(
                kind="epitope",
                name=target.get("epitope_name", "CD25_basiliximab_epitope"),
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
            name=state.get("task_name", "CD25_scFv_Design_Task"),
            children=[make_binder_chain(state), make_target_chain(state)],
        )
    )


def _mask_from_spans(length: int, allowed_spans: List[Tuple[int, int]]) -> List[bool]:
    mask = [False] * length
    for start, end in allowed_spans:
        for idx in range(max(0, start), min(length, end)):
            mask[idx] = True
    return mask


def build_masks(state: Dict[str, Any], memory_bias: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, List[bool]]:
    parts = flatten_binder_parts(state)
    spans = segment_spans(parts)
    policy = state["mutation_policy"]
    always_open = set(policy.get("always_open_segments", []))
    conditionally_open = set(policy.get("conditionally_open_segments", []))
    edit_order = list(strategy.get("preferred_edit_order") or memory_bias.get("preferred_edit_order", []))
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
    if not tree_policy_active:
        for name in sorted(always_open):
            if name in spans and spans[name] not in selected:
                selected.append(spans[name])

    return {
        state["binder"].get("chain_id", "BB"): _mask_from_spans(len(binder_sequence(state)), selected),
        state["target"].get("chain_id", "T"): [False] * len(state["target"]["sequence"]),
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

    for i, aa in enumerate(state["target"]["sequence"]):
        fixed[state["target"].get("chain_id", "T")][i] = aa

    return fixed


def build_constraint_specs(
    state: Dict[str, Any],
    memory_bias: Dict[str, Any],
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    target_chain = state["target"].get("chain_id", "T")
    binder_chain = state["binder"].get("chain_id", "BB")
    epitope_name = state["target"].get("epitope_name", "CD25_basiliximab_epitope")
    no_cys = set("ADEFGHIKLMNPQRSTVWY")

    cdr_favored = set(strategy.get("cdr_favored_residues") or memory_bias.get("cdr_favored_sparse", []))
    if not cdr_favored:
        cdr_favored = set("YWHNQSTRDE")

    linker_gs_min = float(strategy.get("linker_gs_min", memory_bias.get("linker_gs_min", 0.60)))
    linker_hydrophobic_max = float(strategy.get("linker_hydrophobic_max", memory_bias.get("linker_hydrophobic_max", 0.15)))
    linker_charged_max = float(strategy.get("linker_charged_max", memory_bias.get("linker_charged_max", 0.20)))
    desired_cdr3_hydro = float(strategy.get("desired_cdr3_hydro", 0.30))

    return [
        {"kind": "alphabet", "weight": 1.0, "params": {"allowed": no_cys}},
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


def build_sa_config(strategy: Dict[str, Any]) -> Dict[str, Any]:
    mcts_output_dir = Path(str(strategy.get("mcts_output_dir", "inner_loop")))
    if not mcts_output_dir.is_absolute():
        mcts_output_dir = PROJECT_ROOT / mcts_output_dir

    external_kb_path = Path(str(strategy.get("external_kb_path", "data/external_kb/external_prior_cache.json")))
    if not external_kb_path.is_absolute():
        external_kb_path = PROJECT_ROOT / external_kb_path

    raw_embedding_manifest = strategy.get("external_kb_embedding_manifest", "")
    if raw_embedding_manifest:
        external_kb_embedding_manifest = Path(str(raw_embedding_manifest))
        if not external_kb_embedding_manifest.is_absolute():
            external_kb_embedding_manifest = PROJECT_ROOT / external_kb_embedding_manifest
        external_kb_embedding_manifest_value = str(external_kb_embedding_manifest)
    else:
        external_kb_embedding_manifest_value = None

    return {
        "iterations": int(strategy.get("iterations", 1200)),
        "init_temp": float(strategy.get("init_temp", 2.0)),
        "cooling": float(strategy.get("cooling", 0.995)),
        "mutation_rate": float(strategy.get("mutation_rate", 0.06)),
        "resample_segment_prob": float(strategy.get("resample_segment_prob", 0.08)),
        "progen_weight": float(strategy.get("progen_weight", 1.0)),
        "progen_chains": ["BB"],
        "progen_reduce": "length_weighted",
        "chai1_enabled": bool(strategy.get("chai1_enabled", True)),
        "chai1_top_frac": float(strategy.get("chai1_top_frac", 0.01)),
        "chai1_min_candidates": int(strategy.get("chai1_min_candidates", 1)),
        "chai1_max_candidates": int(strategy.get("chai1_max_candidates", 3)),
        "chai1_num_trunk_recycles": 3,
        "chai1_num_diffn_timesteps": 50,
        "protenix_model_name": str(strategy.get("protenix_model_name", "protenix_mini_esm_v0.5.0")),
        "protenix_conda_env": str(strategy.get("protenix_conda_env", "pytorch")),
        "protenix_seed": int(strategy.get("protenix_seed", 101)),
        "mutation_ops": dict(strategy.get("mutation_ops", {
            "point": 0.75,
            "block": 0.15,
            "segment_resample": 0.07,
            "swap": 0.03,
        })),
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
        "external_kb_path": str(external_kb_path),
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
        "plddt_scale": float(score.get("plddt_scale", 100.0)),
        "fast_loss_nonneg": bool(score.get("fast_loss_nonneg", True)),
    }


def build_case_inputs(
    strategy: Dict[str, Any],
    design_state_path: Optional[str] = None,
    memory_path: Optional[str] = None,
) -> Tuple[Blueprint, List[Dict[str, Any]], Dict[str, Any], Dict[str, List[bool]], Dict[str, str], Dict[str, Dict[int, str]], Dict[str, Any], Dict[str, Any]]:
    state = load_design_state(design_state_path)
    memory = load_memory_yaml(memory_path or state.get("memory_path"))
    memory_bias = extract_memory_bias(memory, state)
    strategy = normalize_strategy_tree(state, strategy, memory_bias)
    state = apply_strategy_tree_to_state(state, strategy)

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
        "task_name": state.get("task_name", "CD25_scFv_Design_Task"),
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "binder_architecture": state["binder"].get("architecture", "VH-Linker-VL"),
        "target_name": state["target"].get("name", "CD25"),
        "epitope_name": state["target"].get("epitope_name"),
        "epitope_spans": state["target"].get("epitope_spans", []),
    }
    out["score_config"] = score_cfg
    out["design_state_version"] = state.get("version")
    return out
