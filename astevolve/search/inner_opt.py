from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

from astevolve.core.constraints import (
    AA,
    CHARGED,
    HYDROPHOBIC,
    ChaiPlddtDeltaTerm,
    build_terms_from_specs,
    energy_breakdown,
)
from astevolve.models.registry import (
    run_structure_confidence_complex,
    run_structure_confidence_multichain,
    sequence_loglikelihood,
)
from astevolve.evaluation.multistate_objectives import evaluate_multistate_objectives
from astevolve.metrics.structure import summarize_structure_metrics


@dataclass
class SAConfig:
    iterations: int = 1200
    init_temp: float = 2.0
    cooling: float = 0.995
    mutation_rate: float = 0.03
    resample_segment_prob: float = 0.05
    seed: Optional[int] = None

    progen_weight: float = 1.0
    progen_chains: Optional[List[str]] = None
    progen_reduce: str = "length_weighted"
    sequence_prior_model: str = "progen"

    # ---- 缂佹挻鐎０鍕ゴ瀵偓閸忕绱欑€涙顔岄崥宥勭箽閻?chai1_ 閸撳秶绱戞禒銉ュ悑鐎硅妫柊宥囩枂閿?----
    chai1_enabled: bool = True
    chai1_top_frac: float = 0.01
    chai1_min_candidates: int = 1
    chai1_max_candidates: int = 5

    # ---- 娴犮儰绗呴弮褍鐡у▓鍏哥箽閻ｆ瑤浜掗崗鐓庮啇閺冄囧帳缂?dict閿涘奔绲炬稉宥呭晙娴ｈ法鏁?----
    chai1_num_trunk_recycles: int = 3
    chai1_num_diffn_timesteps: int = 50
    chai1_use_esm_embeddings: bool = True

    # ---- protenix 娑撴挾鏁ら柊宥囩枂 ----
    protenix_model_name: str = "protenix_mini_esm_v0.5.0"
    protenix_conda_env: str = "pytorch"
    protenix_seed: int = 101
    protenix_complex_use_msa: Optional[bool] = None
    protenix_complex_cycle: Optional[int] = None
    protenix_complex_step: Optional[int] = None
    protenix_complex_sample: Optional[int] = None
    protenix_complex_use_default_params: Optional[bool] = None
    protenix_complex_timeout: Optional[int] = None
    structure_model: str = "protenix"
    structure_model_name: Optional[str] = None
    esmfold2_mode: str = "local"
    esmfold2_conda_env: Optional[str] = None
    esmfold2_num_loops: int = 3
    esmfold2_num_sampling_steps: int = 32
    esmfold2_num_diffusion_samples: int = 1
    multistate_objectives_enabled: bool = True
    multistate_objective_weight: float = 1.0

    mutation_ops: Dict[str, float] = field(default_factory=lambda: {
        "point": 0.42,
        "block": 0.12,
        "segment_resample": 0.08,
        "site_resample": 0.07,
        "segment_mutagenesis": 0.08,
        "motif_graft": 0.05,
        "cdr_resample": 0.05,
        "pocket_motif_swap": 0.05,
        "linker_length_perturb": 0.02,
        "domain_length_perturb": 0.02,
        "region_shuffle": 0.04,
        "swap": 0.04,
    })

    history_size: int = 50

    # ---- node-aware search ----
    search_method: str = "mcts"
    mcts_c_puct: float = 1.4
    mcts_max_depth: int = 4
    mcts_reward_scale: float = 1.0
    mcts_output_dir: str = "inner_loop"
    mcts_save_tree: bool = True
    mcts_save_variants: bool = True
    mcts_memory_enabled: bool = True
    external_kb_enabled: bool = True
    external_kb_path: Optional[str] = None
    external_kb_weight: float = 0.7
    external_kb_embedding_manifest: Optional[str] = None
    external_kb_retrieval_enabled: bool = True
    external_kb_retrieval_top_k: int = 20
    external_kb_retrieval_weight: float = 0.6
    external_kb_device: str = "auto"
    external_kb_max_length: int = 128
    node_edit_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _to_bool_mask(mask, L: int) -> np.ndarray:
    if mask is None:
        return np.ones(L, dtype=bool)
    if isinstance(mask, list):
        mask = np.array(mask, dtype=bool)
    if isinstance(mask, np.ndarray):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.size != L:
            raise ValueError(f"Mask length mismatch: {mask.size} vs {L}")
        return mask
    raise ValueError("mask must be list, np.ndarray, or None")


def random_chain_sequence(length: int, rng: np.random.Generator) -> str:
    idx = rng.integers(0, len(AA), size=length)
    return "".join(AA[i] for i in idx)


def init_seqs(
    chain_lengths: Dict[str, int],
    rng: np.random.Generator,
    template_seqs: Optional[Dict[str, str]] = None,
    fixed_residues: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    for cid, L in chain_lengths.items():
        base = list(random_chain_sequence(L, rng))

        if template_seqs and cid in template_seqs:
            t = template_seqs[cid]
            if len(t) != L:
                raise ValueError(
                    f"Template length mismatch for chain {cid}: {len(t)} vs {L}"
                )
            base = list(t)

        if fixed_residues and cid in fixed_residues:
            for pos, aa in fixed_residues[cid].items():
                if 0 <= pos < L:
                    base[pos] = aa

        seqs[cid] = "".join(base)
    return seqs


def _choose_op(rng: np.random.Generator, op_weights: Dict[str, float]) -> str:
    ops = list(op_weights.keys())
    w = np.array([op_weights[k] for k in ops], dtype=float)
    if w.sum() <= 0:
        return "point"
    w = w / w.sum()
    return ops[int(rng.choice(len(ops), p=w))]


def _mutate_point(
    seq_list: List[str], designable: np.ndarray, rng: np.random.Generator, k: int
) -> List[int]:
    if designable.size == 0:
        return []
    k = min(k, designable.size)
    pos = rng.choice(designable, size=k, replace=False)
    for i in pos:
        seq_list[i] = AA[int(rng.integers(0, len(AA)))]
    return list(map(int, pos))


def _mutate_block(
    seq_list: List[str], designable: np.ndarray, rng: np.random.Generator, block_len: int
) -> List[int]:
    if designable.size == 0:
        return []
    start = int(rng.choice(designable))
    positions = [
        i for i in range(start, min(len(seq_list), start + block_len))
        if i in set(designable)
    ]
    for i in positions:
        seq_list[i] = AA[int(rng.integers(0, len(AA)))]
    return positions


def _mutate_swap(
    seq_list: List[str], designable: np.ndarray, rng: np.random.Generator
) -> List[int]:
    if designable.size < 2:
        return []
    i, j = rng.choice(designable, size=2, replace=False)
    seq_list[i], seq_list[j] = seq_list[j], seq_list[i]
    return [int(i), int(j)]


def _mutate_region_shuffle(
    seq_list: List[str], designable: np.ndarray, rng: np.random.Generator, k: int
) -> List[int]:
    if designable.size < 2:
        return []
    k = min(max(2, k), designable.size)
    pos = list(map(int, rng.choice(designable, size=k, replace=False)))
    old = [seq_list[i] for i in pos]
    shuffled = old[:]
    rng.shuffle(shuffled)
    if shuffled == old and len(shuffled) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    for i, aa in zip(pos, shuffled):
        seq_list[i] = aa
    return pos


def mutate_seqs(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    new = {k: list(v) for k, v in seqs.items()}
    move: Dict[str, Any] = {"op": None, "positions": {}, "segments": []}

    op = _choose_op(rng, cfg.mutation_ops)
    move["op"] = op

    if op in {"segment_resample", "segment_mutagenesis", "motif_graft"} or (
        op == "block" and rng.random() < cfg.resample_segment_prob
    ):
        seg = rng.choice(compiled["segments"])
        cid = seg.chain_id
        mask = masks[cid]
        designable = [int(i) for i in seg.indices() if bool(mask[i])]
        if op in {"segment_mutagenesis", "motif_graft"}:
            k = min(len(designable), max(4, int(round(cfg.mutation_rate * len(designable) * 2))))
            selected = set(map(int, rng.choice(np.asarray(designable), size=k, replace=False))) if designable else set()
        else:
            selected = set(designable)
        positions: List[int] = []
        for i in designable:
            if i in selected:
                new[cid][i] = AA[int(rng.integers(0, len(AA)))]
                positions.append(i)
        move["segments"] = [(cid, seg.name, seg.spans)]
        move["positions"][cid] = positions
        return {k: "".join(v) for k, v in new.items()}, move

    for cid, s_list in new.items():
        L = len(s_list)
        designable = np.where(masks[cid])[0]
        if designable.size == 0:
            continue

        k = max(1, int(round(cfg.mutation_rate * designable.size)))
        if op == "point":
            pos = _mutate_point(s_list, designable, rng, k)
        elif op == "site_resample":
            pos = _mutate_point(s_list, designable, rng, max(k, min(4, designable.size)))
        elif op == "block":
            block_len = int(rng.integers(2, 6))
            pos = _mutate_block(s_list, designable, rng, block_len)
        elif op == "region_shuffle":
            pos = _mutate_region_shuffle(s_list, designable, rng, max(k, min(8, designable.size)))
        elif op == "swap":
            pos = _mutate_swap(s_list, designable, rng)
        else:
            pos = _mutate_point(s_list, designable, rng, k)

        move["positions"][cid] = pos

    return {k: "".join(v) for k, v in new.items()}, move


def _progen_score(
    seqs: Dict[str, str], chains: Optional[List[str]], reduce: str, model: str = "progen"
) -> Dict[str, float]:
    if chains is None:
        chains = list(seqs.keys())
    scores = []
    total_len = 0
    sum_loglik = 0.0
    for cid in chains:
        s = seqs.get(cid, "")
        if not s:
            continue
        out = sequence_loglikelihood(s, provider=model)
        loglik_sum = out["loglik_sum"]
        loglik_avg = out["loglik_avg"]
        L = len(s)
        scores.append((loglik_sum, loglik_avg, L))
        sum_loglik += loglik_sum
        total_len += L

    if not scores:
        return {"loglik_sum": 0.0, "loglik_avg": 0.0}

    if reduce == "mean":
        avg = float(np.mean([x[1] for x in scores]))
        return {"loglik_sum": float(sum_loglik), "loglik_avg": avg}
    else:
        avg = float(sum_loglik / max(1, total_len))
        return {"loglik_sum": float(sum_loglik), "loglik_avg": avg}


def compute_segment_scores(
    seqs: Dict[str, str], compiled: Dict[str, Any]
) -> List[Dict[str, Any]]:
    out = []
    for seg in compiled["segments"]:
        frag = seg.extract(seqs.get(seg.chain_id, ""))
        if not frag:
            continue
        L = len(frag)
        hydro = sum(1 for c in frag if c in HYDROPHOBIC) / L
        charged = sum(1 for c in frag if c in CHARGED) / L
        flexible = sum(1 for c in frag if c in set("GS")) / L
        polar = sum(1 for c in frag if c in set("STNQY")) / L
        out.append({
            "chain_id": seg.chain_id,
            "kind": seg.kind,
            "name": seg.name,
            "spans": seg.spans,
            "total_length": seg.total_length,
            "is_contiguous": seg.is_contiguous,
            "length": L,
            "hydro_frac": float(hydro),
            "charged_frac": float(charged),
            "flexible_frac": float(flexible),
            "polar_frac": float(polar),
        })
    return out


def _safe_stat(values: List[float], kind: str) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if kind == "min":
        return float(arr.min())
    if kind == "max":
        return float(arr.max())
    return float(arr.mean())


def compute_node_plddt(
    compiled: Dict[str, Any],
    residue_plddt: Optional[Dict[str, List[float]]],
) -> Dict[str, Dict[str, Any]]:
    if not residue_plddt:
        return {}

    name_counts: Dict[str, int] = {}
    for seg in compiled.get("segments", []):
        name_counts[seg.name] = name_counts.get(seg.name, 0) + 1

    out: Dict[str, Dict[str, Any]] = {}
    for seg in compiled.get("segments", []):
        chain_vals = residue_plddt.get(seg.chain_id)
        if not chain_vals:
            continue
        vals: List[float] = []
        for idx in seg.indices():
            if 0 <= int(idx) < len(chain_vals):
                vals.append(float(chain_vals[int(idx)]))
        if not vals:
            continue
        key = seg.name if name_counts.get(seg.name, 0) == 1 else f"{seg.chain_id}:{seg.name}"
        out[key] = {
            "chain_id": seg.chain_id,
            "kind": seg.kind,
            "name": seg.name,
            "spans": seg.spans,
            "residue_count": len(vals),
            "plddt_mean": _safe_stat(vals, "mean"),
            "plddt_min": _safe_stat(vals, "min"),
            "plddt_max": _safe_stat(vals, "max"),
        }
    return out


def _mean_indexed_plddt(copies: List[List[float]]) -> List[float]:
    if not copies:
        return []
    max_len = max((len(copy) for copy in copies), default=0)
    out: List[float] = []
    for idx in range(max_len):
        vals = [float(copy[idx]) for copy in copies if idx < len(copy)]
        if vals:
            out.append(float(sum(vals) / len(vals)))
    return out


def _residue_plddt_by_source_chain(
    confidence: Dict[str, Any],
    entity_units: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    residue_plddt = confidence.get("residue_plddt", {}) or {}
    if not residue_plddt or not entity_units:
        return {}

    by_source: Dict[str, List[List[float]]] = {}
    for unit in entity_units:
        source_chain = unit.get("source_chain")
        label = unit.get("label")
        if not source_chain or not label:
            continue
        vals = residue_plddt.get(str(label))
        if not vals:
            continue
        cleaned: List[float] = []
        for value in vals:
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                continue
        if cleaned:
            by_source.setdefault(str(source_chain), []).append(cleaned)

    return {
        source_chain: _mean_indexed_plddt(copies)
        for source_chain, copies in by_source.items()
        if copies
    }


def _resolve_complex_entities(
    raw_entities: List[Dict[str, Any]],
    seqs: Dict[str, str],
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        source_chain = item.pop("source_chain", None)
        if source_chain:
            seq = seqs.get(str(source_chain), "")
            if not seq:
                continue
            item["sequence"] = seq
        entities.append(item)
    return entities


def _asym_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"X{index + 1}"


def _infer_complex_entity_units(
    raw_entities: List[Dict[str, Any]],
    report_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    unit_index = 0
    for entity_index, report in enumerate(report_entities, start=1):
        if not isinstance(report, dict):
            continue
        raw = raw_entities[entity_index - 1] if entity_index - 1 < len(raw_entities) and isinstance(raw_entities[entity_index - 1], dict) else {}
        label = str(report.get("label") or raw.get("id") or raw.get("name") or f"entity{entity_index}")
        try:
            count = max(1, int(report.get("count", raw.get("count", 1))))
        except (TypeError, ValueError):
            count = 1
        for copy_index in range(1, count + 1):
            unit_label = label if count == 1 else f"{label}_{copy_index}"
            units.append(
                {
                    "entity_index": entity_index,
                    "label": unit_label,
                    "base_label": label,
                    "kind": report.get("kind"),
                    "copy_index": copy_index,
                    "asym_id": _asym_id(unit_index),
                    "source_chain": raw.get("source_chain"),
                }
            )
            unit_index += 1
    return units


def _mean_numeric(values: List[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_state_node_plddt(state_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    node_items: Dict[str, Dict[str, Any]] = {}
    means: List[float] = []
    mins: List[float] = []
    low_nodes: List[Dict[str, Any]] = []

    for state_result in state_results:
        state_name = str(state_result.get("name") or "state")
        summary = state_result.get("structure_metrics", {}) or {}
        for node_name, item in (summary.get("node_plddt", {}) or {}).items():
            if not isinstance(item, dict):
                continue
            key = f"{state_name}:{node_name}"
            record = dict(item)
            record["state"] = state_name
            node_items[key] = record

            mean_val = _float_or_none(record.get("plddt_mean"))
            min_val = _float_or_none(record.get("plddt_min"))
            if mean_val is not None:
                means.append(mean_val)
                if mean_val < 70.0:
                    low_nodes.append(
                        {
                            "state": state_name,
                            "node": str(node_name),
                            "plddt_mean": mean_val,
                            "plddt_min": min_val,
                        }
                    )
            if min_val is not None:
                mins.append(min_val)

    low_nodes = sorted(low_nodes, key=lambda x: float(x.get("plddt_mean", 0.0)))[:10]
    return node_items, {
        "node_count": len(node_items),
        "node_plddt_mean": _mean_numeric(means),
        "node_plddt_min": min(mins) if mins else None,
        "low_confidence_nodes": low_nodes,
    }


def _aggregate_complex_state_metrics(state_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalars: Dict[str, List[float]] = {}
    interface_contact_count = 0.0
    interface_residue_pair_count = 0.0
    clash_count = 0.0
    interface_plddts: List[float] = []

    for result in state_results:
        summary = result.get("structure_metrics", {}) or {}
        for key, value in (summary.get("scalar", {}) or {}).items():
            try:
                scalars.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
        interface = summary.get("interface", {}) or {}
        interface_contact_count += float(interface.get("total_contact_count") or 0.0)
        interface_residue_pair_count += float(interface.get("total_residue_pair_count") or 0.0)
        clash_count += float(interface.get("clash_count") or 0.0)
        if interface.get("interface_plddt_mean") is not None:
            try:
                interface_plddts.append(float(interface["interface_plddt_mean"]))
            except (TypeError, ValueError):
                pass

    node_plddt, node_summary = _aggregate_state_node_plddt(state_results)
    scalar_mean = {
        key: float(sum(vals) / len(vals))
        for key, vals in scalars.items()
        if vals
    }
    interface_summary = {
        "available": bool(state_results),
        "total_contact_count": interface_contact_count,
        "total_residue_pair_count": interface_residue_pair_count,
        "clash_count": clash_count,
        "interface_plddt_mean": _mean_numeric(interface_plddts),
        "pairs": {},
    }
    return {
        "scalar": scalar_mean,
        "chain_plddt": {},
        "node_plddt": node_plddt,
        "node_summary": node_summary,
        "interface": interface_summary,
        "dockq": {
            "available": False,
            "dockq": None,
            "reason": "true DockQ requires a native/reference complex",
        },
        "states": state_results,
        "cif_path": None,
    }


def _evaluate_complex_states(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    cfg: SAConfig,
) -> Tuple[Optional[float], Dict[str, Any], Dict[str, Dict[str, Any]], Optional[str], Optional[str]]:
    state = compiled.get("_design_state", {}) or {}
    complex_states = state.get("complex_states", [])
    if not isinstance(complex_states, list) or not complex_states:
        return None, {}, {}, None, None
    if str(cfg.structure_model).lower() != "protenix":
        return None, {}, {}, None, None

    state_results: List[Dict[str, Any]] = []
    first_out_dir: Optional[str] = None
    first_summary_json: Optional[str] = None
    confidence_metrics_by_state: Dict[str, Dict[str, Any]] = {}

    for index, raw_state in enumerate(complex_states, start=1):
        if not isinstance(raw_state, dict):
            continue
        name = str(raw_state.get("name") or f"state_{index}")
        raw_entities = list(raw_state.get("entities", []))
        entities = _resolve_complex_entities(raw_entities, seqs)
        if not entities:
            continue
        confidence = run_structure_confidence_complex(
            provider="protenix",
            pred_name=name,
            entities=entities,
            constraint=raw_state.get("constraint"),
            covalent_bonds=raw_state.get("covalent_bonds"),
            metric=str(raw_state.get("metric", "plddt")),
            seed=cfg.protenix_seed,
            model_name=cfg.structure_model_name or cfg.protenix_model_name,
            conda_env=cfg.protenix_conda_env,
            use_msa=cfg.protenix_complex_use_msa,
            cycle=cfg.protenix_complex_cycle,
            step=cfg.protenix_complex_step,
            sample=cfg.protenix_complex_sample,
            use_default_params=cfg.protenix_complex_use_default_params,
            timeout=cfg.protenix_complex_timeout,
        )
        entity_units = _infer_complex_entity_units(raw_entities, confidence.get("entities", []))
        source_residue_plddt = _residue_plddt_by_source_chain(confidence, entity_units)
        node_plddt = compute_node_plddt(compiled, source_residue_plddt)
        summary = summarize_structure_metrics(confidence, node_plddt=node_plddt)
        state_result = {
            "name": name,
            "role": raw_state.get("role"),
            "objective": raw_state.get("objective"),
            "entities": confidence.get("entities", []),
            "entity_units": entity_units,
            "polymer_units": confidence.get("polymer_units", []),
            "metric_units": confidence.get("metric_units", []),
            "confidence_metrics": dict(confidence.get("metrics", {}) or {}),
            "structure_metrics": summary,
            "input_json": confidence.get("input_json"),
            "summary_json": confidence.get("summary_json"),
            "out_dir": confidence.get("out_dir"),
        }
        state_results.append(state_result)
        confidence_metrics_by_state[name] = state_result["confidence_metrics"]
        first_out_dir = first_out_dir or confidence.get("out_dir")
        first_summary_json = first_summary_json or confidence.get("summary_json")

    if not state_results:
        return None, {}, {}, first_out_dir, first_summary_json

    aggregate = _aggregate_complex_state_metrics(state_results)
    objective_specs = state.get("multistate_objectives") or state.get("objectives") or []
    if cfg.multistate_objectives_enabled and objective_specs:
        aggregate["multistate_objectives"] = evaluate_multistate_objectives(
            aggregate,
            objective_specs,
            compiled=compiled,
            design_state=state,
        )
    plddt = aggregate.get("scalar", {}).get("plddt")
    return (
        float(plddt) if plddt is not None else 0.0,
        aggregate,
        confidence_metrics_by_state,
        first_out_dir,
        first_summary_json,
    )


def _install_confidence_cache(
    compiled: Dict[str, Any],
    seqs: Dict[str, str],
    chains: List[Tuple[str, str]],
    confidence: Dict[str, Any],
) -> None:
    struct_cache: Dict[Any, Any] = {}
    chain_ids = [cid for cid, _ in chains]
    struct_cache[("residue_plddt_signature", tuple(chain_ids))] = tuple(
        (cid, seqs.get(cid, "")) for cid in chain_ids
    )

    for metric_name, value in (confidence.get("metrics", {}) or {}).items():
        try:
            struct_cache[("scalar", str(metric_name), tuple(chain_ids))] = float(value)
        except (TypeError, ValueError):
            continue

    for cid, vals in (confidence.get("residue_plddt", {}) or {}).items():
        struct_cache[("residue_plddt", str(cid))] = [float(x) for x in vals]

    compiled["_struct_cache"] = struct_cache


def _extract_plddt_delta(
    seqs: Dict[str, str],
    compiled: Dict[str, Any],
    terms_chai: List[tuple[float, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    for _, t in terms_chai:
        if isinstance(t, ChaiPlddtDeltaTerm):
            try:
                mA = t._plddt_for(seqs, t.chains_A)
                mB = t._plddt_for(seqs, t.chains_B)
            except Exception:
                return None, None, None
            if mA is None or mB is None:
                return None, None, None
            # 缂佺喍绔存潻鏂挎礀 (A-B 閻?delta, A 閻ㄥ嫬鈧? B 閻ㄥ嫬鈧?
            return float(mA - mB), float(mA), float(mB)
    return None, None, None


def _seqs_hash(seqs: Dict[str, str]) -> str:
    payload = "|".join(f"{k}:{seqs[k]}" for k in sorted(seqs))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_yaml(path: Path, data: Any) -> None:
    try:
        import yaml  # type: ignore

        path.write_text(
            yaml.safe_dump(_jsonable(data), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        path.write_text(
            json.dumps(_jsonable(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _score_fast_candidate(
    seqs: Dict[str, str],
    terms_fast: List[tuple[float, Any]],
    cfg: SAConfig,
    compiled: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    breakdown = energy_breakdown(seqs, compiled, terms_fast)
    if float(cfg.progen_weight) <= 0.0:
        progen = {"loglik_sum": 0.0, "loglik_avg": 0.0}
    else:
        progen = _progen_score(
            seqs,
            cfg.progen_chains,
            cfg.progen_reduce,
            cfg.sequence_prior_model,
        )
    fast_loss = breakdown["total"] + cfg.progen_weight * (-progen["loglik_avg"])
    return breakdown, progen, float(fast_loss)


def _designable_segments(
    compiled: Dict[str, Any],
    masks: Dict[str, np.ndarray],
) -> List[Tuple[Any, List[int]]]:
    out: List[Tuple[Any, List[int]]] = []
    for seg in compiled["segments"]:
        mask = masks.get(seg.chain_id)
        if mask is None:
            continue
        positions = [
            int(i)
            for i in seg.indices()
            if 0 <= int(i) < len(mask) and bool(mask[int(i)])
        ]
        if positions:
            out.append((seg, positions))
    return out


def _memory_node_block(internal_memory: Optional[Dict[str, Any]], section: str, node_name: str) -> Dict[str, Any]:
    if not internal_memory:
        return {}
    block = internal_memory.get(section, {})
    if not isinstance(block, dict):
        return {}
    for key in (node_name, node_name.lower(), node_name.casefold()):
        val = block.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _window_priority_map(internal_memory: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not internal_memory:
        return {}
    windows = internal_memory.get("optimization_windows", {})
    editable = windows.get("editable_windows", {}) if isinstance(windows, dict) else {}
    priority = {}
    weights = {
        "highest_priority": 2.8,
        "medium_priority": 1.6,
        "low_priority": 0.8,
    }
    for group, weight in weights.items():
        for item in editable.get(group, []) or []:
            if isinstance(item, dict) and "node" in item:
                priority[str(item["node"])] = weight

    protected = windows.get("protected_windows", {}) if isinstance(windows, dict) else {}
    for name in protected.get("strongly_protected", []) or []:
        priority[str(name)] = min(priority.get(str(name), 1.0), 0.05)
    for name in protected.get("conditionally_protected", []) or []:
        priority[str(name)] = min(priority.get(str(name), 1.0), 0.35)
    return priority


def _node_policy(cfg: SAConfig, seg: Any) -> Dict[str, Any]:
    policies = cfg.node_edit_policies or {}
    if not isinstance(policies, dict):
        return {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        val = policies.get(key)
        if isinstance(val, dict):
            return val
    return {}


def _policy_float(policy: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(policy.get(key, default))
    except (TypeError, ValueError):
        return default


def _policy_int(policy: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(round(float(policy.get(key, default))))
    except (TypeError, ValueError):
        return default


def _segment_prior(
    seg: Any,
    internal_memory: Optional[Dict[str, Any]],
    external_kb: Optional[Any] = None,
    external_weight: float = 0.0,
    node_policy: Optional[Dict[str, Any]] = None,
) -> float:
    priority = _window_priority_map(internal_memory).get(seg.name, 1.0)
    if seg.kind == "cdr":
        priority *= 1.25
    elif seg.kind == "linker":
        priority *= 0.85
    elif seg.kind == "framework":
        priority *= 0.55

    adaptive = internal_memory.get("adaptive_memory", {}) if internal_memory else {}
    motif_memory = adaptive.get("motif_memory", {}) if isinstance(adaptive, dict) else {}
    motif = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(motif_memory.get(key), dict):
            motif = motif_memory[key]
            break
    if motif:
        confidence = float(motif.get("confidence") or 0.0)
        support = float(motif.get("support_count") or 0.0)
        priority *= 1.0 + min(1.0, confidence) + min(0.5, 0.03 * support)

    node_stats = adaptive.get("node_level_statistics", {}) if isinstance(adaptive, dict) else {}
    stats = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(node_stats.get(key), dict):
            stats = node_stats[key]
            break
    if stats:
        try:
            priority *= max(0.5, min(1.75, float(stats.get("priority_multiplier", 1.0))))
        except (TypeError, ValueError):
            pass

    external_prior = _external_prior_for_segment(external_kb, seg)
    priority *= _external_priority_multiplier(external_prior, external_weight)

    if isinstance(node_policy, dict) and node_policy:
        priority *= max(0.01, _policy_float(node_policy, "priority_boost", 1.0))

    return max(0.01, float(priority))


def _aa_class_members(class_name: str) -> str:
    table = {
        "aromatic": "YWHF",
        "polar_uncharged": "STNQY",
        "contextual_charge": "RKHDE",
        "flexible_small": "GSA",
        "positive": "RKH",
        "negative": "DE",
        "hydrophobic": "AILMFWVY",
        "charged": "RKHDE",
        "small": "GAS",
        "turn_loop": "GSPNDT",
    }
    return table.get(class_name, "")


def _external_prior_for_segment(
    external_kb: Optional[Any],
    seg: Any,
    sequence: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    if external_kb is None:
        return {}
    try:
        if sequence and hasattr(external_kb, "get_sequence_prior"):
            prior = external_kb.get_sequence_prior(
                node_name=seg.name,
                node_kind=seg.kind,
                chain_id=seg.chain_id,
                sequence=sequence,
                top_k=top_k,
            )
            return prior if isinstance(prior, dict) else {}
        if hasattr(external_kb, "get_node_prior"):
            prior = external_kb.get_node_prior(seg.name, seg.kind, seg.chain_id)
            return prior if isinstance(prior, dict) else {}
    except Exception:
        return {}
    return {}


def _external_priority_multiplier(prior: Dict[str, Any], weight: float) -> float:
    if not prior:
        return 1.0
    raw = prior.get("priority_boost", prior.get("priority_multiplier", 1.0))
    try:
        boost = float(raw)
    except (TypeError, ValueError):
        boost = 1.0
    try:
        confidence = float(prior.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    strength = max(0.0, float(weight)) * max(0.0, min(1.0, confidence))
    return max(0.05, 1.0 + (boost - 1.0) * strength)


def _apply_external_residue_prior(
    weights: np.ndarray,
    prior: Dict[str, Any],
    external_weight: float,
) -> np.ndarray:
    if not prior:
        return weights

    aa_index = {aa: i for i, aa in enumerate(AA)}
    try:
        confidence = float(prior.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    strength = max(0.0, float(external_weight)) * max(0.0, min(1.0, confidence))
    if strength <= 0:
        return weights

    for aa in prior.get("favored_residues", []) or []:
        aa = str(aa)
        if aa in aa_index:
            weights[aa_index[aa]] *= 1.0 + 1.8 * strength

    for class_name in prior.get("favored_residue_classes", []) or []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= 1.0 + 1.2 * strength

    for aa in prior.get("disfavored_residues", []) or []:
        aa = str(aa)
        if aa in aa_index:
            weights[aa_index[aa]] *= max(0.03, 1.0 - 0.85 * strength)

    for class_name in prior.get("disfavored_residue_classes", []) or []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= max(0.05, 1.0 - 0.65 * strength)

    aa_weights = prior.get("aa_weights", {})
    if isinstance(aa_weights, dict):
        raw = np.zeros(len(AA), dtype=float)
        for aa, value in aa_weights.items():
            aa = str(aa)
            if aa not in aa_index:
                continue
            try:
                raw[aa_index[aa]] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        if raw.sum() > 0:
            freq = raw / raw.sum()
            uniform = 1.0 / len(AA)
            multipliers = np.clip(freq / uniform, 0.20, 5.0)
            weights *= np.power(multipliers, 0.45 * strength)

    return weights


def _policy_residue_prior(node_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(node_policy, dict) or not node_policy:
        return {}
    prior: Dict[str, Any] = {
        "confidence": node_policy.get("confidence", node_policy.get("policy_weight", 1.0)),
        "favored_residues": node_policy.get("favored_residues", []),
        "favored_residue_classes": node_policy.get("favored_residue_classes", []),
        "disfavored_residues": node_policy.get("disfavored_residues", []),
        "disfavored_residue_classes": node_policy.get("disfavored_residue_classes", []),
        "aa_weights": node_policy.get("aa_weights", {}),
    }
    return prior


def _aa_weights_for_segment(
    seg: Any,
    internal_memory: Optional[Dict[str, Any]],
    external_kb: Optional[Any] = None,
    external_weight: float = 0.0,
    external_prior: Optional[Dict[str, Any]] = None,
    node_policy: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    weights = np.ones(len(AA), dtype=float)
    aa_index = {aa: i for i, aa in enumerate(AA)}

    if "C" in aa_index:
        weights[aa_index["C"]] *= 0.05

    if seg.kind == "linker":
        for aa in "GS":
            weights[aa_index[aa]] *= 6.0
        for aa in "AT":
            weights[aa_index[aa]] *= 1.8
        for aa in HYDROPHOBIC:
            weights[aa_index[aa]] *= 0.25
        for aa in CHARGED:
            weights[aa_index[aa]] *= 0.45
    elif seg.kind == "cdr":
        for aa in "YWHNQSTRDE":
            weights[aa_index[aa]] *= 2.0
        for aa in "ILMFV":
            weights[aa_index[aa]] *= 0.75
    elif seg.kind == "framework":
        for aa in "GSPNQ":
            weights[aa_index[aa]] *= 1.2
        for aa in "CWF":
            weights[aa_index[aa]] *= 0.35

    windows = internal_memory.get("optimization_windows", {}) if internal_memory else {}
    node_bias = windows.get("node_specific_bias", {}) if isinstance(windows, dict) else {}
    bias = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(node_bias.get(key), dict):
            bias = node_bias[key]
            break
    for class_name in bias.get("preferred_residue_classes", []) if bias else []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= 1.6

    adaptive = internal_memory.get("adaptive_memory", {}) if internal_memory else {}
    motif_memory = adaptive.get("motif_memory", {}) if isinstance(adaptive, dict) else {}
    motif = {}
    for key in (seg.name, seg.name.lower(), seg.name.casefold()):
        if isinstance(motif_memory.get(key), dict):
            motif = motif_memory[key]
            break
    for aa in motif.get("enriched_residues", []) if motif else []:
        if aa in aa_index:
            weights[aa_index[aa]] *= 2.5
    confidence = float(motif.get("confidence") or 0.0) if motif else 0.0
    class_boost = 1.0 + 1.1 * min(1.0, confidence)
    for class_name in motif.get("favored_classes", []) if motif else []:
        for aa in _aa_class_members(str(class_name)):
            if aa in aa_index:
                weights[aa_index[aa]] *= class_boost

    if external_prior is None:
        external_prior = _external_prior_for_segment(external_kb, seg)
    weights = _apply_external_residue_prior(weights, external_prior, external_weight)

    policy_prior = _policy_residue_prior(node_policy)
    if policy_prior:
        weights = _apply_external_residue_prior(
            weights,
            policy_prior,
            _policy_float(node_policy or {}, "policy_weight", 1.0),
        )

    weights = np.maximum(weights, 1e-6)
    return weights / weights.sum()


def _sample_aa(
    rng: np.random.Generator,
    weights: np.ndarray,
    old_aa: Optional[str] = None,
) -> str:
    for _ in range(6):
        aa = AA[int(rng.choice(len(AA), p=weights))]
        if old_aa is None or aa != old_aa:
            return aa
    return AA[int(rng.choice(len(AA), p=weights))]


def _sample_aa_from_pool(rng: np.random.Generator, pool: str, old_aa: Optional[str] = None) -> str:
    residues = [aa for aa in str(pool or "") if aa in AA]
    if not residues:
        residues = list(AA)
    for _ in range(6):
        aa = str(rng.choice(np.asarray(residues)))
        if old_aa is None or aa != old_aa:
            return aa
    return str(rng.choice(np.asarray(residues)))


def _node_default_motifs(seg: Any, node_policy: Optional[Dict[str, Any]]) -> List[str]:
    motifs = _policy_motifs(node_policy)
    if motifs:
        return motifs
    kind = str(getattr(seg, "kind", "") or "").lower()
    name = str(getattr(seg, "name", "") or "").lower()
    if kind == "cdr" or "cdr" in name:
        return ["YYG", "GYW", "RYY", "DYY", "NSY", "STY"]
    if "efhand" in name or "calcium" in name:
        return ["DGD", "DND", "EDE", "DAD", "NDE", "DSE"]
    if "pdz" in name or "groove" in name:
        return ["GYF", "HST", "KQY", "STV", "YGD"]
    if kind == "pocket" or "pocket" in name:
        return ["DY", "EY", "YH", "DEN", "STN", "NQY"]
    return ["GS", "ST", "NQ"]


def _relative_to_abs_position(seg: Any, raw_pos: Any) -> Optional[int]:
    try:
        pos = int(raw_pos)
    except (TypeError, ValueError):
        return None
    indices = [int(x) for x in seg.indices()]
    if 0 <= pos < len(indices):
        return indices[pos]
    if pos in indices:
        return pos
    return None


def _position_sampling_probs(
    seg: Any,
    positions: List[int],
    node_policy: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    """Convert AST site anchors into MCTS mutation-position probabilities."""
    if not positions or not isinstance(node_policy, dict) or not node_policy:
        return None

    pos_to_idx = {int(pos): idx for idx, pos in enumerate(positions)}
    weights = np.ones(len(positions), dtype=float)

    def boost_abs(abs_pos: Optional[int], factor: Any) -> None:
        if abs_pos is None or int(abs_pos) not in pos_to_idx:
            return
        try:
            value = float(factor)
        except (TypeError, ValueError):
            value = 1.0
        weights[pos_to_idx[int(abs_pos)]] *= max(0.05, value)

    raw_position_weights = node_policy.get("position_weights", {})
    if isinstance(raw_position_weights, dict):
        for raw_pos, raw_weight in raw_position_weights.items():
            boost_abs(_relative_to_abs_position(seg, raw_pos), raw_weight)

    hotspot_weight = _policy_float(node_policy, "hotspot_weight", 2.0)
    for raw_pos in node_policy.get("hotspot_positions", []) or []:
        boost_abs(_relative_to_abs_position(seg, raw_pos), hotspot_weight)

    raw_anchor = node_policy.get("site_anchors", {})
    if isinstance(raw_anchor, dict) and seg.name in raw_anchor and isinstance(raw_anchor[seg.name], dict):
        raw_anchor = raw_anchor[seg.name]

    if isinstance(raw_anchor, dict):
        anchor_weight = _policy_float(raw_anchor, "weight", 2.0)
        for raw_pos in raw_anchor.get("relative_positions", raw_anchor.get("positions", [])) or []:
            boost_abs(_relative_to_abs_position(seg, raw_pos), anchor_weight)

        seg_indices = [int(x) for x in seg.indices()]
        for raw_range in raw_anchor.get("relative_ranges", []) or []:
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                continue
            try:
                start = max(0, int(raw_range[0]))
                end = min(len(seg_indices), int(raw_range[1]))
            except (TypeError, ValueError):
                continue
            for rel_pos in range(start, max(start, end)):
                boost_abs(_relative_to_abs_position(seg, rel_pos), anchor_weight)

    weights = np.maximum(weights, 1e-8)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    probs = weights / total
    if np.allclose(probs, np.ones(len(probs)) / len(probs)):
        return None
    return probs


def _policy_abs_positions(seg: Any, node_policy: Optional[Dict[str, Any]], field: str) -> List[int]:
    if not isinstance(node_policy, dict):
        return []
    out: List[int] = []
    for raw_pos in node_policy.get(field, []) or []:
        pos = _relative_to_abs_position(seg, raw_pos)
        if pos is not None and int(pos) not in out:
            out.append(int(pos))
    return out


def _policy_anchor_positions(seg: Any, node_policy: Optional[Dict[str, Any]]) -> List[int]:
    if not isinstance(node_policy, dict):
        return []
    out = _policy_abs_positions(seg, node_policy, "anchor_positions")
    raw_anchor = node_policy.get("site_anchors", {})
    if isinstance(raw_anchor, dict) and seg.name in raw_anchor and isinstance(raw_anchor[seg.name], dict):
        raw_anchor = raw_anchor[seg.name]
    if isinstance(raw_anchor, dict):
        for raw_pos in raw_anchor.get("relative_positions", raw_anchor.get("positions", [])) or []:
            pos = _relative_to_abs_position(seg, raw_pos)
            if pos is not None and int(pos) not in out:
                out.append(int(pos))
    return out


def _policy_motifs(node_policy: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(node_policy, dict):
        return []
    raw: List[Any] = []
    for field in ("graft_motifs", "motif_candidates", "fill_residues"):
        value = node_policy.get(field)
        if isinstance(value, str):
            raw.append(value)
        elif isinstance(value, list):
            raw.extend(value)
    motifs: List[str] = []
    for item in raw:
        motif = "".join(ch for ch in str(item).upper() if ch in AA)
        if 2 <= len(motif) <= 48 and motif not in motifs:
            motifs.append(motif)
    return motifs


def _sample_positions(
    positions: List[int],
    rng: np.random.Generator,
    k: int,
    probs: Optional[np.ndarray] = None,
    preferred: Optional[List[int]] = None,
) -> List[int]:
    if not positions:
        return []
    preferred = [int(p) for p in (preferred or []) if int(p) in set(positions)]
    chosen: List[int] = []
    if preferred:
        rng.shuffle(preferred)
        chosen.extend(preferred[: min(len(preferred), max(1, k))])
    remaining = [int(p) for p in positions if int(p) not in set(chosen)]
    if len(chosen) < k and remaining:
        if probs is not None and len(probs) == len(positions):
            prob_by_pos = {int(pos): float(prob) for pos, prob in zip(positions, probs)}
            rem_probs = np.asarray([prob_by_pos[int(pos)] for pos in remaining], dtype=float)
            total = float(rem_probs.sum())
            rem_probs = rem_probs / total if total > 0 else None
        else:
            rem_probs = None
        extra = rng.choice(
            np.asarray(remaining),
            size=min(len(remaining), k - len(chosen)),
            replace=False,
            p=rem_probs,
        ).tolist()
        chosen.extend(int(x) for x in extra)
    return sorted(set(chosen))


def _graft_motif_into_node(
    seq_list: List[str],
    seg: Any,
    positions: List[int],
    motif: str,
    rng: np.random.Generator,
    node_policy: Optional[Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    pos_set = set(int(p) for p in positions)
    indices = [int(x) for x in seg.indices()]
    anchors = _policy_anchor_positions(seg, node_policy) + _policy_abs_positions(seg, node_policy, "hotspot_positions")
    candidate_starts = [int(p) for p in anchors if int(p) in pos_set]
    if not candidate_starts:
        candidate_starts = [
            start
            for start in positions
            if all((int(start) + offset) in pos_set for offset in range(len(motif)))
        ]
    if not candidate_starts and indices:
        max_start_idx = max(0, len(indices) - len(motif))
        candidate_starts = [
            indices[offset]
            for offset in range(0, max_start_idx + 1)
            if all(indices[offset + j] in pos_set for j in range(len(motif)))
        ]
    if not candidate_starts:
        return [], []

    start = int(rng.choice(np.asarray(candidate_starts)))
    changes: List[Dict[str, Any]] = []
    chosen: List[int] = []
    for offset, aa in enumerate(motif):
        pos = start + offset
        if pos not in pos_set or pos >= len(seq_list):
            continue
        old = seq_list[pos]
        if old == aa:
            continue
        seq_list[pos] = aa
        chosen.append(pos)
        changes.append({"position": int(pos), "from": old, "to": aa, "motif": motif})
    return chosen, changes


def _mutate_node_seqs(
    seqs: Dict[str, str],
    seg: Any,
    designable_positions: List[int],
    rng: np.random.Generator,
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    internal_memory: Optional[Dict[str, Any]],
    external_kb: Optional[Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    new = {k: list(v) for k, v in seqs.items()}
    cid = seg.chain_id
    positions = [
        i
        for i in designable_positions
        if cid in new and 0 <= i < len(new[cid]) and bool(masks[cid][i])
    ]
    move: Dict[str, Any] = {
        "op": None,
        "node": seg.name,
        "node_kind": seg.kind,
        "chain_id": cid,
        "positions": {cid: []},
        "segments": [(cid, seg.name, seg.spans)],
        "changes": [],
    }
    if not positions:
        return seqs, move

    node_policy = _node_policy(cfg, seg)
    protected = set(_policy_abs_positions(seg, node_policy, "protected_positions"))
    if protected:
        positions = [p for p in positions if int(p) not in protected]
    if not positions:
        return seqs, move
    op_weights = node_policy.get("mutation_ops", cfg.mutation_ops) if node_policy else cfg.mutation_ops
    if not isinstance(op_weights, dict):
        op_weights = cfg.mutation_ops
    op = _choose_op(rng, op_weights)
    move["op"] = op
    position_probs = _position_sampling_probs(seg, positions, node_policy)
    current_fragment = seg.extract(seqs.get(cid, ""))
    external_prior = _external_prior_for_segment(
        external_kb,
        seg,
        sequence=current_fragment if cfg.external_kb_retrieval_enabled else None,
        top_k=cfg.external_kb_retrieval_top_k,
    )
    weights = _aa_weights_for_segment(
        seg,
        internal_memory,
        external_kb=external_kb,
        external_weight=cfg.external_kb_weight if cfg.external_kb_enabled else 0.0,
        external_prior=external_prior,
        node_policy=node_policy,
    )
    if node_policy:
        move["node_policy"] = {
            "priority_boost": node_policy.get("priority_boost"),
            "mutation_rate": node_policy.get("mutation_rate"),
            "max_mutations_per_step": node_policy.get("max_mutations_per_step"),
            "edit_intent": node_policy.get("edit_intent"),
            "favored_residues": list(node_policy.get("favored_residues", []) or [])[:12],
            "favored_residue_classes": list(node_policy.get("favored_residue_classes", []) or [])[:8],
            "site_anchors": node_policy.get("site_anchors"),
            "anchor_positions": list(node_policy.get("anchor_positions", []) or [])[:16],
            "hotspot_positions": list(node_policy.get("hotspot_positions", []) or [])[:16],
            "graft_motifs": list(node_policy.get("graft_motifs", []) or [])[:6],
            "motif_candidates": list(node_policy.get("motif_candidates", []) or [])[:6],
            "operator_phase": node_policy.get("operator_phase"),
            "large_jump": node_policy.get("large_jump"),
            "secondary_structure": node_policy.get("secondary_structure"),
        }
    if external_prior:
        move["external_prior"] = {
            "source": external_prior.get("source"),
            "priority_boost": external_prior.get("priority_boost"),
            "confidence": external_prior.get("confidence"),
            "favored_residues": external_prior.get("favored_residues", [])[:12],
            "favored_residue_classes": external_prior.get("favored_residue_classes", [])[:8],
            "retrieval": external_prior.get("retrieval"),
        }

    mutation_rate = _policy_float(node_policy, "mutation_rate", cfg.mutation_rate)
    base_k = max(1, int(round(mutation_rate * len(positions))))
    max_step = _policy_int(node_policy, "max_mutations_per_step", 0)
    if max_step > 0:
        base_k = min(base_k, max_step)
    if op == "cdr_resample":
        if str(getattr(seg, "kind", "") or "").lower() == "cdr":
            chosen = list(positions)
        else:
            k = min(len(positions), max(base_k, min(8, len(positions))))
            chosen = _sample_positions(positions, rng, k, position_probs)
    elif op == "pocket_motif_swap":
        motif = str(rng.choice(np.asarray(_node_default_motifs(seg, node_policy))))
        chosen, motif_changes = _graft_motif_into_node(new[cid], seg, positions, motif, rng, node_policy)
        move["motif"] = motif
        move["changes"] = [
            {"chain_id": cid, "node": seg.name, **change}
            for change in motif_changes
        ]
        if chosen:
            move["positions"][cid] = [int(x) for x in chosen]
            return {k: "".join(v) for k, v in new.items()}, move
        preferred = _policy_abs_positions(seg, node_policy, "hotspot_positions") + _policy_anchor_positions(seg, node_policy)
        k = min(len(positions), max(base_k, min(5, len(positions))))
        chosen = _sample_positions(positions, rng, k, position_probs, preferred=preferred)
    elif op == "linker_length_perturb":
        move["virtual_length_delta"] = int(rng.choice(np.asarray([-3, -2, -1, 1, 2, 3])))
        k = len(positions) if str(getattr(seg, "kind", "") or "").lower() == "linker" else min(len(positions), max(base_k, min(6, len(positions))))
        chosen = _sample_positions(positions, rng, k, position_probs)
        for pos in chosen:
            old = new[cid][pos]
            aa = _sample_aa_from_pool(rng, "GSTAQPN", old_aa=old)
            new[cid][pos] = aa
            move["changes"].append({"chain_id": cid, "position": int(pos), "from": old, "to": aa, "node": seg.name})
        move["positions"][cid] = [int(x) for x in chosen]
        return {k: "".join(v) for k, v in new.items()}, move
    elif op == "domain_length_perturb":
        move["virtual_length_delta"] = int(rng.choice(np.asarray([-5, -3, -2, 2, 3, 5])))
        ordered = list(positions)
        edge_count = min(4, len(ordered))
        preferred = ordered[:edge_count] + ordered[-edge_count:]
        preferred += _policy_abs_positions(seg, node_policy, "hotspot_positions") + _policy_anchor_positions(seg, node_policy)
        k = min(len(positions), max(base_k, min(8, len(positions))))
        chosen = _sample_positions(positions, rng, k, position_probs, preferred=preferred)
    elif op == "segment_resample":
        k = min(len(positions), max(base_k, min(4, len(positions))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = _sample_positions(positions, rng, k, position_probs)
    elif op == "segment_mutagenesis":
        jump_floor = min(8, len(positions)) if bool(node_policy.get("large_jump")) else min(5, len(positions))
        k = min(len(positions), max(base_k, jump_floor))
        if max_step > 0:
            k = min(k, max_step)
        preferred = _policy_abs_positions(seg, node_policy, "hotspot_positions") + _policy_anchor_positions(seg, node_policy)
        chosen = _sample_positions(positions, rng, k, position_probs, preferred=preferred)
    elif op == "site_resample":
        preferred = (
            _policy_abs_positions(seg, node_policy, "hotspot_positions")
            + _policy_abs_positions(seg, node_policy, "anchor_positions")
            + _policy_abs_positions(seg, node_policy, "mutable_positions")
            + _policy_anchor_positions(seg, node_policy)
        )
        k = min(len(positions), max(base_k, min(4, len(positions))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = _sample_positions(positions, rng, k, position_probs, preferred=preferred)
    elif op == "motif_graft":
        motifs = _policy_motifs(node_policy)
        if motifs:
            motif = str(rng.choice(np.asarray(motifs)))
            chosen, motif_changes = _graft_motif_into_node(new[cid], seg, positions, motif, rng, node_policy)
            move["motif"] = motif
            move["changes"] = [
                {"chain_id": cid, "node": seg.name, **change}
                for change in motif_changes
            ]
            if chosen:
                move["positions"][cid] = [int(x) for x in chosen]
                return {k: "".join(v) for k, v in new.items()}, move
        k = min(len(positions), max(base_k, min(4, len(positions))))
        chosen = _sample_positions(positions, rng, k, position_probs)
    elif op == "region_shuffle":
        k = min(len(positions), max(2, max(base_k, min(8, len(positions)))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = _sample_positions(positions, rng, k, position_probs)
        old_residues = [new[cid][pos] for pos in chosen]
        shuffled = old_residues[:]
        rng.shuffle(shuffled)
        if shuffled == old_residues and len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]
        for pos, aa in zip(chosen, shuffled):
            old = new[cid][pos]
            new[cid][pos] = aa
            move["changes"].append(
                {"chain_id": cid, "position": int(pos), "from": old, "to": aa, "node": seg.name}
            )
        move["positions"][cid] = [int(x) for x in chosen]
        return {k: "".join(v) for k, v in new.items()}, move
    elif op == "block":
        start = int(rng.choice(np.asarray(positions), p=position_probs))
        block_upper = max(3, min(6, (max_step + 1) if max_step > 0 else 6))
        block_len = int(rng.integers(2, block_upper))
        pos_set = set(positions)
        chosen = [i for i in range(start, start + block_len) if i in pos_set]
        if not chosen:
            chosen = [start]
        if max_step > 0:
            chosen = chosen[:max_step]
    elif op == "swap" and len(positions) >= 2:
        i, j = rng.choice(np.asarray(positions), size=2, replace=False, p=position_probs)
        i, j = int(i), int(j)
        old_i, old_j = new[cid][i], new[cid][j]
        new[cid][i], new[cid][j] = old_j, old_i
        chosen = [i, j]
        move["changes"] = [
            {"chain_id": cid, "position": i, "from": old_i, "to": old_j, "node": seg.name},
            {"chain_id": cid, "position": j, "from": old_j, "to": old_i, "node": seg.name},
        ]
        move["positions"][cid] = chosen
        return {k: "".join(v) for k, v in new.items()}, move
    else:
        k = min(len(positions), base_k)
        chosen = sorted(rng.choice(np.asarray(positions), size=k, replace=False, p=position_probs).tolist())

    for pos in chosen:
        old = new[cid][pos]
        aa = _sample_aa(rng, weights, old_aa=old)
        new[cid][pos] = aa
        move["changes"].append(
            {"chain_id": cid, "position": int(pos), "from": old, "to": aa, "node": seg.name}
        )

    move["positions"][cid] = [int(x) for x in chosen]
    return {k: "".join(v) for k, v in new.items()}, move


def _mcts_child_score(parent: Dict[str, Any], child: Dict[str, Any], cfg: SAConfig) -> float:
    q = 0.0 if child["visits"] == 0 else child["total_reward"] / child["visits"]
    u = (
        float(cfg.mcts_c_puct)
        * float(child.get("prior", 1.0))
        * math.sqrt(max(1.0, float(parent["visits"])))
        / (1.0 + float(child["visits"]))
    )
    return float(q + u)


def _mcts_select_leaf(
    tree: Dict[str, Dict[str, Any]],
    root_id: str,
    cfg: SAConfig,
) -> str:
    node_id = root_id
    while tree[node_id]["children"] and tree[node_id]["depth"] < cfg.mcts_max_depth:
        parent = tree[node_id]
        node_id = max(
            parent["children"],
            key=lambda child_id: _mcts_child_score(parent, tree[child_id], cfg),
        )
    if tree[node_id]["depth"] >= cfg.mcts_max_depth:
        return root_id
    return node_id


def _mcts_backprop(
    tree: Dict[str, Dict[str, Any]],
    node_id: str,
    reward: float,
) -> None:
    cur: Optional[str] = node_id
    while cur is not None:
        node = tree[cur]
        node["visits"] += 1
        node["total_reward"] += float(reward)
        node["best_reward"] = max(float(node.get("best_reward", -1e9)), float(reward))
        cur = node.get("parent")


def _mcts_best_path(tree: Dict[str, Dict[str, Any]], node_id: str) -> List[str]:
    path = []
    cur: Optional[str] = node_id
    while cur is not None:
        path.append(cur)
        cur = tree[cur].get("parent")
    return list(reversed(path))


def _summarize_mcts_round(
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    best_node_id: str,
    root_fast: float,
) -> Dict[str, Any]:
    node_stats: Dict[str, Dict[str, Any]] = {}
    mutation_successes: List[Dict[str, Any]] = []

    for cand in candidates:
        move = cand.get("move", {})
        node_name = str(move.get("node", "unknown"))
        stats = node_stats.setdefault(
            node_name,
            {
                "evaluated": 0,
                "improved": 0,
                "best_fast_loss": None,
                "mean_fast_loss": 0.0,
                "mean_reward": 0.0,
                "top_changes": [],
            },
        )
        stats["evaluated"] += 1
        stats["mean_fast_loss"] += float(cand["fast_loss"])
        stats["mean_reward"] += float(cand.get("reward", 0.0))
        if cand["fast_loss"] < root_fast:
            stats["improved"] += 1
            for change in move.get("changes", [])[:4]:
                mutation_successes.append({
                    **change,
                    "fast_loss": float(cand["fast_loss"]),
                    "reward": float(cand.get("reward", 0.0)),
                })
        best_loss = stats["best_fast_loss"]
        if best_loss is None or cand["fast_loss"] < best_loss:
            stats["best_fast_loss"] = float(cand["fast_loss"])
            stats["top_changes"] = move.get("changes", [])[:8]

    for stats in node_stats.values():
        n = max(1, int(stats["evaluated"]))
        stats["mean_fast_loss"] = float(stats["mean_fast_loss"] / n)
        stats["mean_reward"] = float(stats["mean_reward"] / n)
        stats["success_rate"] = float(stats["improved"] / n)

    promoted = sorted(
        node_stats,
        key=lambda n: (
            node_stats[n]["success_rate"],
            -float(node_stats[n]["best_fast_loss"] or 0.0),
        ),
        reverse=True,
    )[:5]
    suppressed = sorted(
        node_stats,
        key=lambda n: (node_stats[n]["success_rate"], node_stats[n]["mean_reward"]),
    )[:5]

    mutation_successes = sorted(
        mutation_successes,
        key=lambda x: (-float(x["reward"]), float(x["fast_loss"])),
    )[:25]

    return {
        "created_at_unix": int(time.time()),
        "search_method": "mcts",
        "root_fast_loss": float(root_fast),
        "best_node_id": best_node_id,
        "best_path": _mcts_best_path(tree, best_node_id),
        "num_tree_nodes": len(tree),
        "num_evaluated_variants": len(candidates),
        "node_level_statistics": node_stats,
        "internal_memory_update_suggestion": {
            "promote_nodes": promoted,
            "suppress_nodes": suppressed,
            "effective_mutations": mutation_successes,
            "note": "Use this summary to update adaptive_memory; the full MCTS tree is short-term search state.",
        },
    }


def _write_inner_loop_artifacts(
    cfg: SAConfig,
    tree: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    round_summary: Dict[str, Any],
) -> Dict[str, str]:
    out_dir = Path(cfg.mcts_output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    if cfg.mcts_save_tree:
        tree_json = []
        for node in tree.values():
            item = {k: v for k, v in node.items() if k != "seqs"}
            if node.get("seqs") is not None:
                item["seq_hash"] = _seqs_hash(node["seqs"])
            tree_json.append(item)
        path = out_dir / "mcts_tree.json"
        _write_json(path, {"nodes": tree_json, "root": "root"})
        paths["mcts_tree"] = str(path)

    if cfg.mcts_save_variants:
        path = out_dir / "evaluated_variants.json"
        _write_json(path, candidates)
        paths["evaluated_variants"] = str(path)

    path = out_dir / "round_summary.yaml"
    _write_yaml(path, round_summary)
    paths["round_summary"] = str(path)
    return paths


def _run_mcts_search(
    compiled: Dict[str, Any],
    terms_fast: List[tuple[float, Any]],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    rng: np.random.Generator,
    template_seqs: Optional[Dict[str, str]],
    fixed_residues: Optional[Dict[str, Dict[int, str]]],
    internal_memory: Optional[Dict[str, Any]],
    external_kb: Optional[Any],
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float], float, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    chain_lengths = compiled["chain_lengths"]
    root_seqs = init_seqs(
        chain_lengths, rng, template_seqs=template_seqs, fixed_residues=fixed_residues
    )
    root_break, root_progen, root_fast = _score_fast_candidate(root_seqs, terms_fast, cfg, compiled)

    designable = _designable_segments(compiled, masks)
    if not designable:
        history = {"accepted_moves": [], "op_counts": {}, "node_visit_counts": {}}
        return root_seqs, root_break, root_progen, root_fast, history, [], {}

    raw_priors = np.array([
        _segment_prior(
            seg,
            internal_memory if cfg.mcts_memory_enabled else None,
            external_kb if cfg.external_kb_enabled else None,
            cfg.external_kb_weight,
            _node_policy(cfg, seg),
        )
        for seg, _ in designable
    ], dtype=float)
    raw_priors = np.maximum(raw_priors, 1e-6)
    norm_priors = raw_priors / raw_priors.sum()

    tree: Dict[str, Dict[str, Any]] = {
        "root": {
            "id": "root",
            "parent": None,
            "children": [],
            "depth": 0,
            "visits": 0,
            "total_reward": 0.0,
            "best_reward": -1e9,
            "prior": 1.0,
            "move": None,
            "seqs": root_seqs,
            "fast_loss": float(root_fast),
            "constraint_penalty": float(root_break["total"]),
            "progen_loglik_avg": float(root_progen["loglik_avg"]),
        }
    }
    candidates: List[Dict[str, Any]] = []
    history: Dict[str, Any] = {
        "accepted_moves": [],
        "op_counts": {},
        "node_visit_counts": {},
        "search_method": "mcts",
    }

    best = root_seqs
    best_break = root_break
    best_progen = root_progen
    best_fast = float(root_fast)
    best_node_id = "root"
    reward_scale = max(float(cfg.mcts_reward_scale), abs(float(root_fast)) * 0.05, 1.0)

    for step in range(cfg.iterations):
        parent_id = _mcts_select_leaf(tree, "root", cfg)
        parent = tree[parent_id]
        seg_idx = int(rng.choice(len(designable), p=norm_priors))
        seg, positions = designable[seg_idx]
        prop, move = _mutate_node_seqs(
            parent["seqs"],
            seg,
            positions,
            rng,
            cfg,
            masks,
            internal_memory if cfg.mcts_memory_enabled else None,
            external_kb if cfg.external_kb_enabled else None,
        )
        prop_break, prop_progen, prop_fast = _score_fast_candidate(prop, terms_fast, cfg, compiled)
        reward = float(np.tanh((float(root_fast) - float(prop_fast)) / reward_scale))

        node_id = f"n{step + 1}"
        child = {
            "id": node_id,
            "parent": parent_id,
            "children": [],
            "depth": int(parent["depth"]) + 1,
            "visits": 0,
            "total_reward": 0.0,
            "best_reward": -1e9,
            "prior": float(norm_priors[seg_idx]),
            "move": move,
            "seqs": prop,
            "fast_loss": float(prop_fast),
            "constraint_penalty": float(prop_break["total"]),
            "progen_loglik_avg": float(prop_progen["loglik_avg"]),
            "reward": reward,
        }
        tree[node_id] = child
        parent["children"].append(node_id)
        _mcts_backprop(tree, node_id, reward)

        history["op_counts"][move["op"]] = history["op_counts"].get(move["op"], 0) + 1
        history["node_visit_counts"][seg.name] = history["node_visit_counts"].get(seg.name, 0) + 1
        if cfg.history_size > 0:
            history["accepted_moves"].append(move)
            if len(history["accepted_moves"]) > cfg.history_size:
                history["accepted_moves"].pop(0)

        cand = {
            "variant_id": node_id,
            "parent_id": parent_id,
            "seq_hash": _seqs_hash(prop),
            "seqs": prop,
            "fast_loss": float(prop_fast),
            "constraint_penalty": float(prop_break["total"]),
            "progen_loglik_avg": float(prop_progen["loglik_avg"]),
            "progen_loglik_sum": float(prop_progen["loglik_sum"]),
            "reward": reward,
            "move": move,
            "mcts": {
                "depth": child["depth"],
                "prior": child["prior"],
                "path": _mcts_best_path(tree, node_id),
            },
        }
        candidates.append(cand)

        if prop_fast < best_fast:
            best, best_fast = prop, float(prop_fast)
            best_break, best_progen = prop_break, prop_progen
            best_node_id = node_id

    round_summary = _summarize_mcts_round(tree, candidates, best_node_id, float(root_fast))
    artifact_paths = _write_inner_loop_artifacts(cfg, tree, candidates, round_summary)
    search_artifacts = {
        "method": "mcts",
        "best_node_id": best_node_id,
        "best_path": _mcts_best_path(tree, best_node_id),
        "artifact_paths": artifact_paths,
        "round_summary": round_summary,
        "external_kb": external_kb.describe() if external_kb is not None and hasattr(external_kb, "describe") else None,
    }
    return best, best_break, best_progen, best_fast, history, candidates, search_artifacts


def optimize_multichain(
    compiled: Dict[str, Any],
    constraint_specs: list[dict],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]] = None,
    fixed_residues: Optional[Dict[str, Dict[int, str]]] = None,
    internal_memory: Optional[Dict[str, Any]] = None,
    external_kb: Optional[Any] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    chain_lengths = compiled["chain_lengths"]

    terms_fast = build_terms_from_specs(constraint_specs, stage="fast")
    terms_chai = build_terms_from_specs(constraint_specs, stage="chai")

    search_artifacts: Dict[str, Any] = {}
    if str(cfg.search_method).lower() == "mcts":
        best, best_break, best_progen, best_fast, history, candidates, search_artifacts = _run_mcts_search(
            compiled=compiled,
            terms_fast=terms_fast,
            cfg=cfg,
            masks=masks,
            rng=rng,
            template_seqs=template_seqs,
            fixed_residues=fixed_residues,
            internal_memory=internal_memory,
            external_kb=external_kb,
        )
    else:
        cur = init_seqs(
            chain_lengths, rng, template_seqs=template_seqs, fixed_residues=fixed_residues
        )
        cur_break, cur_progen, cur_fast = _score_fast_candidate(cur, terms_fast, cfg, compiled)

        best = cur
        best_break = cur_break
        best_progen = cur_progen
        best_fast = cur_fast

        T = float(cfg.init_temp)

        history = {"accepted_moves": [], "op_counts": {}, "search_method": "sa"}
        candidates = []

        for step in range(cfg.iterations):
            prop, move = mutate_seqs(cur, compiled, rng, cfg, masks=masks)

            prop_break, prop_progen, prop_fast = _score_fast_candidate(prop, terms_fast, cfg, compiled)

            accept = False
            if prop_fast <= cur_fast:
                accept = True
            else:
                if T > 1e-8 and rng.random() < float(np.exp((cur_fast - prop_fast) / T)):
                    accept = True

            if accept:
                cur, cur_fast = prop, prop_fast
                cur_break, cur_progen = prop_break, prop_progen

                history["op_counts"][move["op"]] = (
                    history["op_counts"].get(move["op"], 0) + 1
                )
                if cfg.history_size > 0:
                    history["accepted_moves"].append(move)
                    if len(history["accepted_moves"]) > cfg.history_size:
                        history["accepted_moves"].pop(0)

                if cur_fast < best_fast:
                    best, best_fast = cur, cur_fast
                    best_break, best_progen = cur_break, cur_progen

            candidates.append({
                "variant_id": f"sa_{step + 1}",
                "seq_hash": _seqs_hash(prop),
                "seqs": prop,
                "fast_loss": float(prop_fast),
                "constraint_penalty": float(prop_break["total"]),
                "progen_loglik_avg": float(prop_progen["loglik_avg"]),
                "progen_loglik_sum": float(prop_progen["loglik_sum"]),
                "move": move,
            })

            T *= float(cfg.cooling)

    # ---- 缂佹挻鐎０鍕ゴ闂冭埖顔岄敍鍧otenix 閺囧じ鍞?chai1閿?----
    chai_results: List[Dict[str, Any]] = []
    best_plddt: Optional[float] = None
    if cfg.chai1_enabled and candidates:
        candidates_sorted = sorted(candidates, key=lambda x: x["fast_loss"])
        k = max(
            cfg.chai1_min_candidates,
            int(round(cfg.chai1_top_frac * len(candidates_sorted))),
        )
        k = min(k, cfg.chai1_max_candidates, len(candidates_sorted))
        top = candidates_sorted[:k]

        for c in top:
            chains: List[Tuple[str, str]] = []
            confidence: Dict[str, Any] = {"metrics": {}, "chain_metrics": {}, "residue_plddt": {}}
            node_plddt: Dict[str, Dict[str, Any]] = {}
            chain_plddt: Dict[str, float] = {}
            structure_metrics: Dict[str, Any] = {}
            protenix_out_dir: Optional[str] = None
            protenix_summary_json: Optional[str] = None
            multistate_objectives: Dict[str, Any] = {}
            multistate_score = 0.0
            multistate_loss = 0.0
            try:
                complex_plddt, complex_summary, complex_confidence, complex_out_dir, complex_summary_json = _evaluate_complex_states(
                    c["seqs"],
                    compiled,
                    cfg,
                )
                if complex_summary:
                    plddt = float(complex_plddt or 0.0)
                    confidence = {
                        "metrics": dict(complex_summary.get("scalar", {}) or {}),
                        "chain_metrics": {},
                        "residue_plddt": {},
                    }
                    node_plddt = {}
                    chain_plddt = {}
                    structure_metrics = complex_summary
                    protenix_out_dir = complex_out_dir
                    protenix_summary_json = complex_summary_json
                    c_complex_confidence_metrics = complex_confidence
                    multistate_objectives = dict(complex_summary.get("multistate_objectives", {}) or {})
                    if multistate_objectives.get("enabled"):
                        multistate_score = float(multistate_objectives.get("normalized_score") or 0.0)
                        multistate_loss = float(cfg.multistate_objective_weight) * float(multistate_objectives.get("loss") or 0.0)
                    raise StopIteration

                def _get_chains_B(tc):
                    for _, t in tc:
                        if isinstance(t, ChaiPlddtDeltaTerm):
                            return list(t.chains_B)
                    return None

                chains_B = _get_chains_B(terms_chai)
                if chains_B:
                    chains = [(cid, c["seqs"][cid]) for cid in chains_B]
                else:
                    chains = [(cid, c["seqs"][cid]) for cid in compiled["chain_order"]]

                # 娴?terms_chai 娑擃厽顥呭ù瀣╁▏閻劎娈?metric
                def _get_metric(tc):
                    for _, t in tc:
                        if isinstance(t, ChaiPlddtDeltaTerm):
                            return getattr(t, 'metric', 'plddt')
                    return 'plddt'

                _metric = _get_metric(terms_chai)
                pred_name = "__".join(cid for cid, _ in chains) or "pred"

                structure_kwargs: Dict[str, Any] = {
                    "metric": _metric,
                    "seed": cfg.protenix_seed,
                    "model_name": cfg.structure_model_name or cfg.protenix_model_name,
                }
                if str(cfg.structure_model).lower() == "esmfold2":
                    structure_kwargs.update(
                        {
                            "mode": cfg.esmfold2_mode,
                            "conda_env": cfg.esmfold2_conda_env,
                            "num_loops": cfg.esmfold2_num_loops,
                            "num_sampling_steps": cfg.esmfold2_num_sampling_steps,
                            "num_diffusion_samples": cfg.esmfold2_num_diffusion_samples,
                        }
                    )
                elif str(cfg.structure_model).lower() == "protenix":
                    structure_kwargs.update(
                        {
                            "conda_env": cfg.protenix_conda_env,
                            "timeout": cfg.protenix_complex_timeout,
                            "use_msa": cfg.protenix_complex_use_msa,
                            "cycle": cfg.protenix_complex_cycle,
                            "step": cfg.protenix_complex_step,
                            "sample": cfg.protenix_complex_sample,
                            "use_default_params": cfg.protenix_complex_use_default_params,
                        }
                    )

                confidence = run_structure_confidence_multichain(
                    pred_name=pred_name,
                    chains=chains,
                    provider=cfg.structure_model,
                    **structure_kwargs,
                )
                plddt = float(confidence.get("metrics", {}).get(_metric, 0.0))
                chain_plddt = dict(confidence.get("chain_metrics", {}).get("plddt", {}) or {})
                node_plddt = compute_node_plddt(
                    compiled,
                    confidence.get("residue_plddt", {}) or {},
                )
                structure_metrics = summarize_structure_metrics(confidence, node_plddt=node_plddt)
                protenix_out_dir = confidence.get("out_dir")
                protenix_summary_json = confidence.get("summary_json")
                c_complex_confidence_metrics = {}
            except StopIteration:
                pass
            except Exception:
                plddt = 0.0
                structure_metrics = summarize_structure_metrics(confidence, node_plddt=node_plddt)
                c_complex_confidence_metrics = {}

            struct_pen = 0.0
            if terms_chai:
                compiled["_plddt"] = float(plddt)
                if chains:
                    _install_confidence_cache(compiled, c["seqs"], chains, confidence)
                else:
                    compiled["_struct_cache"] = {}
                struct_pen = energy_breakdown(c["seqs"], compiled, terms_chai)["total"]
                compiled["_plddt"] = None

            struct_pen += float(multistate_loss)

            plddt_delta, plddt_A, plddt_B = _extract_plddt_delta(
                c["seqs"], compiled, terms_chai
            )
            compiled["_struct_cache"] = {}

            c2 = dict(c)
            c2["plddt"] = float(plddt)
            c2["confidence_metrics"] = dict(confidence.get("metrics", {}) or {})
            if c_complex_confidence_metrics:
                c2["complex_state_confidence_metrics"] = c_complex_confidence_metrics
            c2["chain_plddt"] = chain_plddt
            c2["node_plddt"] = node_plddt
            c2["structure_metrics"] = structure_metrics
            c2["protenix_out_dir"] = protenix_out_dir
            c2["protenix_summary_json"] = protenix_summary_json
            c2["multistate_objectives"] = multistate_objectives
            c2["multistate_score"] = float(multistate_score)
            c2["multistate_loss"] = float(multistate_loss)
            c2["struct_penalty"] = float(struct_pen)
            c2["combined_loss"] = float(c["fast_loss"] + struct_pen)
            c2["plddt_delta"] = plddt_delta
            c2["plddt_A"] = plddt_A
            c2["plddt_B"] = plddt_B
            chai_results.append(c2)

            if best_plddt is None or plddt > best_plddt:
                best_plddt = float(plddt)

    final = best
    final_break = best_break
    final_progen = best_progen
    final_fast = best_fast
    final_plddt = best_plddt
    final_struct = 0.0
    final_combined = float(best_fast)
    final_delta = None
    final_plddt_A = None
    final_plddt_B = None
    final_chain_plddt: Dict[str, float] = {}
    final_node_plddt: Dict[str, Dict[str, Any]] = {}
    final_confidence_metrics: Dict[str, float] = {}
    final_structure_metrics: Dict[str, Any] = {}
    final_protenix_out_dir = None
    final_protenix_summary_json = None
    final_multistate_objectives: Dict[str, Any] = {}
    final_multistate_score = 0.0
    final_multistate_loss = 0.0

    if chai_results:
        chai_best = min(
            chai_results,
            key=lambda x: (
                x.get("combined_loss", x["fast_loss"]),
                -float(x.get("plddt", 0.0)),
            ),
        )
        final = chai_best["seqs"]
        final_break = {"total": chai_best["constraint_penalty"]}
        final_progen = {
            "loglik_avg": chai_best["progen_loglik_avg"],
            "loglik_sum": chai_best["progen_loglik_sum"],
        }
        final_fast = chai_best["fast_loss"]
        final_plddt = chai_best["plddt"]
        final_struct = chai_best.get("struct_penalty", 0.0)
        final_combined = chai_best.get("combined_loss", float(final_fast))
        final_delta = chai_best.get("plddt_delta", None)
        final_plddt_A = chai_best.get("plddt_A", None)
        final_plddt_B = chai_best.get("plddt_B", None)
        final_confidence_metrics = chai_best.get("confidence_metrics", {}) or {}
        final_chain_plddt = chai_best.get("chain_plddt", {}) or {}
        final_node_plddt = chai_best.get("node_plddt", {}) or {}
        final_structure_metrics = chai_best.get("structure_metrics", {}) or {}
        final_protenix_out_dir = chai_best.get("protenix_out_dir")
        final_protenix_summary_json = chai_best.get("protenix_summary_json")
        final_multistate_objectives = chai_best.get("multistate_objectives", {}) or {}
        final_multistate_score = float(chai_best.get("multistate_score", 0.0) or 0.0)
        final_multistate_loss = float(chai_best.get("multistate_loss", 0.0) or 0.0)

    if search_artifacts and chai_results:
        structure_eval_path = None
        try:
            out_dir = Path(cfg.mcts_output_dir)
            if not out_dir.is_absolute():
                out_dir = Path.cwd() / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            structure_eval_path = out_dir / "structure_evaluated_variants.json"
            _write_json(structure_eval_path, chai_results)
            search_artifacts.setdefault("artifact_paths", {})["structure_evaluated_variants"] = str(structure_eval_path)
        except Exception:
            structure_eval_path = None
        search_artifacts["structure_evaluation_summary"] = {
            "evaluated": len(chai_results),
            "best_metrics": final_confidence_metrics,
            "best_interface": (final_structure_metrics.get("interface") if final_structure_metrics else None),
            "best_node_summary": (final_structure_metrics.get("node_summary") if final_structure_metrics else None),
            "dockq": (final_structure_metrics.get("dockq") if final_structure_metrics else None),
            "dockq_proxy": (final_structure_metrics.get("dockq_proxy") if final_structure_metrics else None),
            "multistate_objectives": final_multistate_objectives,
            "artifact_path": str(structure_eval_path) if structure_eval_path else None,
        }

    out = {
        "seqs": final,
        "fast_loss": float(final_fast),
        "constraint_penalty": float(final_break["total"]),
        "progen_loglik_avg": float(final_progen["loglik_avg"]),
        "progen_loglik_sum": float(final_progen["loglik_sum"]),
        "chai_plddt": float(final_plddt) if final_plddt is not None else None,
        "chai_struct_penalty": float(final_struct),
        "chai_combined_loss": float(final_combined),
        "chai_evaluated": len(chai_results),
        "chai_results": chai_results[: min(10, len(chai_results))],
        "plddt_delta": final_delta,
        "plddt_A": final_plddt_A,
        "plddt_B": final_plddt_B,
        "confidence_metrics": final_confidence_metrics,
        "chain_plddt": final_chain_plddt,
        "node_plddt": final_node_plddt,
        "structure_metrics": final_structure_metrics,
        "protenix_out_dir": final_protenix_out_dir,
        "protenix_summary_json": final_protenix_summary_json,
        "multistate_objectives": final_multistate_objectives,
        "multistate_score": float(final_multistate_score),
        "multistate_loss": float(final_multistate_loss),
        "mutation_history": history,
        "segment_scores": compute_segment_scores(final, compiled),
        "search_method": str(cfg.search_method).lower(),
        "search_artifacts": search_artifacts,
    }
    return out
