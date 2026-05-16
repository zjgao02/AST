from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

from constraints import (
    AA, HYDROPHOBIC, CHARGED,
    energy_breakdown, build_terms_from_specs,
    ChaiPlddtDeltaTerm,          # 实际是 ProtenixPlddtDeltaTerm 的别名
)
from progen_api import sequence_loglikelihood
from protenix_api import run_protenix_plddt_multichain


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

    # ---- 结构预测开关（字段名保留 chai1_ 前缀以兼容旧配置） ----
    chai1_enabled: bool = True
    chai1_top_frac: float = 0.01
    chai1_min_candidates: int = 1
    chai1_max_candidates: int = 5

    # ---- 以下旧字段保留以兼容旧配置 dict，但不再使用 ----
    chai1_num_trunk_recycles: int = 3
    chai1_num_diffn_timesteps: int = 50
    chai1_use_esm_embeddings: bool = True

    # ---- protenix 专用配置 ----
    protenix_model_name: str = "protenix_mini_esm_v0.5.0"
    protenix_conda_env: str = "protenix_mini"
    protenix_seed: int = 101

    mutation_ops: Dict[str, float] = field(default_factory=lambda: {
        "point": 0.6,
        "block": 0.2,
        "segment_resample": 0.15,
        "swap": 0.05,
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

    if op == "segment_resample" or (
        op == "block" and rng.random() < cfg.resample_segment_prob
    ):
        seg = rng.choice(compiled["segments"])
        cid = seg.chain_id
        mask = masks[cid]
        positions: List[int] = []
        for i in seg.indices():
            if mask[i]:
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
        elif op == "block":
            block_len = int(rng.integers(2, 6))
            pos = _mutate_block(s_list, designable, rng, block_len)
        elif op == "swap":
            pos = _mutate_swap(s_list, designable, rng)
        else:
            pos = _mutate_point(s_list, designable, rng, k)

        move["positions"][cid] = pos

    return {k: "".join(v) for k, v in new.items()}, move


def _progen_score(
    seqs: Dict[str, str], chains: Optional[List[str]], reduce: str
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
        out = sequence_loglikelihood(s)
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
            # 统一返回 (A-B 的 delta, A 的值, B 的值)
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
    progen = _progen_score(seqs, cfg.progen_chains, cfg.progen_reduce)
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
    op_weights = node_policy.get("mutation_ops", cfg.mutation_ops) if node_policy else cfg.mutation_ops
    if not isinstance(op_weights, dict):
        op_weights = cfg.mutation_ops
    op = _choose_op(rng, op_weights)
    move["op"] = op
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
    if op == "segment_resample":
        k = min(len(positions), max(base_k, min(4, len(positions))))
        if max_step > 0:
            k = min(k, max_step)
        chosen = sorted(rng.choice(positions, size=k, replace=False).tolist())
    elif op == "block":
        start = int(rng.choice(positions))
        block_upper = max(3, min(6, (max_step + 1) if max_step > 0 else 6))
        block_len = int(rng.integers(2, block_upper))
        pos_set = set(positions)
        chosen = [i for i in range(start, start + block_len) if i in pos_set]
        if not chosen:
            chosen = [start]
        if max_step > 0:
            chosen = chosen[:max_step]
    elif op == "swap" and len(positions) >= 2:
        i, j = rng.choice(positions, size=2, replace=False)
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
        chosen = sorted(rng.choice(positions, size=k, replace=False).tolist())

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

    # ---- 结构预测阶段（protenix 替代 chai1） ----
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
            try:
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

                # 从 terms_chai 中检测使用的 metric
                def _get_metric(tc):
                    for _, t in tc:
                        if isinstance(t, ChaiPlddtDeltaTerm):
                            return getattr(t, 'metric', 'plddt')
                    return 'plddt'

                _metric = _get_metric(terms_chai)
                pred_name = "__".join(cid for cid, _ in chains) or "pred"

                plddt = run_protenix_plddt_multichain(
                    pred_name=pred_name,
                    chains=chains,
                    metric=_metric,                    # ← 新增
                    seed=cfg.protenix_seed,
                    model_name=cfg.protenix_model_name,
                    conda_env=cfg.protenix_conda_env,
                )
            except Exception:
                plddt = 0.0

            struct_pen = 0.0
            if terms_chai:
                compiled["_struct_cache"] = {}
                compiled["_plddt"] = float(plddt)
                struct_pen = energy_breakdown(c["seqs"], compiled, terms_chai)["total"]
                compiled["_plddt"] = None
                compiled["_struct_cache"] = {}

            plddt_delta, plddt_A, plddt_B = _extract_plddt_delta(
                c["seqs"], compiled, terms_chai
            )

            c2 = dict(c)
            c2["plddt"] = float(plddt)
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
        "mutation_history": history,
        "segment_scores": compute_segment_scores(final, compiled),
        "search_method": str(cfg.search_method).lower(),
        "search_artifacts": search_artifacts,
    }
    return out
