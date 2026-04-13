from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
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


def optimize_multichain(
    compiled: Dict[str, Any],
    constraint_specs: list[dict],
    cfg: SAConfig,
    masks: Dict[str, np.ndarray],
    template_seqs: Optional[Dict[str, str]] = None,
    fixed_residues: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    chain_lengths = compiled["chain_lengths"]

    terms_fast = build_terms_from_specs(constraint_specs, stage="fast")
    terms_chai = build_terms_from_specs(constraint_specs, stage="chai")

    cur = init_seqs(
        chain_lengths, rng, template_seqs=template_seqs, fixed_residues=fixed_residues
    )
    cur_break = energy_breakdown(cur, compiled, terms_fast)
    cur_progen = _progen_score(cur, cfg.progen_chains, cfg.progen_reduce)
    cur_fast = cur_break["total"] + cfg.progen_weight * (-cur_progen["loglik_avg"])

    best = cur
    best_break = cur_break
    best_progen = cur_progen
    best_fast = cur_fast

    T = float(cfg.init_temp)

    history: Dict[str, Any] = {"accepted_moves": [], "op_counts": {}}
    candidates: List[Dict[str, Any]] = []

    for _ in range(cfg.iterations):
        prop, move = mutate_seqs(cur, compiled, rng, cfg, masks=masks)

        prop_break = energy_breakdown(prop, compiled, terms_fast)
        prop_progen = _progen_score(prop, cfg.progen_chains, cfg.progen_reduce)
        prop_fast = prop_break["total"] + cfg.progen_weight * (-prop_progen["loglik_avg"])

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
            "seqs": prop,
            "fast_loss": float(prop_fast),
            "constraint_penalty": float(prop_break["total"]),
            "progen_loglik_avg": float(prop_progen["loglik_avg"]),
            "progen_loglik_sum": float(prop_progen["loglik_sum"]),
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

                plddt = run_protenix_plddt_multichain(
                    pred_name=chain_ids[0],
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
                compiled["_plddt"] = float(plddt)
                struct_pen = energy_breakdown(c["seqs"], compiled, terms_chai)["total"]
                compiled["_plddt"] = None

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
    }
    return out