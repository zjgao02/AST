from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import math


AA_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def _safe_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _round(value: Any, ndigits: int = 4) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return round(out, ndigits)


def _aa_classes(aa: str) -> List[str]:
    aa = str(aa).upper()
    classes: List[str] = []
    if aa in "YWHF":
        classes.append("aromatic")
    if aa in "STNQY":
        classes.append("polar_uncharged")
    if aa in "RKHDE":
        classes.append("contextual_charge")
    if aa in "GSPNDT":
        classes.append("turn_loop")
    if aa in "GSA":
        classes.append("flexible_small")
    return classes


def _node_key(block: Dict[str, Any], node: str) -> str:
    for key in (node, node.lower(), node.casefold()):
        if key in block:
            return str(key)
    return str(node)


def _top_counter_dict(counter: Counter, limit: int) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(max(1, int(limit))) if int(v) > 0}


def _counter_from_dict(data: Any) -> Counter:
    counter: Counter = Counter()
    if isinstance(data, dict):
        for key, value in data.items():
            count = _safe_int(value, 0)
            if count > 0:
                counter[str(key)] += count
    return counter


def _bounded_mutations(mutations: Any, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(mutations, list):
        return out
    for item in mutations:
        if not isinstance(item, dict):
            continue
        aa_to = str(item.get("to", "")).upper()
        if aa_to not in AA_CANONICAL:
            continue
        compact = {
            "node": str(item.get("node", "")),
            "chain_id": str(item.get("chain_id", "")),
            "position": _safe_int(item.get("position"), 0),
            "from": str(item.get("from", ""))[:1],
            "to": aa_to,
        }
        reward = _round(item.get("reward"), 6)
        if reward is not None:
            compact["reward"] = reward
        fast_loss = _round(item.get("fast_loss"), 6)
        if fast_loss is not None:
            compact["fast_loss"] = fast_loss
        out.append(compact)
        if len(out) >= limit:
            break
    return out


def _seq_hash(seqs: Any) -> Optional[str]:
    if not isinstance(seqs, dict) or not seqs:
        return None
    payload = "|".join(f"{k}:{seqs[k]}" for k in sorted(seqs))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _blend(old: Any, new: float, alpha: float = 0.25) -> float:
    old_val = _safe_float(old, new)
    return float((1.0 - alpha) * old_val + alpha * new)


def update_internal_memory(
    memory_path: Path,
    search_output: Dict[str, Any],
    *,
    max_recent_runs: int = 10,
    max_residues_per_node: int = 8,
    dry_run: bool = False,
) -> Dict[str, Any]:
    yaml = _safe_import_yaml()
    if yaml is None:
        return {"updated": False, "reason": "PyYAML is not available"}

    memory_path = Path(memory_path)
    if not memory_path.exists():
        return {"updated": False, "reason": f"memory file not found: {memory_path}"}

    try:
        memory = yaml.safe_load(memory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"updated": False, "reason": f"failed to read memory: {exc}"}
    if not isinstance(memory, dict):
        return {"updated": False, "reason": "memory YAML did not contain a mapping"}

    timestamp = _now_iso()
    search_artifacts = search_output.get("search_artifacts", {}) or {}
    round_summary = search_artifacts.get("round_summary", {}) or {}
    suggestion = round_summary.get("internal_memory_update_suggestion", {}) or {}
    node_stats = round_summary.get("node_level_statistics", {}) or {}
    effective_mutations = _bounded_mutations(
        suggestion.get("effective_mutations", []),
        max(4, max_residues_per_node * 2),
    )

    metadata = memory.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["last_auto_update"] = timestamp

    adaptive = memory.setdefault("adaptive_memory", {})
    if not isinstance(adaptive, dict):
        adaptive = {}
        memory["adaptive_memory"] = adaptive
    motif_memory = adaptive.setdefault("motif_memory", {})
    if not isinstance(motif_memory, dict):
        motif_memory = {}
        adaptive["motif_memory"] = motif_memory
    node_level = adaptive.setdefault("node_level_statistics", {})
    if not isinstance(node_level, dict):
        node_level = {}
        adaptive["node_level_statistics"] = node_level

    mutations_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for change in effective_mutations:
        node = str(change.get("node", ""))
        if node:
            mutations_by_node.setdefault(node, []).append(change)

    touched_nodes: List[str] = []
    for node, stats in node_stats.items():
        if not isinstance(stats, dict):
            continue
        node_name = str(node)
        touched_nodes.append(node_name)

        stat_key = _node_key(node_level, node_name)
        stat_block = node_level.setdefault(stat_key, {})
        if not isinstance(stat_block, dict):
            stat_block = {}
            node_level[stat_key] = stat_block

        evaluated = max(1, _safe_int(stats.get("evaluated"), 1))
        improved = max(0, _safe_int(stats.get("improved"), 0))
        success_rate = _safe_float(stats.get("success_rate"), improved / evaluated)
        mean_reward = _safe_float(stats.get("mean_reward"), 0.0)
        priority_delta = max(-0.25, min(0.35, success_rate * 0.25 + mean_reward * 0.5))
        priority_multiplier = max(0.5, min(1.75, 1.0 + priority_delta))

        stat_block["evaluated_total"] = _safe_int(stat_block.get("evaluated_total"), 0) + evaluated
        stat_block["improved_total"] = _safe_int(stat_block.get("improved_total"), 0) + improved
        stat_block["last_success_rate"] = _round(success_rate, 4)
        stat_block["last_mean_reward"] = _round(mean_reward, 6)
        stat_block["last_best_fast_loss"] = _round(stats.get("best_fast_loss"), 6)
        stat_block["priority_multiplier"] = _round(priority_multiplier, 4)
        stat_block["last_update"] = timestamp

        node_changes = mutations_by_node.get(node_name, [])
        residue_counter = _counter_from_dict(stat_block.get("top_residue_frequency", {}))
        class_counter = _counter_from_dict(stat_block.get("top_class_frequency", {}))
        for change in node_changes:
            aa_to = str(change.get("to", "")).upper()
            residue_counter[aa_to] += 1
            for class_name in _aa_classes(aa_to):
                class_counter[class_name] += 1
        stat_block["top_residue_frequency"] = _top_counter_dict(residue_counter, max_residues_per_node)
        stat_block["top_class_frequency"] = _top_counter_dict(class_counter, 5)

        if node_changes:
            mean_mut_count = len(node_changes)
            stat_block["mean_mutation_count"] = _round(
                _blend(stat_block.get("mean_mutation_count"), mean_mut_count, alpha=0.25),
                4,
            )

        node_plddt = (search_output.get("node_plddt", {}) or {}).get(node_name, {})
        if isinstance(node_plddt, dict):
            for key in ("plddt_mean", "plddt_min", "plddt_max"):
                if key in node_plddt:
                    stat_block[f"last_{key}"] = _round(node_plddt.get(key), 4)

        motif_key = _node_key(motif_memory, node_name)
        motif = motif_memory.setdefault(motif_key, {})
        if not isinstance(motif, dict):
            motif = {}
            motif_memory[motif_key] = motif
        motif.setdefault("enriched_motifs", [])
        motif.setdefault("depleted_patterns", [])
        motif["enriched_residues"] = list(stat_block.get("top_residue_frequency", {}).keys())[
            :max_residues_per_node
        ]
        motif["favored_classes"] = list(stat_block.get("top_class_frequency", {}).keys())[:5]
        motif["support_count"] = _safe_int(motif.get("support_count"), 0) + improved
        old_conf = _safe_float(motif.get("confidence"), 0.0)
        conf_delta = max(0.0, min(0.12, 0.035 * improved + 0.04 * success_rate + max(0.0, mean_reward) * 0.15))
        motif["confidence"] = _round(min(0.85, old_conf * 0.92 + conf_delta), 4)
        motif["last_update"] = timestamp

    run_id = f"run_{_safe_int(round_summary.get('created_at_unix'), int(datetime.now().timestamp()))}"
    artifact_paths = search_artifacts.get("artifact_paths", {}) or {}
    round_path = artifact_paths.get("round_summary")
    if round_path:
        run_id = Path(str(round_path)).parent.name

    compact_run = {
        "run_id": run_id,
        "updated_at": timestamp,
        "search_method": search_output.get("search_method", round_summary.get("search_method", "")),
        "best_node_id": search_artifacts.get("best_node_id", round_summary.get("best_node_id")),
        "best_path": search_artifacts.get("best_path", round_summary.get("best_path", [])),
        "seq_hash": _seq_hash(search_output.get("seqs")),
        "fast_loss": _round(search_output.get("fast_loss"), 6),
        "constraint_penalty": _round(search_output.get("constraint_penalty"), 6),
        "progen_loglik_avg": _round(search_output.get("progen_loglik_avg"), 6),
        "chai_plddt": _round(search_output.get("chai_plddt"), 4),
        "effective_mutations": effective_mutations[:max_residues_per_node],
    }

    candidate_stats = memory.setdefault("candidate_statistics", {})
    if not isinstance(candidate_stats, dict):
        candidate_stats = {}
        memory["candidate_statistics"] = candidate_stats
    recent = candidate_stats.setdefault("recent_inner_loop_runs", [])
    if not isinstance(recent, list):
        recent = []
    recent = [item for item in recent if not isinstance(item, dict) or item.get("run_id") != run_id]
    recent.insert(0, compact_run)
    recent = recent[: max(1, int(max_recent_runs))]
    candidate_stats["recent_inner_loop_runs"] = recent
    candidate_stats["max_recent_inner_loop_runs"] = max(1, int(max_recent_runs))
    candidate_stats["max_residues_per_node"] = max(1, int(max_residues_per_node))

    best_seen = candidate_stats.get("best_seen")
    current_loss = _safe_float(search_output.get("chai_combined_loss", search_output.get("fast_loss")), 1e9)
    previous_loss = _safe_float(best_seen.get("combined_loss") if isinstance(best_seen, dict) else None, 1e9)
    if current_loss <= previous_loss:
        candidate_stats["best_seen"] = {
            "run_id": run_id,
            "updated_at": timestamp,
            "seq_hash": compact_run["seq_hash"],
            "combined_loss": _round(current_loss, 6),
            "fast_loss": compact_run["fast_loss"],
            "chai_plddt": compact_run["chai_plddt"],
        }

    run_state = memory.setdefault("run_state", {})
    if not isinstance(run_state, dict):
        run_state = {}
        memory["run_state"] = run_state
    run_state["last_inner_loop_run"] = {
        "run_id": run_id,
        "updated_at": timestamp,
        "touched_nodes": touched_nodes,
        "num_effective_mutations": len(effective_mutations),
        "memory_policy": "bounded_summary_only",
    }

    result = {
        "updated": True,
        "path": str(memory_path),
        "run_id": run_id,
        "touched_nodes": touched_nodes,
        "effective_mutations": len(effective_mutations),
        "recent_run_count": len(recent),
        "dry_run": bool(dry_run),
    }

    if dry_run:
        result["memory"] = memory
        return result

    try:
        memory_path.write_text(
            yaml.safe_dump(memory, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
    except Exception as exc:
        return {"updated": False, "reason": f"failed to write memory: {exc}"}

    return result
