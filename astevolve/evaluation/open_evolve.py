import importlib.util
import time
import numpy as np
import traceback
import concurrent.futures
import os
from astevolve.metrics.structure import metric_value
try:
    from openevolve.evaluation_result import EvaluationResult
except ModuleNotFoundError:
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass
    class EvaluationResult:  # local fallback for smoke tests outside OpenEvolve
        metrics: Dict[str, float]
        artifacts: Dict[str, Any]

ARTIFACTS_DIR = "artifacts"


def _case_id_from_program_path(program_path):
    try:
        parts = os.path.abspath(program_path).split(os.sep)
        if "cases" in parts:
            idx = parts.index("cases")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    return "default"


def _best_sequence_dir(program_path):
    path = os.path.join(ARTIFACTS_DIR, _case_id_from_program_path(program_path), "best_sequences")
    os.makedirs(path, exist_ok=True)
    return path


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=10):
    if kwargs is None: kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        return fut.result(timeout=timeout_seconds)


def save_fasta(program_path, seqs, energy, combined_score, plddt, avg_plddt_delta, progen, structure_metrics=None):
    try:
        base_name = os.path.splitext(os.path.basename(program_path))[0]
        filename = f"{base_name}_E{energy:.2f}_S{combined_score:.2f}_P{plddt:.1f}.fasta"
        filepath = os.path.join(_best_sequence_dir(program_path), filename)

        with open(filepath, "w") as f:
            f.write(f"# Program: {base_name}\n")
            f.write(f"# Total Loss: {energy}\n")
            f.write(f"# Combined Score: {combined_score}\n")
            f.write(f"# pLDDT: {plddt}\n")
            f.write(f'# avg_plddt_delta: {avg_plddt_delta}\n')
            f.write(f"# ProGen loglik avg: {progen}\n")
            if structure_metrics:
                scalar = structure_metrics.get("scalar", {}) or {}
                interface = structure_metrics.get("interface", {}) or {}
                f.write(f"# ptm: {scalar.get('ptm')}\n")
                f.write(f"# iptm: {scalar.get('iptm')}\n")
                f.write(f"# ranking_score: {scalar.get('ranking_score')}\n")
                f.write(f"# interface_plddt_mean: {interface.get('interface_plddt_mean')}\n")
                f.write(f"# interface_contact_count: {interface.get('total_contact_count')}\n")
            for chain_id, seq in seqs.items():
                f.write(f">Chain_{chain_id}\n")
                f.write(f"{seq}\n")
        return filepath
    except Exception as e:
        print(f"Error saving FASTA: {e}")
        return None


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _score_from_100(value):
    return _clamp01(float(value) / 100.0) if value is not None else 0.0


def _structure_value(out, key, default=0.0):
    summary = out.get("structure_metrics", {}) or {}
    if key == "plddt" and not summary:
        return float(out.get("chai_plddt") or default)
    return metric_value(summary, key, default=default)


def _compute_combined_score(total_loss, out, score_cfg):
    w_fast = float(score_cfg.get("weight_fast", 0.5))
    w_plddt = float(score_cfg.get("weight_plddt", 0.5))
    w_iptm = float(score_cfg.get("weight_iptm", 0.0))
    w_ptm = float(score_cfg.get("weight_ptm", 0.0))
    w_ranking = float(score_cfg.get("weight_ranking_score", 0.0))
    w_interface_plddt = float(score_cfg.get("weight_interface_plddt", 0.0))
    w_node_min = float(score_cfg.get("weight_node_plddt_min", 0.0))
    w_clash = float(score_cfg.get("weight_clash", 0.0))
    w_multistate = float(score_cfg.get("weight_multistate", 0.0))
    plddt_scale = float(score_cfg.get("plddt_scale", 100.0))
    clash_scale = max(1.0, float(score_cfg.get("clash_scale", 10.0)))
    clamp_nonneg = bool(score_cfg.get("fast_loss_nonneg", True))

    loss = max(0.0, total_loss) if clamp_nonneg else float(total_loss)
    fast_score = 1.0 / (1.0 + max(0.0, loss))

    plddt = _structure_value(out, "plddt", out.get("chai_plddt") or 0.0)
    ptm = _structure_value(out, "ptm", 0.0)
    iptm = _structure_value(out, "iptm", 0.0)
    ranking_score = _structure_value(out, "ranking_score", 0.0)
    interface_plddt = _structure_value(out, "interface_plddt_mean", 0.0)
    node_plddt_min = _structure_value(out, "node_plddt_min", 0.0)
    clash_count = _structure_value(out, "clash_count", 0.0)
    has_clash = _structure_value(out, "has_clash", 0.0)
    multistate_pack = out.get("multistate_objectives", {}) or {}
    multistate_score = _clamp01(out.get("multistate_score", multistate_pack.get("normalized_score", 0.0)))

    plddt_score = _clamp01(float(plddt) / plddt_scale) if plddt is not None else 0.0
    iptm_score = _clamp01(iptm)
    ptm_score = _clamp01(ptm)
    ranking_score_component = _clamp01(ranking_score)
    interface_plddt_score = _score_from_100(interface_plddt)
    node_plddt_min_score = _score_from_100(node_plddt_min)
    clash_penalty = max(_clamp01(has_clash), _clamp01(float(clash_count) / clash_scale))

    structure_score = (
        (w_plddt * plddt_score)
        + (w_iptm * iptm_score)
        + (w_ptm * ptm_score)
        + (w_ranking * ranking_score_component)
        + (w_interface_plddt * interface_plddt_score)
        + (w_node_min * node_plddt_min_score)
        + (w_multistate * multistate_score)
        - (w_clash * clash_penalty)
    )
    combined = (w_fast * fast_score) + structure_score
    return {
        "combined_score": float(combined),
        "fast_score": float(fast_score),
        "plddt_score": float(plddt_score),
        "iptm_score": float(iptm_score),
        "ptm_score": float(ptm_score),
        "ranking_score_component": float(ranking_score_component),
        "interface_plddt_score": float(interface_plddt_score),
        "node_plddt_min_score": float(node_plddt_min_score),
        "multistate_score": float(multistate_score),
        "clash_penalty": float(clash_penalty),
        "structure_score": float(structure_score),
        "plddt": float(plddt or 0.0),
        "ptm": float(ptm or 0.0),
        "iptm": float(iptm or 0.0),
        "ranking_score": float(ranking_score or 0.0),
        "interface_plddt_mean": float(interface_plddt or 0.0),
        "interface_contact_count": float(_structure_value(out, "interface_contact_count", 0.0)),
        "interface_residue_pair_count": float(_structure_value(out, "interface_residue_pair_count", 0.0)),
        "clash_count": float(clash_count or 0.0),
        "node_plddt_mean": float(_structure_value(out, "node_plddt_mean", 0.0)),
        "node_plddt_min": float(node_plddt_min or 0.0),
    }


def evaluate(program_path: str):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "run_search"):
            return EvaluationResult(metrics={"combined_score": 0.0}, artifacts={"error": "Missing run_search"})

        num_trials = 1
        total_losses = []
        progen_scores = []
        plddt_deltas = []
        trial_scores = []
        success = 0

        best_trial_score = float("-inf")
        best_trial_out = None

        for t in range(num_trials):
            try:
                out = run_with_timeout(program.run_search, kwargs={"seed": t}, timeout_seconds=None)
                if not out:
                    print(f"[trial {t}] out is None/False")
                    continue
                if "seqs" not in out:
                    print(f"[trial {t}] missing 'seqs' keys: {list(out.keys())}")
                    continue

                total_loss = float(out.get("fast_loss", 0.0))
                progen = float(out.get("progen_loglik_avg", 0.0))
                chai_results = out.get("chai_results", [])
                plddt_delta = out.get("plddt_delta", None)
                if len(chai_results) > 0:
                    plddt_delta = chai_results[0].get("plddt_delta", plddt_delta)

                score_cfg = out.get("score_config", {})
                score_pack = _compute_combined_score(total_loss, out, score_cfg)
                combined = score_pack["combined_score"]

                total_losses.append(total_loss)
                progen_scores.append(progen)
                if plddt_delta is not None:
                    plddt_deltas.append(float(plddt_delta))
                trial_scores.append(score_pack)

                success += 1

                if combined > best_trial_score:
                    best_trial_score = combined
                    best_trial_out = out

            except Exception as e:
                print(f"[trial {t}] Exception: {e}")
                traceback.print_exc()
                continue

        if success == 0:
            return EvaluationResult(metrics={"combined_score": 0.0}, artifacts={"error": "All trials failed"})

        avg_loss = float(np.mean(total_losses)) if total_losses else 0.0
        avg_progen = float(np.mean(progen_scores)) if progen_scores else 0.0
        avg_plddt_delta = float(np.mean(plddt_deltas)) if plddt_deltas else 0.0
        avg_scores = {}
        for key in (trial_scores[0].keys() if trial_scores else []):
            avg_scores[key] = float(np.mean([s.get(key, 0.0) for s in trial_scores]))

        score_cfg = best_trial_out.get("score_config", {}) if best_trial_out else {}
        best_score_pack = _compute_combined_score(avg_loss, best_trial_out or {}, score_cfg)
        combined = avg_scores.get("combined_score", best_score_pack["combined_score"])

        saved_path = "N/A"
        if best_trial_out is not None:
            saved_path = save_fasta(
                program_path,
                best_trial_out["seqs"],
                avg_loss,
                combined,
                avg_scores.get("plddt", 0.0),
                avg_plddt_delta,
                avg_progen,
                best_trial_out.get("structure_metrics"),
            )

        return EvaluationResult(
            metrics={
                "combined_score": combined,
                "struct_score": avg_scores.get("structure_score", 0.0),
                "total_loss": avg_loss,
                "fast_score": avg_scores.get("fast_score", 0.0),
                "plddt_score": avg_scores.get("plddt_score", 0.0),
                "progen_loglik_avg": avg_progen,
                "plddt": avg_scores.get("plddt", 0.0),
                "plddt_delta": avg_plddt_delta,
                "ptm": avg_scores.get("ptm", 0.0),
                "iptm": avg_scores.get("iptm", 0.0),
                "ranking_score": avg_scores.get("ranking_score", 0.0),
                "interface_plddt_mean": avg_scores.get("interface_plddt_mean", 0.0),
                "interface_contact_count": avg_scores.get("interface_contact_count", 0.0),
                "interface_residue_pair_count": avg_scores.get("interface_residue_pair_count", 0.0),
                "clash_count": avg_scores.get("clash_count", 0.0),
                "node_plddt_mean": avg_scores.get("node_plddt_mean", 0.0),
                "node_plddt_min": avg_scores.get("node_plddt_min", 0.0),
                "multistate_score": avg_scores.get("multistate_score", 0.0),
                "multistate_loss": float(best_trial_out.get("multistate_loss", 0.0) if best_trial_out else 0.0),
            },
            artifacts={
                "best_seqs": best_trial_out.get("seqs") if best_trial_out else None,
                "segment_scores": best_trial_out.get("segment_scores") if best_trial_out else None,
                "mutation_history": best_trial_out.get("mutation_history") if best_trial_out else None,
                "progen_loglik_avg": best_trial_out.get("progen_loglik_avg") if best_trial_out else None,
                "progen_loglik_sum": best_trial_out.get("progen_loglik_sum") if best_trial_out else None,
                "chai_plddt": best_trial_out.get("chai_plddt") if best_trial_out else None,
                "confidence_metrics": best_trial_out.get("confidence_metrics") if best_trial_out else None,
                "structure_metrics": best_trial_out.get("structure_metrics") if best_trial_out else None,
                "chain_plddt": best_trial_out.get("chain_plddt") if best_trial_out else None,
                "node_plddt": best_trial_out.get("node_plddt") if best_trial_out else None,
                "multistate_objectives": best_trial_out.get("multistate_objectives") if best_trial_out else None,
                "chai_evaluated": best_trial_out.get("chai_evaluated") if best_trial_out else None,
                "chain_lengths": best_trial_out.get("chain_lengths") if best_trial_out else None,
                "blueprint_summary": best_trial_out.get("blueprint_summary") if best_trial_out else None,
                "layout_summary": best_trial_out.get("layout_summary") if best_trial_out else None,
                "strategy_schema_report": best_trial_out.get("strategy_schema_report") if best_trial_out else None,
                "segments": best_trial_out.get("segments") if best_trial_out else None,
                "search_artifacts": best_trial_out.get("search_artifacts") if best_trial_out else None,
                "saved_fasta_path": saved_path,
            }
        )

    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": str(e), "traceback": traceback.format_exc()}
        )
    
if __name__ == "__main__":
    from astevolve.cases import resolve_case

    case = resolve_case()
    entry = case.root / "cases" / case.case_id / str(case.metadata.get("entry_program", "initial_program.py"))
    print(evaluate(str(entry)))
