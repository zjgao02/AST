import importlib.util
import time
import numpy as np
import traceback
import concurrent.futures
import os
from openevolve.evaluation_result import EvaluationResult

OUTPUT_DIR = "best_sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=10):
    if kwargs is None: kwargs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        return fut.result(timeout=timeout_seconds)


def save_fasta(program_path, seqs, energy, combined_score, plddt, avg_plddt_delta, progen):
    try:
        base_name = os.path.splitext(os.path.basename(program_path))[0]
        filename = f"{base_name}_E{energy:.2f}_S{combined_score:.2f}_P{plddt:.1f}.fasta"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w") as f:
            f.write(f"# Program: {base_name}\n")
            f.write(f"# Total Loss: {energy}\n")
            f.write(f"# Combined Score: {combined_score}\n")
            f.write(f"# pLDDT: {plddt}\n")
            f.write(f'# avg_plddt_delta: {avg_plddt_delta}\n')
            f.write(f"# ProGen loglik avg: {progen}\n")
            for chain_id, seq in seqs.items():
                f.write(f">Chain_{chain_id}\n")
                f.write(f"{seq}\n")
        return filepath
    except Exception as e:
        print(f"Error saving FASTA: {e}")
        return None


def _compute_combined_score(total_loss, plddt, score_cfg):
    w_fast = float(score_cfg.get("weight_fast", 0.5))
    w_plddt = float(score_cfg.get("weight_plddt", 0.5))
    plddt_scale = float(score_cfg.get("plddt_scale", 100.0))
    clamp_nonneg = bool(score_cfg.get("fast_loss_nonneg", True))

    loss = max(0.0, total_loss) if clamp_nonneg else float(total_loss)
    fast_score = 1.0 / (1.0 + max(0.0, loss))

    plddt_score = 0.0
    if plddt is not None:
        plddt_score = max(0.0, min(1.0, float(plddt) / plddt_scale))

    combined = (w_fast * fast_score) + (w_plddt * plddt_score)
    return float(combined), float(fast_score), float(plddt_score)


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
        plddts = []
        plddt_deltas = []
        success = 0

        best_trial_score = -1.0
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
                plddt = out.get("chai_plddt", None)
                chai_results = out.get("chai_results", [])
                plddt_delta = None
                if len(chai_results) > 0:
                    plddt_delta = chai_results[0].get("plddt_delta", None)

                score_cfg = out.get("score_config", {})
                # combined, fast_score, plddt_score = _compute_combined_score(total_loss, plddt, score_cfg)
                combined, fast_score, plddt_score = _compute_combined_score(total_loss, plddt_delta, score_cfg)

                total_losses.append(total_loss)
                progen_scores.append(progen)
                if plddt is not None:
                    plddts.append(float(plddt))
                if plddt_delta is not None:
                    plddt_deltas.append(float(plddt_delta))

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
        avg_plddt = float(np.mean(plddts)) if plddts else 0.0
        avg_plddt_delta = float(np.mean(plddt_deltas)) if plddt_deltas else 0.0

        score_cfg = best_trial_out.get("score_config", {}) if best_trial_out else {}
        combined, fast_score, plddt_score = _compute_combined_score(avg_loss, avg_plddt, score_cfg)

        saved_path = "N/A"
        if best_trial_out is not None:
            saved_path = save_fasta(
                program_path,
                best_trial_out["seqs"],
                avg_loss,
                combined,
                avg_plddt,
                avg_plddt_delta,
                avg_progen,
            )

        return EvaluationResult(
            metrics={
                "combined_score": combined,
                "iptm": avg_plddt,
                "iptm_delta": avg_plddt_delta,
                "total_loss": avg_loss,
                "fast_score": fast_score,
                "plddt_score": plddt_score,
                "progen_loglik_avg": avg_progen,
            },
            artifacts={
                "best_seqs": best_trial_out.get("seqs") if best_trial_out else None,
                "segment_scores": best_trial_out.get("segment_scores") if best_trial_out else None,
                "mutation_history": best_trial_out.get("mutation_history") if best_trial_out else None,
                "progen_loglik_avg": best_trial_out.get("progen_loglik_avg") if best_trial_out else None,
                "progen_loglik_sum": best_trial_out.get("progen_loglik_sum") if best_trial_out else None,
                "chai_plddt": best_trial_out.get("chai_plddt") if best_trial_out else None,
                "chai_evaluated": best_trial_out.get("chai_evaluated") if best_trial_out else None,
                "chain_lengths": best_trial_out.get("chain_lengths") if best_trial_out else None,
                "blueprint_summary": best_trial_out.get("blueprint_summary") if best_trial_out else None,
                "segments": best_trial_out.get("segments") if best_trial_out else None,
                "saved_fasta_path": saved_path,
            }
        )

    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": str(e), "traceback": traceback.format_exc()}
        )
    
if __name__ == "__main__":
    r = evaluate('./initial_program.py')
    print(r)