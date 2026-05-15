# EVOLVE-BLOCK-START
from __future__ import annotations


def propose_strategy():
    """
    OpenEvolve should mutate strategy knobs here, not the CD25 case identity.
    Fixed case data lives in design_state.json and engine/.
    """
    return {
        "preferred_edit_order": [
            "VH_CDR3",
            "VH_CDR2",
            "VL_CDR3",
            "VH_CDR1",
            "VL_CDR1",
            "VL_CDR2",
            "Linker_core",
            "VH_FR2",
            "VH_FR3",
            "VL_FR2",
            "VL_FR3",
        ],
        "iterations": 1200,
        "init_temp": 2.0,
        "cooling": 0.995,
        "mutation_rate": 0.06,
        "resample_segment_prob": 0.08,
        "mutation_ops": {
            "point": 0.75,
            "block": 0.15,
            "segment_resample": 0.07,
            "swap": 0.03,
        },
        "search_method": "mcts",
        "mcts_c_puct": 1.4,
        "mcts_max_depth": 4,
        "mcts_reward_scale": 1.0,
        "mcts_output_dir": "inner_loop",
        "mcts_save_tree": True,
        "mcts_save_variants": True,
        "mcts_memory_enabled": True,
        "external_kb_enabled": True,
        "external_kb_path": "data/antibody_kb/sabdab_external_prior_cache.json",
        "external_kb_weight": 0.7,
        "external_kb_embedding_manifest": "data/antibody_kb/embedding_manifest_esm2_t6_8M_sabdab_cdr.json",
        "external_kb_retrieval_enabled": True,
        "external_kb_retrieval_top_k": 20,
        "external_kb_retrieval_weight": 0.6,
        "external_kb_device": "auto",
        "external_kb_max_length": 128,
        "progen_weight": 1.0,
        "chai1_enabled": True,
        "chai1_top_frac": 0.01,
        "chai1_min_candidates": 1,
        "chai1_max_candidates": 3,
        "history_size": 50,
        "linker_gs_min": 0.60,
        "linker_hydrophobic_max": 0.15,
        "linker_charged_max": 0.20,
        "cdr_favored_residues": ["Y", "W", "H", "N", "Q", "S", "T", "R", "D", "E"],
        "cdr_hydrophobic_max": 0.40,
        "cdr_charged_max": 0.45,
        "desired_cdr3_hydro": 0.30,
        "max_hydrophobic_run": 2,
        "max_charged_run": 2,
        "score_config": {
            "weight_fast": 1.0,
            "weight_plddt": 5.0,
            "plddt_scale": 100.0,
            "fast_loss_nonneg": True,
        },
    }


# EVOLVE-BLOCK-END

from typing import Optional

from engine.case_builder import build_case_inputs, run_design_search


def run_search(seed: Optional[int] = None):
    return run_design_search(propose_strategy(), seed=seed)


def preview_case():
    bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg, state = build_case_inputs(
        propose_strategy()
    )
    compiled = bp.compile()
    return {
        "task_name": state["task_name"],
        "chain_lengths": compiled["chain_lengths"],
        "chain_order": compiled["chain_order"],
        "segments": [
            {
                "chain_id": s.chain_id,
                "kind": s.kind,
                "name": s.name,
                "spans": s.spans,
                "total_length": s.total_length,
            }
            for s in compiled["segments"]
        ],
        "constraint_kinds": [x["kind"] for x in constraint_specs],
        "sa_config": sa_cfg,
        "mask_true_counts": {k: int(sum(v)) for k, v in masks.items()},
        "template_lengths": {k: len(v) for k, v in templates.items()},
        "fixed_residue_counts": {k: len(v) for k, v in fixed_residues.items()},
        "score_config": score_cfg,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(preview_case(), ensure_ascii=False, indent=2))
