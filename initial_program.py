# EVOLVE-BLOCK-START
from __future__ import annotations


def propose_strategy():
    """
    OpenEvolve should mutate the design strategy tree here, not the CD25 case
    identity. Fixed case data lives in design_state.json and engine/.

    The tree mirrors the AST hierarchy. Internal nodes describe domains/modules;
    leaf nodes describe concrete editable sequence segments. The engine will
    translate this tree into:
    - editable masks
    - optional segment length changes
    - node-specific mutation priors
    - MCTS node priorities
    """
    return {
        "strategy_tree": {
            "name": "CD25_scFv_Design_Task",
            "kind": "complex",
            "mutable": False,
            "children": [
                {
                    "name": "Binder_scFv",
                    "kind": "chain",
                    "chain_id": "BB",
                    "mutable": True,
                    "children": [
                        {
                            "name": "VH_domain",
                            "kind": "domain",
                            "mutable": True,
                            "children": [
                                {
                                    "name": "VH_FR1",
                                    "kind": "framework",
                                    "mutable": False,
                                    "target_length": 21,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "preserve heavy-chain framework entry stability",
                                        "priority_boost": 0.05,
                                    },
                                },
                                {
                                    "name": "VH_CDR1",
                                    "kind": "cdr",
                                    "mutable": True,
                                    "target_length": 14,
                                    "length_range": [11, 15],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "tune polar/aromatic contacts while preserving loop plausibility",
                                        "priority_boost": 1.15,
                                        "mutation_rate": 0.05,
                                        "max_mutations_per_step": 2,
                                        "mutation_ops": {"point": 0.82, "block": 0.12, "segment_resample": 0.04, "swap": 0.02},
                                        "favored_residues": ["Y", "S", "T", "N", "Q", "H", "R", "D"],
                                        "favored_residue_classes": ["aromatic", "polar_uncharged"],
                                        "disfavored_residues": ["C"],
                                        "policy_weight": 0.65,
                                    },
                                },
                                {
                                    "name": "VH_FR2",
                                    "kind": "framework",
                                    "mutable": True,
                                    "target_length": 13,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "allow sparse support mutations near heavy-chain CDR geometry",
                                        "priority_boost": 0.40,
                                        "mutation_rate": 0.015,
                                        "max_mutations_per_step": 1,
                                        "mutation_ops": {"point": 0.95, "block": 0.03, "segment_resample": 0.0, "swap": 0.02},
                                        "favored_residues": ["S", "T", "N", "Q", "G"],
                                        "disfavored_residues": ["C", "W", "F"],
                                        "policy_weight": 0.30,
                                    },
                                },
                                {
                                    "name": "VH_CDR2",
                                    "kind": "cdr",
                                    "mutable": True,
                                    "target_length": 16,
                                    "length_range": [12, 18],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "shape secondary paratope contacts and electrostatic complementarity",
                                        "priority_boost": 1.45,
                                        "mutation_rate": 0.065,
                                        "max_mutations_per_step": 3,
                                        "mutation_ops": {"point": 0.72, "block": 0.18, "segment_resample": 0.07, "swap": 0.03},
                                        "favored_residues": ["Y", "H", "S", "T", "N", "Q", "R", "D", "E"],
                                        "favored_residue_classes": ["aromatic", "polar_uncharged", "contextual_charge"],
                                        "disfavored_residues": ["C"],
                                        "policy_weight": 0.75,
                                    },
                                },
                                {
                                    "name": "VH_FR3",
                                    "kind": "framework",
                                    "mutable": True,
                                    "target_length": 33,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "permit sparse framework support mutations without destabilizing VH",
                                        "priority_boost": 0.35,
                                        "mutation_rate": 0.012,
                                        "max_mutations_per_step": 1,
                                        "mutation_ops": {"point": 0.96, "block": 0.02, "segment_resample": 0.0, "swap": 0.02},
                                        "favored_residues": ["S", "T", "N", "Q", "G", "A"],
                                        "disfavored_residues": ["C", "W"],
                                        "policy_weight": 0.25,
                                    },
                                },
                                {
                                    "name": "VH_CDR3",
                                    "kind": "cdr",
                                    "mutable": True,
                                    "target_length": 8,
                                    "length_range": [6, 18],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "dominant hotspot-facing loop; optimize shape, charge, and aromatic packing",
                                        "priority_boost": 2.80,
                                        "mutation_rate": 0.11,
                                        "max_mutations_per_step": 4,
                                        "mutation_ops": {"point": 0.58, "block": 0.24, "segment_resample": 0.14, "swap": 0.04},
                                        "favored_residues": ["Y", "W", "H", "R", "D", "E", "S", "T", "N", "Q", "G"],
                                        "favored_residue_classes": ["aromatic", "polar_uncharged", "contextual_charge", "turn_loop"],
                                        "disfavored_residues": ["C"],
                                        "policy_weight": 0.95,
                                    },
                                },
                                {
                                    "name": "VH_FR4",
                                    "kind": "framework",
                                    "mutable": False,
                                    "target_length": 10,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "preserve VH terminal framework",
                                        "priority_boost": 0.05,
                                    },
                                },
                            ],
                        },
                        {
                            "name": "Linker_module",
                            "kind": "linker",
                            "mutable": True,
                            "children": [
                                {
                                    "name": "Linker_head",
                                    "kind": "linker",
                                    "mutable": False,
                                    "target_length": 5,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "preserve linker entry spacing",
                                        "priority_boost": 0.05,
                                    },
                                },
                                {
                                    "name": "Linker_core",
                                    "kind": "linker",
                                    "mutable": True,
                                    "target_length": 5,
                                    "length_range": [5, 10],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "tune VH-VL spacing while staying glycine/serine rich",
                                        "priority_boost": 0.65,
                                        "mutation_rate": 0.04,
                                        "max_mutations_per_step": 2,
                                        "mutation_ops": {"point": 0.70, "block": 0.16, "segment_resample": 0.12, "swap": 0.02},
                                        "favored_residues": ["G", "S", "A", "T"],
                                        "favored_residue_classes": ["flexible_small"],
                                        "disfavored_residue_classes": ["hydrophobic", "charged"],
                                        "fill_residues": "GGGGS",
                                        "policy_weight": 0.80,
                                    },
                                },
                                {
                                    "name": "Linker_tail",
                                    "kind": "linker",
                                    "mutable": False,
                                    "target_length": 5,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "preserve linker exit spacing",
                                        "priority_boost": 0.05,
                                    },
                                },
                            ],
                        },
                        {
                            "name": "VL_domain",
                            "kind": "domain",
                            "mutable": True,
                            "children": [
                                {
                                    "name": "VL_FR1",
                                    "kind": "framework",
                                    "mutable": False,
                                    "target_length": 23,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "preserve light-chain framework entry stability",
                                        "priority_boost": 0.05,
                                    },
                                },
                                {
                                    "name": "VL_CDR1",
                                    "kind": "cdr",
                                    "mutable": True,
                                    "target_length": 10,
                                    "length_range": [8, 12],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "adjust light-chain polar contact support",
                                        "priority_boost": 1.00,
                                        "mutation_rate": 0.045,
                                        "max_mutations_per_step": 2,
                                        "mutation_ops": {"point": 0.80, "block": 0.14, "segment_resample": 0.04, "swap": 0.02},
                                        "favored_residues": ["Y", "S", "T", "N", "Q", "H", "D"],
                                        "favored_residue_classes": ["polar_uncharged"],
                                        "disfavored_residues": ["C"],
                                        "policy_weight": 0.60,
                                    },
                                },
                                {
                                    "name": "VL_FR2",
                                    "kind": "framework",
                                    "mutable": True,
                                    "target_length": 15,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "allow sparse VL support changes near CDR2",
                                        "priority_boost": 0.28,
                                        "mutation_rate": 0.01,
                                        "max_mutations_per_step": 1,
                                        "mutation_ops": {"point": 0.96, "block": 0.02, "segment_resample": 0.0, "swap": 0.02},
                                        "favored_residues": ["S", "T", "N", "Q", "G"],
                                        "disfavored_residues": ["C", "W", "F"],
                                        "policy_weight": 0.22,
                                    },
                                },
                                {
                                    "name": "VL_CDR2",
                                    "kind": "cdr",
                                    "mutable": True,
                                    "target_length": 7,
                                    "length_range": [7, 8],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "small light-chain contact loop; prefer sparse polar edits",
                                        "priority_boost": 0.85,
                                        "mutation_rate": 0.04,
                                        "max_mutations_per_step": 1,
                                        "mutation_ops": {"point": 0.88, "block": 0.08, "segment_resample": 0.02, "swap": 0.02},
                                        "favored_residues": ["S", "T", "N", "Q", "Y", "D"],
                                        "favored_residue_classes": ["polar_uncharged"],
                                        "disfavored_residues": ["C"],
                                        "policy_weight": 0.55,
                                    },
                                },
                                {
                                    "name": "VL_FR3",
                                    "kind": "framework",
                                    "mutable": True,
                                    "target_length": 32,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "permit sparse VL framework support mutations",
                                        "priority_boost": 0.30,
                                        "mutation_rate": 0.01,
                                        "max_mutations_per_step": 1,
                                        "mutation_ops": {"point": 0.96, "block": 0.02, "segment_resample": 0.0, "swap": 0.02},
                                        "favored_residues": ["S", "T", "N", "Q", "G", "A"],
                                        "disfavored_residues": ["C", "W"],
                                        "policy_weight": 0.22,
                                    },
                                },
                                {
                                    "name": "VL_CDR3",
                                    "kind": "cdr",
                                    "mutable": True,
                                    "target_length": 9,
                                    "length_range": [7, 12],
                                    "length_mutable": True,
                                    "edit_policy": {
                                        "edit_intent": "light-chain partner loop for interface shape and polarity",
                                        "priority_boost": 1.55,
                                        "mutation_rate": 0.07,
                                        "max_mutations_per_step": 3,
                                        "mutation_ops": {"point": 0.68, "block": 0.20, "segment_resample": 0.08, "swap": 0.04},
                                        "favored_residues": ["Y", "H", "S", "T", "N", "Q", "R", "D", "G"],
                                        "favored_residue_classes": ["aromatic", "polar_uncharged", "turn_loop"],
                                        "disfavored_residues": ["C"],
                                        "policy_weight": 0.75,
                                    },
                                },
                                {
                                    "name": "VL_FR4",
                                    "kind": "framework",
                                    "mutable": False,
                                    "target_length": 14,
                                    "length_mutable": False,
                                    "edit_policy": {
                                        "edit_intent": "preserve VL terminal framework",
                                        "priority_boost": 0.05,
                                    },
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "Target_CD25",
                    "kind": "chain",
                    "chain_id": "T",
                    "mutable": False,
                    "children": [
                        {
                            "name": "CD25_basiliximab_epitope",
                            "kind": "epitope",
                            "mutable": False,
                            "edit_policy": {
                                "edit_intent": "fixed target epitope prior; use for conditioning only",
                                "priority_boost": 0.0,
                            },
                        }
                    ],
                },
            ],
        },
        # Fallback for older engine versions. The current engine derives this
        # from strategy_tree, so OpenEvolve should treat this as secondary.
        "preferred_edit_order": [],
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
        "protenix_conda_env": "pytorch",
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
