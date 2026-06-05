# EVOLVE-BLOCK-START
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    env_root = os.environ.get("ASTEVOLVE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), here.parent, *here.parents]
    for candidate in candidates:
        if (candidate / "astevolve").is_dir() and (candidate / "engine").is_dir():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.default_strategy import base_strategy


def propose_strategy():
    """
    OpenEvolve should evolve the AST layout strategy here.

    It should change which design regions exist, which concrete nodes each
    region controls, how length budgets/ranges are assigned, and what mutation
    priors guide the inner loop. It should not hard-code final antibody
    sequences, change runtime settings, or change the fixed CD25 case identity.
    """
    strategy = base_strategy()

    strategy["layout_plan"] = {
        "binder_domain_order": ["VH_domain", "Linker_module", "VL_domain"],
        "secondary_structure_priors": {
            "VH_CDR1": "loop",
            "VH_CDR2": "loop",
            "VH_CDR3": "loop",
            "VL_CDR1": "loop",
            "VL_CDR2": "loop",
            "VL_CDR3": "loop",
            "VH_FR2": "beta",
            "VH_FR3": "beta",
            "VL_FR2": "beta",
            "VL_FR3": "beta",
        },
        "design_regions": [
            {
                "name": "primary_heavy_chain_hotspot",
                "role": "dominant CD25 hotspot-facing paratope window",
                "position": 1,
                "bind_to": ["VH_CDR3", "VH_CDR2"],
                "secondary_structure": "loop",
                "priority_boost": 1.35,
                "length_budget": 25,
                "node_weights": {"VH_CDR3": 1.7, "VH_CDR2": 1.0},
                "length_ranges": {"VH_CDR3": [7, 18], "VH_CDR2": [12, 18]},
                "site_anchors": {
                    "VH_CDR3": {
                        "relative_ranges": [[2, 7]],
                        "weight": 2.4,
                        "favored_residues": ["Y", "W", "H", "R", "D", "E"],
                        "favored_residue_classes": ["aromatic", "contextual_charge"],
                    },
                    "VH_CDR2": {
                        "relative_positions": [3, 6, 9],
                        "weight": 1.7,
                        "favored_residues": ["Y", "H", "S", "T", "N", "Q"],
                    },
                },
                "mutation_rate": 0.10,
                "max_mutations_per_step": 8,
                "operator_phase": "explore",
                "large_jump": True,
                "motif_candidates": ["YH", "YW", "RY", "DY", "STY"],
                "mutation_ops": {
                    "point": 0.25,
                    "block": 0.20,
                    "site_resample": 0.22,
                    "segment_mutagenesis": 0.18,
                    "motif_graft": 0.12,
                    "swap": 0.03,
                },
                "favored_residues": ["Y", "W", "H", "R", "D", "E", "S", "T", "N", "Q", "G"],
                "favored_residue_classes": ["aromatic", "polar_uncharged", "contextual_charge", "turn_loop"],
                "disfavored_residues": ["C"],
                "policy_weight": 0.95,
            },
            {
                "name": "light_chain_shape_support",
                "role": "VL loops shape the partner side of the CD25 interface",
                "position": 2,
                "bind_to": ["VL_CDR3", "VL_CDR1", "VL_CDR2"],
                "secondary_structure": "loop",
                "priority_boost": 1.05,
                "length_budget": 27,
                "node_weights": {"VL_CDR3": 1.35, "VL_CDR1": 1.0, "VL_CDR2": 0.7},
                "length_ranges": {"VL_CDR3": [7, 12], "VL_CDR1": [8, 12], "VL_CDR2": [7, 8]},
                "site_anchors": {
                    "VL_CDR3": {
                        "relative_ranges": [[2, 6]],
                        "weight": 1.8,
                        "favored_residues": ["Y", "H", "S", "T", "N", "Q"],
                    }
                },
                "mutation_rate": 0.055,
                "max_mutations_per_step": 5,
                "operator_phase": "refine",
                "motif_candidates": ["YH", "SY", "TY", "NQ"],
                "mutation_ops": {
                    "point": 0.42,
                    "block": 0.18,
                    "site_resample": 0.20,
                    "segment_mutagenesis": 0.10,
                    "motif_graft": 0.06,
                    "swap": 0.04,
                },
                "favored_residues": ["Y", "H", "S", "T", "N", "Q", "R", "D", "G"],
                "favored_residue_classes": ["polar_uncharged", "turn_loop"],
                "disfavored_residues": ["C"],
                "policy_weight": 0.72,
            },
            {
                "name": "framework_geometry_support",
                "role": "sparse support edits near CDR geometry without changing framework length",
                "position": 3,
                "bind_to": ["VH_FR2", "VH_FR3", "VL_FR2", "VL_FR3"],
                "secondary_structure": "beta",
                "priority_boost": 0.38,
                "length_mutable": False,
                "mutation_rate": 0.012,
                "max_mutations_per_step": 1,
                "operator_phase": "stabilize",
                "mutation_ops": {
                    "point": 0.96,
                    "block": 0.02,
                    "segment_resample": 0.0,
                    "swap": 0.02,
                },
                "favored_residues": ["S", "T", "N", "Q", "G", "A"],
                "disfavored_residues": ["C", "W"],
                "policy_weight": 0.25,
            },
            {
                "name": "vh_vl_spacing_module",
                "role": "adjust inter-domain spacing while preserving glycine/serine linker behavior",
                "position": 4,
                "bind_to": ["Linker_core"],
                "secondary_structure": "loop",
                "priority_boost": 0.70,
                "target_length": 5,
                "length_range": [5, 10],
                "length_mutable": True,
                "mutation_rate": 0.04,
                "max_mutations_per_step": 2,
                "operator_phase": "stabilize",
                "mutation_ops": {
                    "point": 0.70,
                    "block": 0.16,
                    "segment_resample": 0.12,
                    "swap": 0.02,
                },
                "favored_residues": ["G", "S", "A", "T"],
                "favored_residue_classes": ["flexible_small"],
                "disfavored_residue_classes": ["hydrophobic", "charged"],
                "fill_residues": "GGGGS",
                "policy_weight": 0.80,
            },
        ],
    }

    return strategy


# EVOLVE-BLOCK-END

from copy import deepcopy
from typing import Any, Dict, Optional

from engine.case_builder import build_case_inputs, run_design_search
from engine.default_strategy import base_strategy as _base_runtime_strategy
from astevolve.runtime.case_context import current_case_kwargs


_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "iterations": 1200,
    "init_temp": 2.0,
    "cooling": 0.995,
    "mutation_rate": 0.06,
    "resample_segment_prob": 0.08,
    "mutation_ops": {
        "point": 0.55,
        "block": 0.14,
        "segment_resample": 0.08,
        "site_resample": 0.08,
        "segment_mutagenesis": 0.07,
        "motif_graft": 0.04,
        "region_shuffle": 0.03,
        "swap": 0.03,
    },
    "search_method": "mcts",
    "mcts_c_puct": 1.35,
    "mcts_max_depth": 4,
    "mcts_reward_scale": 1.0,
    "mcts_output_dir": "artifacts/cd25_scfv/inner_loop_runs/latest",
    "mcts_save_tree": True,
    "mcts_save_variants": True,
    "mcts_memory_enabled": True,
    "memory_auto_update_enabled": True,
    "memory_update_max_recent_runs": 10,
    "memory_update_max_residues_per_node": 8,
    "history_size": 50,
    "progen_weight": 1.0,
    "sequence_prior_model": "progen",
    "external_kb_enabled": True,
    "external_kb_path": "data/antibody_kb/sabdab_external_prior_cache.json",
    "external_kb_weight": 0.70,
    "external_kb_embedding_manifest": "data/antibody_kb/embedding_manifest_esm2_t6_8M_sabdab_cdr.json",
    "external_kb_retrieval_enabled": True,
    "external_kb_retrieval_top_k": 20,
    "external_kb_retrieval_weight": 0.60,
    "external_kb_device": "auto",
    "external_kb_max_length": 128,
    "structure_model": "protenix",
    "structure_model_name": None,
    "protenix_model_name": "protenix_mini_esm_v0.5.0",
    "protenix_conda_env": "pytorch",
    "protenix_seed": 101,
    "protenix_complex_use_msa": None,
    "protenix_complex_cycle": None,
    "protenix_complex_step": None,
    "protenix_complex_sample": None,
    "protenix_complex_use_default_params": None,
    "protenix_complex_timeout": None,
    "esmfold2_mode": "local",
    "esmfold2_conda_env": None,
    "esmfold2_num_loops": 3,
    "esmfold2_num_sampling_steps": 32,
    "esmfold2_num_diffusion_samples": 1,
    "multistate_objectives_enabled": True,
    "multistate_objective_weight": 1.0,
    "chai1_enabled": True,
    "chai1_top_frac": 0.01,
    "chai1_min_candidates": 1,
    "chai1_max_candidates": 3,
    "max_hydrophobic_run": 2,
    "max_charged_run": 2,
    "secondary_structure_weight": 0.35,
    "linker_gs_min": 0.60,
    "linker_hydrophobic_max": 0.15,
    "linker_charged_max": 0.20,
    "cdr_favored_residues": ["Y", "W", "H", "N", "Q", "S", "T", "R", "D", "E"],
    "cdr_hydrophobic_max": 0.40,
    "cdr_charged_max": 0.45,
    "desired_cdr3_hydro": 0.30,
    "score_config": {
        "weight_fast": 1.0,
        "weight_plddt": 5.0,
        "weight_iptm": 0.5,
        "weight_ptm": 0.5,
        "weight_ranking_score": 0.0,
        "weight_interface_plddt": 1.0,
        "weight_node_plddt_min": 0.5,
        "weight_clash": 1.0,
        "weight_multistate": 2.0,
        "plddt_scale": 100.0,
        "clash_scale": 10.0,
        "fast_loss_nonneg": True,
    },
}


def _apply_locked_runtime_defaults(strategy):
    merged = _base_runtime_strategy()
    if isinstance(strategy, dict):
        merged.update(strategy)
    merged.update(deepcopy(_LOCKED_RUNTIME_DEFAULTS))
    return merged


def _env_bool(name: str, default: bool) -> bool:
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    import os

    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _apply_runtime_overrides(strategy):
    import os

    if "ASTEVOLVE_INNER_ITERATIONS" in os.environ:
        strategy["iterations"] = _env_int("ASTEVOLVE_INNER_ITERATIONS", strategy.get("iterations", 1200))
    if "ASTEVOLVE_ENABLE_PROTENIX" in os.environ:
        strategy["chai1_enabled"] = _env_bool("ASTEVOLVE_ENABLE_PROTENIX", strategy.get("chai1_enabled", True))
    if "ASTEVOLVE_ENABLE_EXTERNAL_KB" in os.environ:
        strategy["external_kb_enabled"] = _env_bool(
            "ASTEVOLVE_ENABLE_EXTERNAL_KB",
            strategy.get("external_kb_enabled", True),
        )
    if "ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL" in os.environ:
        strategy["external_kb_retrieval_enabled"] = _env_bool(
            "ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL",
            strategy.get("external_kb_retrieval_enabled", True),
        )
    if "ASTEVOLVE_PROGEN_WEIGHT" in os.environ:
        strategy["progen_weight"] = _env_float("ASTEVOLVE_PROGEN_WEIGHT", strategy.get("progen_weight", 1.0))
    if "ASTEVOLVE_SEQUENCE_PRIOR_MODEL" in os.environ:
        strategy["sequence_prior_model"] = os.environ["ASTEVOLVE_SEQUENCE_PRIOR_MODEL"]
    if "ASTEVOLVE_EXTERNAL_KB_PATH" in os.environ:
        strategy["external_kb_path"] = os.environ["ASTEVOLVE_EXTERNAL_KB_PATH"]
    if "ASTEVOLVE_EXTERNAL_KB_EMBEDDING_MANIFEST" in os.environ:
        strategy["external_kb_embedding_manifest"] = os.environ["ASTEVOLVE_EXTERNAL_KB_EMBEDDING_MANIFEST"]
    if "ASTEVOLVE_STRUCTURE_MODEL" in os.environ:
        strategy["structure_model"] = os.environ["ASTEVOLVE_STRUCTURE_MODEL"]
    if "ASTEVOLVE_STRUCTURE_MODEL_NAME" in os.environ:
        strategy["structure_model_name"] = os.environ["ASTEVOLVE_STRUCTURE_MODEL_NAME"]
    protenix_conda_env = os.environ.get("ASTEVOLVE_PROTENIX_CONDA_ENV") or os.environ.get("ASTEVOLVE_CONDA_ENV")
    if protenix_conda_env:
        strategy["protenix_conda_env"] = protenix_conda_env
    if os.environ.get("ASTEVOLVE_PROTENIX_MODEL_NAME"):
        strategy["protenix_model_name"] = os.environ["ASTEVOLVE_PROTENIX_MODEL_NAME"]
    if "ASTEVOLVE_ESMFOLD2_MODE" in os.environ:
        strategy["esmfold2_mode"] = os.environ["ASTEVOLVE_ESMFOLD2_MODE"]
    if "ASTEVOLVE_ESMFOLD2_CONDA_ENV" in os.environ:
        strategy["esmfold2_conda_env"] = os.environ["ASTEVOLVE_ESMFOLD2_CONDA_ENV"]
    if "ASTEVOLVE_ESMFOLD2_NUM_LOOPS" in os.environ:
        strategy["esmfold2_num_loops"] = _env_int(
            "ASTEVOLVE_ESMFOLD2_NUM_LOOPS",
            strategy.get("esmfold2_num_loops", 3),
        )
    if "ASTEVOLVE_ESMFOLD2_NUM_SAMPLING_STEPS" in os.environ:
        strategy["esmfold2_num_sampling_steps"] = _env_int(
            "ASTEVOLVE_ESMFOLD2_NUM_SAMPLING_STEPS",
            strategy.get("esmfold2_num_sampling_steps", 32),
        )
    if "ASTEVOLVE_ESMFOLD2_NUM_DIFFUSION_SAMPLES" in os.environ:
        strategy["esmfold2_num_diffusion_samples"] = _env_int(
            "ASTEVOLVE_ESMFOLD2_NUM_DIFFUSION_SAMPLES",
            strategy.get("esmfold2_num_diffusion_samples", 1),
        )
    if "ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA" in os.environ:
        strategy["protenix_complex_use_msa"] = _env_bool(
            "ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA",
            strategy.get("protenix_complex_use_msa", False),
        )
    if "ASTEVOLVE_PROTENIX_COMPLEX_CYCLE" in os.environ:
        strategy["protenix_complex_cycle"] = _env_int("ASTEVOLVE_PROTENIX_COMPLEX_CYCLE", strategy.get("protenix_complex_cycle", 1))
    if "ASTEVOLVE_PROTENIX_COMPLEX_STEP" in os.environ:
        strategy["protenix_complex_step"] = _env_int("ASTEVOLVE_PROTENIX_COMPLEX_STEP", strategy.get("protenix_complex_step", 1))
    if "ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE" in os.environ:
        strategy["protenix_complex_sample"] = _env_int("ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE", strategy.get("protenix_complex_sample", 1))
    if "ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS" in os.environ:
        strategy["protenix_complex_use_default_params"] = _env_bool(
            "ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS",
            strategy.get("protenix_complex_use_default_params", False),
        )
    if "ASTEVOLVE_PROTENIX_COMPLEX_TIMEOUT" in os.environ:
        strategy["protenix_complex_timeout"] = _env_int("ASTEVOLVE_PROTENIX_COMPLEX_TIMEOUT", strategy.get("protenix_complex_timeout", 240))
    if "ASTEVOLVE_MCTS_OUTPUT_DIR" in os.environ:
        strategy["mcts_output_dir"] = os.environ["ASTEVOLVE_MCTS_OUTPUT_DIR"]
    if "ASTEVOLVE_SAVE_MCTS_TREE" in os.environ:
        strategy["mcts_save_tree"] = _env_bool("ASTEVOLVE_SAVE_MCTS_TREE", strategy.get("mcts_save_tree", True))
    if "ASTEVOLVE_SAVE_VARIANTS" in os.environ:
        strategy["mcts_save_variants"] = _env_bool(
            "ASTEVOLVE_SAVE_VARIANTS",
            strategy.get("mcts_save_variants", True),
        )
    if "ASTEVOLVE_MCTS_MEMORY_ENABLED" in os.environ:
        strategy["mcts_memory_enabled"] = _env_bool(
            "ASTEVOLVE_MCTS_MEMORY_ENABLED",
            strategy.get("mcts_memory_enabled", True),
        )
    if "ASTEVOLVE_MEMORY_AUTO_UPDATE" in os.environ:
        strategy["memory_auto_update_enabled"] = _env_bool(
            "ASTEVOLVE_MEMORY_AUTO_UPDATE",
            strategy.get("memory_auto_update_enabled", True),
        )
    return strategy


def _runtime_strategy():
    return _apply_runtime_overrides(_apply_locked_runtime_defaults(propose_strategy()))


def _current_case_kwargs():
    return current_case_kwargs("cd25_scfv")


def run_search(seed: Optional[int] = None):
    return run_design_search(
        _runtime_strategy(),
        seed=seed,
        **_current_case_kwargs(),
    )


def preview_case():
    bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg, state = build_case_inputs(
        _runtime_strategy(),
        **_current_case_kwargs(),
    )
    compiled = bp.compile()
    return {
        "task_name": state["task_name"],
        "chain_lengths": compiled["chain_lengths"],
        "chain_order": compiled["chain_order"],
        "binder_domain_order": state["binder"].get("domain_order", ["VH_domain", "Linker_module", "VL_domain"]),
        "layout_summary": state.get("_layout_summary", {}),
        "strategy_schema_report": state.get("_strategy_schema_report", {}),
        "node_policy_names": list(state.get("_node_edit_policies", {}).keys()),
        "locked_runtime_fields": sorted(_LOCKED_RUNTIME_DEFAULTS),
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

