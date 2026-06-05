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
    """OpenEvolve edits the TetR layout DSL, not runtime or solver settings."""
    strategy = base_strategy()

    strategy["layout_plan"] = {
        "binder_domain_order": ["DNA_binding_domain", "TED_ligand_dimer_domain"],
        "secondary_structure_priors": {
            "HTH_helix1": "helix",
            "HTH_recognition_helix": "helix",
            "HTH_exit_loop": "loop",
            "domain_connector": "loop",
            "allosteric_hinge": "loop",
            "relay_helix": "helix",
            "dimerization_interface": "helix",
        },
        "design_regions": [
            {
                "name": "dopamine_pocket_specificity",
                "role": "reshape TetR ligand pocket for dopamine catechol and protonatable amine recognition",
                "position": 1,
                "bind_to": ["ligand_pocket_core"],
                "secondary_structure": "helix",
                "priority_boost": 1.45,
                "length_mutable": False,
                "mutation_rate": 0.085,
                "max_mutations_per_step": 4,
                "mutation_ops": {"point": 0.70, "block": 0.22, "segment_resample": 0.04, "swap": 0.04},
                "site_anchors": {
                    "ligand_pocket_core": {
                        "relative_ranges": [[4, 18], [24, 37]],
                        "weight": 2.4,
                        "favored_residues": ["D", "E", "Y", "H", "S", "T", "N", "Q", "W", "F"],
                        "favored_residue_classes": ["polar_uncharged", "aromatic", "contextual_charge"]
                    }
                },
                "favored_residues": ["D", "E", "Y", "H", "S", "T", "N", "Q", "W", "F", "G"],
                "disfavored_residues": ["C"],
                "policy_weight": 0.95
            },
            {
                "name": "allosteric_relay_coupling",
                "role": "tune hinge and relay helix so dopamine-pocket changes can propagate toward the DNA-binding domain",
                "position": 2,
                "bind_to": ["allosteric_hinge", "relay_helix", "domain_connector", "HTH_exit_loop"],
                "secondary_structure": "loop",
                "priority_boost": 1.15,
                "length_mutable": False,
                "mutation_rate": 0.055,
                "max_mutations_per_step": 3,
                "mutation_ops": {"point": 0.78, "block": 0.16, "segment_resample": 0.02, "swap": 0.04},
                "favored_residues": ["G", "S", "T", "N", "Q", "A", "D", "E", "K", "R"],
                "disfavored_residues": ["C"],
                "policy_weight": 0.80
            },
            {
                "name": "tetO_recognition_guardrail",
                "role": "allow sparse DNA-recognition tuning without destroying HTH geometry",
                "position": 3,
                "bind_to": ["HTH_recognition_helix"],
                "secondary_structure": "helix",
                "priority_boost": 0.55,
                "length_mutable": False,
                "mutation_rate": 0.018,
                "max_mutations_per_step": 1,
                "mutation_ops": {"point": 0.96, "block": 0.02, "segment_resample": 0.0, "swap": 0.02},
                "favored_residues": ["K", "R", "Q", "N", "S", "T", "Y", "H"],
                "disfavored_residues": ["C", "P"],
                "policy_weight": 0.35
            },
            {
                "name": "dimer_interface_support",
                "role": "maintain homodimer packing while pocket mutations are explored",
                "position": 4,
                "bind_to": ["dimerization_interface"],
                "secondary_structure": "helix",
                "priority_boost": 0.50,
                "length_mutable": False,
                "mutation_rate": 0.018,
                "max_mutations_per_step": 1,
                "mutation_ops": {"point": 0.94, "block": 0.03, "segment_resample": 0.0, "swap": 0.03},
                "favored_residues": ["A", "L", "I", "V", "S", "T", "Q", "N", "E", "K"],
                "disfavored_residues": ["C", "P", "W"],
                "policy_weight": 0.35
            }
        ]
    }
    return strategy


# EVOLVE-BLOCK-END

from copy import deepcopy
from typing import Any, Dict, Optional

from astevolve.runtime.case_context import current_case_kwargs
from engine.case_builder import build_case_inputs, run_design_search
from engine.default_strategy import base_strategy as _base_runtime_strategy


_LOCKED_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "iterations": 240,
    "init_temp": 2.0,
    "cooling": 0.995,
    "mutation_rate": 0.055,
    "resample_segment_prob": 0.08,
    "mutation_ops": {
        "point": 0.75,
        "block": 0.15,
        "segment_resample": 0.07,
        "swap": 0.03,
    },
    "search_method": "mcts",
    "mcts_c_puct": 1.25,
    "mcts_max_depth": 4,
    "mcts_reward_scale": 1.0,
    "mcts_output_dir": "artifacts/tetr_dopamine/inner_loop_runs/latest",
    "mcts_save_tree": True,
    "mcts_save_variants": True,
    "mcts_memory_enabled": True,
    "memory_auto_update_enabled": True,
    "memory_update_max_recent_runs": 10,
    "memory_update_max_residues_per_node": 8,
    "history_size": 50,
    "progen_weight": 0.5,
    "sequence_prior_model": "progen",
    "external_kb_enabled": True,
    "external_kb_path": "data/atf_kb/atf_interpro_external_prior_cache.json",
    "external_kb_weight": 0.40,
    "external_kb_embedding_manifest": "data/atf_kb/embedding_manifest_esm2_t6_8M_atf_interpro.json",
    "external_kb_retrieval_enabled": False,
    "external_kb_retrieval_top_k": 12,
    "external_kb_retrieval_weight": 0.25,
    "external_kb_device": "auto",
    "external_kb_max_length": 128,
    "structure_model": "protenix",
    "structure_model_name": None,
    "protenix_model_name": "protenix_mini_esm_v0.5.0",
    "protenix_conda_env": "pytorch",
    "protenix_seed": 101,
    "protenix_complex_use_msa": False,
    "protenix_complex_cycle": 1,
    "protenix_complex_step": 1,
    "protenix_complex_sample": 1,
    "protenix_complex_use_default_params": False,
    "protenix_complex_timeout": 240,
    "esmfold2_mode": "local",
    "esmfold2_conda_env": None,
    "esmfold2_num_loops": 3,
    "esmfold2_num_sampling_steps": 32,
    "esmfold2_num_diffusion_samples": 1,
    "multistate_objectives_enabled": True,
    "multistate_objective_weight": 1.0,
    "chai1_enabled": True,
    "chai1_top_frac": 0.02,
    "chai1_min_candidates": 1,
    "chai1_max_candidates": 2,
    "max_hydrophobic_run": 2,
    "max_charged_run": 2,
    "secondary_structure_weight": 0.35,
    "pocket_hydrophobic_max": 0.55,
    "score_config": {
        "weight_fast": 1.0,
        "weight_plddt": 2.0,
        "weight_iptm": 0.75,
        "weight_ptm": 1.0,
        "weight_interface_plddt": 1.0,
        "weight_node_plddt_min": 0.5,
        "weight_clash": 1.5,
        "weight_multistate": 2.5,
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
        strategy["iterations"] = _env_int("ASTEVOLVE_INNER_ITERATIONS", strategy.get("iterations", 240))
    if "ASTEVOLVE_ENABLE_PROTENIX" in os.environ:
        strategy["chai1_enabled"] = _env_bool("ASTEVOLVE_ENABLE_PROTENIX", strategy.get("chai1_enabled", True))
    if "ASTEVOLVE_PROGEN_WEIGHT" in os.environ:
        strategy["progen_weight"] = _env_float("ASTEVOLVE_PROGEN_WEIGHT", strategy.get("progen_weight", 0.5))
    if "ASTEVOLVE_ENABLE_EXTERNAL_KB" in os.environ:
        strategy["external_kb_enabled"] = _env_bool(
            "ASTEVOLVE_ENABLE_EXTERNAL_KB",
            strategy.get("external_kb_enabled", True),
        )
    if "ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL" in os.environ:
        strategy["external_kb_retrieval_enabled"] = _env_bool(
            "ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL",
            strategy.get("external_kb_retrieval_enabled", False),
        )
    if "ASTEVOLVE_EXTERNAL_KB_PATH" in os.environ:
        strategy["external_kb_path"] = os.environ["ASTEVOLVE_EXTERNAL_KB_PATH"]
    if "ASTEVOLVE_EXTERNAL_KB_EMBEDDING_MANIFEST" in os.environ:
        strategy["external_kb_embedding_manifest"] = os.environ["ASTEVOLVE_EXTERNAL_KB_EMBEDDING_MANIFEST"]
    protenix_conda_env = os.environ.get("ASTEVOLVE_PROTENIX_CONDA_ENV") or os.environ.get("ASTEVOLVE_CONDA_ENV")
    if protenix_conda_env:
        strategy["protenix_conda_env"] = protenix_conda_env
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
    return strategy


def _runtime_strategy():
    return _apply_runtime_overrides(_apply_locked_runtime_defaults(propose_strategy()))


def run_search(seed: Optional[int] = None):
    return run_design_search(
        _runtime_strategy(),
        seed=seed,
        **current_case_kwargs("tetr_dopamine"),
    )


def preview_case():
    bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg, state = build_case_inputs(
        _runtime_strategy(),
        **current_case_kwargs("tetr_dopamine"),
    )
    compiled = bp.compile()
    return {
        "task_name": state["task_name"],
        "chain_lengths": compiled["chain_lengths"],
        "chain_order": compiled["chain_order"],
        "binder_domain_order": state["binder"].get("domain_order", []),
        "layout_summary": state.get("_layout_summary", {}),
        "strategy_schema_report": state.get("_strategy_schema_report", {}),
        "node_policy_names": list(state.get("_node_edit_policies", {}).keys()),
        "locked_runtime_fields": sorted(_LOCKED_RUNTIME_DEFAULTS),
        "complex_states": [s.get("name") for s in state.get("complex_states", [])],
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
        "constraint_specs": constraint_specs,
        "sa_config": sa_cfg,
        "score_config": score_cfg,
        "mask_true_counts": {k: int(sum(v)) for k, v in masks.items()},
        "template_lengths": {k: len(v) for k, v in templates.items()},
        "fixed_residue_counts": {k: len(v) for k, v in fixed_residues.items()},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(preview_case(), ensure_ascii=False, indent=2, default=str))
