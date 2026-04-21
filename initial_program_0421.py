# EVOLVE-BLOCK-START
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from protein_lang import Node, Blueprint


# ============================================================
# Fixed case definition: CD25 scFv (VH-Linker-VL) + CD25 target
# ============================================================

VH_SEQ = (
    "QVQLVQSGAEVKKPGSSVKVS"
    "CKASGYTFTSYRMH"
    "WVRQAPGQGLEWI"
    "GYINPSTGYTEYNQKF"
    "KDKATITADESTNTAYMELSSLRSEDTAVYYCA"
    "RGGGVFDY"
    "WGQGTLVTVS"
)

LINKER_SEQ = "GGGGSGGGGSGGGGS"

VL_SEQ = (
    "DIQMTQSPSTLSASVGDRVTITC"
    "SASSSISYMH"
    "WYQQKPGKAPKLLIY"
    "TTSNLAS"
    "GVPARFSGSGSGTEFTLTISSLQPDDFATYYC"
    "HQRSTYPLT"
    "FGQGTKVEVKRTVA"
)

TARGET_SEQ = (
    "ELCDDDPPEIPHATFKAMAYKEGTMLNCECKRGFRRIKSGSLYMLCTGNSSHSSWDNQCQCTSSATRN"
    "TTKQVTPQPEEQKERKTTEMQSPMQPVDQASLPGHCREPPPWENEATERIYHFVVGQMVYYQCVQGYRA"
    "LHRGPAESVCKMTHGKTRWTQPQLICTGEMETSQFPGEEKPQASPEGRPESETSCLVTTTDFQIQTEMA"
    "ATMETSIFTTE"
)


# ============================================================
# Optional memory.yaml reading
# ============================================================

def _safe_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        return None


def _load_memory_yaml(memory_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Soft dependency:
    - If yaml package or file is missing, return {} and continue.
    - This keeps initial_program runnable even before memory pipeline is finalized.
    """
    yaml = _safe_import_yaml()
    if yaml is None:
        return {}

    candidates: List[Path] = []
    if memory_path:
        candidates.append(Path(memory_path))
    candidates.extend([
        Path("memory.yaml"),
        Path(__file__).with_name("memory.yaml"),
    ])

    for p in candidates:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


# ============================================================
# Sequence segmentation helpers
# ============================================================

def _ordered_segments_for_binder() -> List[Tuple[str, str]]:
    """
    Returns ordered leaf segment names and sequences for chain BB.
    """
    return [
        ("VH_FR1", "QVQLVQSGAEVKKPGSSVKVS"),
        ("VH_CDR1", "CKASGYTFTSYRMH"),
        ("VH_FR2", "WVRQAPGQGLEWI"),
        ("VH_CDR2", "GYINPSTGYTEYNQKF"),
        ("VH_FR3", "KDKATITADESTNTAYMELSSLRSEDTAVYYCA"),
        ("VH_CDR3", "RGGGVFDY"),
        ("VH_FR4", "WGQGTLVTVS"),
        ("Linker_head", LINKER_SEQ[:5]),
        ("Linker_core", LINKER_SEQ[5:10]),
        ("Linker_tail", LINKER_SEQ[10:]),
        ("VL_FR1", "DIQMTQSPSTLSASVGDRVTITC"),
        ("VL_CDR1", "SASSSISYMH"),
        ("VL_FR2", "WYQQKPGKAPKLLIY"),
        ("VL_CDR2", "TTSNLAS"),
        ("VL_FR3", "GVPARFSGSGSGTEFTLTISSLQPDDFATYYC"),
        ("VL_CDR3", "HQRSTYPLT"),
        ("VL_FR4", "FGQGTKVEVKRTVA"),
    ]


def _segment_spans_from_parts(parts: List[Tuple[str, str]]) -> Dict[str, Tuple[int, int]]:
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name, seq in parts:
        spans[name] = (cursor, cursor + len(seq))
        cursor += len(seq)
    return spans


def _find_target_epitope_window(
    seq: str,
    center_hint: Optional[str] = None,
    default_start: int = 120,
    default_len: int = 40,
) -> Tuple[int, int]:
    """
    Current target is a single CD25 sequence. Since no experimentally confirmed epitope
    is wired in yet, we create a coarse candidate window.
    Priority:
    1. memory hint if provided
    2. motif string if found
    3. middle-biased default window
    """
    if center_hint:
        idx = seq.find(center_hint)
        if idx >= 0:
            start = max(0, idx - 10)
            end = min(len(seq), idx + len(center_hint) + 15)
            return start, end

    start = max(0, min(default_start, max(0, len(seq) - default_len)))
    end = min(len(seq), start + default_len)
    return start, end


# ============================================================
# Memory-derived soft priors
# ============================================================

def _extract_memory_bias(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull only soft biases needed by initial_program.
    No hard dependency on exact memory schema.
    """
    out: Dict[str, Any] = {
        "preferred_edit_order": [
            "VH_CDR3", "VH_CDR2", "VL_CDR3", "VH_CDR1", "VL_CDR1", "VL_CDR2",
            "Linker_core", "VH_FR2", "VH_FR3", "VL_FR2", "VL_FR3"
        ],
        "global_allowed_new_positions": list("ADEGHKNPQRSTY"),
        "global_disfavored_new_positions": ["C"],
        "linker_default_sequence": LINKER_SEQ,
        "linker_gs_min": 0.60,
        "linker_hydrophobic_max": 0.15,
        "linker_charged_max": 0.20,
        "cdr_favored_sparse": ["Y", "W", "H", "N", "Q", "S", "T", "R", "D", "E"],
    }

    try:
        stable = memory.get("stable_priors", {})
        scfv_prior = stable.get("scfv_prior", {})
        aa_prior = stable.get("aa_prior", {})
        linker_prior = stable.get("linker_prior", {})

        general = scfv_prior.get("general", {})
        if isinstance(general.get("preferred_edit_order"), list):
            out["preferred_edit_order"] = general["preferred_edit_order"]

        subprefs = aa_prior.get("substitution_preferences", {})
        if isinstance(subprefs.get("global_default_allowed_new_positions"), list):
            out["global_allowed_new_positions"] = subprefs["global_default_allowed_new_positions"]
        if isinstance(subprefs.get("globally_disfavored_new_positions"), list):
            out["global_disfavored_new_positions"] = subprefs["globally_disfavored_new_positions"]

        if isinstance(linker_prior.get("default_sequence"), str):
            out["linker_default_sequence"] = linker_prior["default_sequence"]

        comp_targets = linker_prior.get("composition_targets", {})
        if "gly_ser_fraction_preferred_min" in comp_targets:
            out["linker_gs_min"] = float(comp_targets["gly_ser_fraction_preferred_min"])
        if "hydrophobic_fraction_preferred_max" in comp_targets:
            out["linker_hydrophobic_max"] = float(comp_targets["hydrophobic_fraction_preferred_max"])
        if "charged_fraction_preferred_max" in comp_targets:
            out["linker_charged_max"] = float(comp_targets["charged_fraction_preferred_max"])

        cdr_design = scfv_prior.get("cdr_design_bias", {})
        if isinstance(cdr_design.get("favored_residues_sparse"), list):
            out["cdr_favored_sparse"] = cdr_design["favored_residues_sparse"]

    except Exception:
        pass

    return out


# ============================================================
# AST construction
# ============================================================

def _make_binder_chain() -> Node:
    """
    Build a single-chain scFv chain BB:
    VH -> Linker -> VL
    """
    return Node(
        kind="chain",
        name="Binder_scFv",
        props={"chain_id": "BB"},
        children=[
            Node(
                kind="domain",
                name="VH_domain",
                children=[
                    Node(kind="framework", name="VH_FR1", length=21),
                    Node(kind="cdr", name="VH_CDR1", length=14),
                    Node(kind="framework", name="VH_FR2", length=13),
                    Node(kind="cdr", name="VH_CDR2", length=16),
                    Node(kind="framework", name="VH_FR3", length=32),
                    Node(kind="cdr", name="VH_CDR3", length=8),
                    Node(kind="framework", name="VH_FR4", length=10),
                ],
            ),
            Node(
                kind="linker",
                name="Linker_module",
                children=[
                    Node(kind="linker", name="Linker_head", length=5),
                    Node(kind="linker", name="Linker_core", length=5),
                    Node(kind="linker", name="Linker_tail", length=5),
                ],
            ),
            Node(
                kind="domain",
                name="VL_domain",
                children=[
                    Node(kind="framework", name="VL_FR1", length=23),
                    Node(kind="cdr", name="VL_CDR1", length=10),
                    Node(kind="framework", name="VL_FR2", length=15),
                    Node(kind="cdr", name="VL_CDR2", length=7),
                    Node(kind="framework", name="VL_FR3", length=32),
                    Node(kind="cdr", name="VL_CDR3", length=9),
                    Node(kind="framework", name="VL_FR4", length=14),
                ],
            ),
        ],
    )


def _make_target_chain(ep_start: int, ep_end: int) -> Node:
    children: List[Node] = []
    if ep_start > 0:
        children.append(Node(kind="domain", name="CD25_global", length=ep_start))
    children.append(Node(kind="epitope", name="CD25_epitope_candidate", length=ep_end - ep_start))
    if ep_end < len(TARGET_SEQ):
        children.append(Node(kind="domain", name="CD25_non_epitope", length=len(TARGET_SEQ) - ep_end))

    return Node(
        kind="chain",
        name="Target_CD25",
        props={"chain_id": "T"},
        children=children,
    )


# ============================================================
# Mutation mask construction
# ============================================================

def _mask_from_spans(length: int, allowed_spans: List[Tuple[int, int]]) -> List[bool]:
    mask = [False] * length
    for s, e in allowed_spans:
        for i in range(max(0, s), min(length, e)):
            mask[i] = True
    return mask


def _build_binder_mask(
    binder_len: int,
    segment_spans: Dict[str, Tuple[int, int]],
    preferred_edit_order: List[str],
) -> List[bool]:
    """
    Initial policy:
    - open all CDRs
    - open linker_core only
    - open limited supporting FR nodes (VH_FR2/VH_FR3/VL_FR2/VL_FR3)
    - keep FR1/FR4/linker junctions closed by default
    """
    always_open = {
        "VH_CDR1", "VH_CDR2", "VH_CDR3",
        "VL_CDR1", "VL_CDR2", "VL_CDR3",
    }

    conditionally_open = {
        "Linker_core",
        "VH_FR2", "VH_FR3",
        "VL_FR2", "VL_FR3",
    }

    selected: List[Tuple[int, int]] = []

    # Prefer memory order if available, but still keep a hard whitelist.
    for name in preferred_edit_order:
        if name in always_open or name in conditionally_open:
            if name in segment_spans:
                selected.append(segment_spans[name])

    # Ensure all main CDRs are open even if memory schema changes.
    for name in sorted(always_open):
        if name in segment_spans and segment_spans[name] not in selected:
            selected.append(segment_spans[name])

    return _mask_from_spans(binder_len, selected)


def _fixed_residues_for_protection(
    segment_spans: Dict[str, Tuple[int, int]],
    linker_seq: str,
) -> Dict[str, Dict[int, str]]:
    """
    Hard-fix only a small set of positions that we really do not want to drift
    in the initial stage. The rest is controlled via masks.
    Here we keep linker head/tail junctions fixed and leave linker core mutable.
    """
    fixed: Dict[str, Dict[int, str]] = {"BB": {}, "T": {}}

    # Fix linker head and tail to keep architecture stable initially.
    lh_s, lh_e = segment_spans["Linker_head"]
    lc_s, lc_e = segment_spans["Linker_core"]
    lt_s, lt_e = segment_spans["Linker_tail"]

    full_linker = linker_seq
    linker_head = full_linker[:5]
    linker_tail = full_linker[10:]

    for i, aa in enumerate(linker_head):
        fixed["BB"][lh_s + i] = aa
    for i, aa in enumerate(linker_tail):
        fixed["BB"][lt_s + i] = aa

    # Fix full target sequence.
    for i, aa in enumerate(TARGET_SEQ):
        fixed["T"][i] = aa

    return fixed


# ============================================================
# Constraint construction
# ============================================================

def _build_constraint_specs(
    memory_bias: Dict[str, Any],
    epitope_name: str = "CD25_epitope_candidate",
) -> List[Dict[str, Any]]:
    """
    Keep constraint kinds inside the current allowed list:
    alphabet / fixed_chain_sequence / fixed_residues / ss_proxy /
    hydrophobic_pattern / interface_proxy / segment_composition / max_run
    """
    no_cys = set("ADEFGHIKLMNPQRSTVWY")

    linker_gs_min = float(memory_bias.get("linker_gs_min", 0.60))
    linker_hydrophobic_max = float(memory_bias.get("linker_hydrophobic_max", 0.15))
    linker_charged_max = float(memory_bias.get("linker_charged_max", 0.20))

    cdr_favored = set(memory_bias.get("cdr_favored_sparse", ["Y", "W", "H", "N", "Q", "S", "T", "R", "D", "E"]))

    specs: List[Dict[str, Any]] = [
        # Basic alphabet and full target fixation
        {"kind": "alphabet", "weight": 1.0, "params": {"allowed": no_cys}},
        {"kind": "fixed_chain_sequence", "weight": 1.0, "params": {"chain_id": "T", "sequence": TARGET_SEQ}},

        # Global hydrophobic pattern:
        # domains keep some compactness, linker stays nonsticky
        {"kind": "hydrophobic_pattern", "weight": 1.2, "params": {
            "domain_min_hydro": 0.22,
            "linker_max_hydro": linker_hydrophobic_max,
        }},

        # Linker flexibility / nonstickiness
        {"kind": "segment_composition", "weight": 2.0, "params": {
            "aa_set": "GSAT",
            "min_frac": linker_gs_min,
            "max_frac": 1.0,
            "segment_filter": {"chain_id": "BB", "kind": "linker"},
        }},
        {"kind": "segment_composition", "weight": 1.3, "params": {
            "aa_set": "AILMFWVY",
            "min_frac": 0.0,
            "max_frac": linker_hydrophobic_max,
            "segment_filter": {"chain_id": "BB", "kind": "linker"},
        }},
        {"kind": "segment_composition", "weight": 1.0, "params": {
            "aa_set": "KRDE",
            "min_frac": 0.0,
            "max_frac": linker_charged_max,
            "segment_filter": {"chain_id": "BB", "kind": "linker"},
        }},

        # Avoid long problematic runs globally on binder
        {"kind": "max_run", "weight": 1.2, "params": {
            "aa_set": "AILMFWVY",
            "max_run": 2,
            "segment_filter": {"chain_id": "BB"},
        }},
        {"kind": "max_run", "weight": 1.2, "params": {
            "aa_set": "KRDE",
            "max_run": 2,
            "segment_filter": {"chain_id": "BB"},
        }},

        # Encourage interface-capable chemistry in CDRs, but not too hydrophobic
        {"kind": "segment_composition", "weight": 1.2, "params": {
            "aa_set": "".join(sorted(cdr_favored)),
            "min_frac": 0.20,
            "max_frac": 0.85,
            "segment_filter": {"chain_id": "BB", "kind": "cdr"},
        }},
        {"kind": "segment_composition", "weight": 1.0, "params": {
            "aa_set": "AILMFWVY",
            "min_frac": 0.05,
            "max_frac": 0.40,
            "segment_filter": {"chain_id": "BB", "kind": "cdr"},
        }},
        {"kind": "segment_composition", "weight": 0.8, "params": {
            "aa_set": "KRDE",
            "min_frac": 0.05,
            "max_frac": 0.45,
            "segment_filter": {"chain_id": "BB", "kind": "cdr"},
        }},

        # Framework should stay more conservative and not become sticky
        {"kind": "segment_composition", "weight": 1.0, "params": {
            "aa_set": "AILMFWVY",
            "min_frac": 0.10,
            "max_frac": 0.38,
            "segment_filter": {"chain_id": "BB", "kind": "framework"},
        }},
        {"kind": "segment_composition", "weight": 0.8, "params": {
            "aa_set": "KRDE",
            "min_frac": 0.05,
            "max_frac": 0.35,
            "segment_filter": {"chain_id": "BB", "kind": "framework"},
        }},

        # Loose interface proxy:
        # binder side should not be completely non-hydrophobic,
        # but we deliberately keep this mild because true binding is not available here.
        {"kind": "interface_proxy", "weight": 0.8, "params": {
            "binder_chain": "BB",
            "binder_segment": "VH_CDR3",
            "target_chain": "T",
            "target_segment": epitope_name,
            "desired_binder_hydro": 0.30,
        }},
    ]

    return specs


# ============================================================
# Search config and score config
# ============================================================

def _build_sa_config(memory_bias: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep this compatible with current inner_opt.SAConfig.
    """
    preferred = memory_bias.get("preferred_edit_order", [])
    linker_bias = "Linker_core" in preferred

    return {
        "iterations": 1200,
        "init_temp": 2.0,
        "cooling": 0.995,
        "mutation_rate": 0.06,            # much lower than toy peptide case
        "resample_segment_prob": 0.08,

        "progen_weight": 1.0,
        "progen_chains": ["BB"],
        "progen_reduce": "length_weighted",

        "chai1_enabled": True,
        "chai1_top_frac": 0.01,
        "chai1_min_candidates": 1,
        "chai1_max_candidates": 3,
        "chai1_num_trunk_recycles": 3,
        "chai1_num_diffn_timesteps": 50,

        # Bias toward sparse point mutation for scFv stability
        "mutation_ops": {
            "point": 0.75,
            "block": 0.15,
            "segment_resample": 0.07 if linker_bias else 0.05,
            "swap": 0.03,
        },
        "history_size": 50,
    }


def _build_score_config() -> Dict[str, Any]:
    """
    Keep score_config stable.
    """
    return {
        "weight_fast": 1,
        "weight_plddt": 5,
        "plddt_scale": 100.0,
        "fast_loss_nonneg": True,
    }


# ============================================================
# Main blueprint/config proposal
# ============================================================

def propose_blueprint_and_config(memory_path: Optional[str] = None):
    memory = _load_memory_yaml(memory_path)
    memory_bias = _extract_memory_bias(memory)

    binder_parts = _ordered_segments_for_binder()
    binder_spans = _segment_spans_from_parts(binder_parts)

    # Coarse candidate epitope window on CD25.
    # Can be replaced later by memory-driven or structure-driven refinement.
    ep_start, ep_end = _find_target_epitope_window(
        TARGET_SEQ,
        center_hint=None,
        default_start=120,
        default_len=42,
    )

    chain_BB = _make_binder_chain()
    chain_T = _make_target_chain(ep_start, ep_end)

    bp = Blueprint(
        root=Node(
            kind="complex",
            name="CD25_scFv_Design_Task",
            children=[chain_BB, chain_T],
        )
    )

    binder_len = len(VH_SEQ + LINKER_SEQ + VL_SEQ)
    masks = {
        "BB": _build_binder_mask(
            binder_len=binder_len,
            segment_spans=binder_spans,
            preferred_edit_order=memory_bias["preferred_edit_order"],
        ),
        "T": [False] * len(TARGET_SEQ),
    }

    templates = {
        "BB": VH_SEQ + memory_bias.get("linker_default_sequence", LINKER_SEQ) + VL_SEQ,
        "T": TARGET_SEQ,
    }

    fixed_residues = _fixed_residues_for_protection(
        segment_spans=binder_spans,
        linker_seq=memory_bias.get("linker_default_sequence", LINKER_SEQ),
    )

    constraint_specs = _build_constraint_specs(
        memory_bias=memory_bias,
        epitope_name="CD25_epitope_candidate",
    )

    sa_config = _build_sa_config(memory_bias)
    score_config = _build_score_config()

    return bp, constraint_specs, sa_config, masks, templates, fixed_residues, score_config


# EVOLVE-BLOCK-END


# ============================================================
# Stable execution wrapper
# ============================================================

import numpy as np
from inner_opt import optimize_multichain, SAConfig


def run_search(seed: int | None = None, memory_path: Optional[str] = None):
    bp, constraint_specs, sa_cfg, masks, templates, fixed_residues, score_cfg = (
        propose_blueprint_and_config(memory_path=memory_path)
    )

    compiled = bp.compile()
    masks_np = {k: np.array(v, dtype=bool) for k, v in masks.items()}
    cfg = SAConfig(**sa_cfg, seed=seed)

    out = optimize_multichain(
        compiled,
        constraint_specs,
        cfg,
        masks=masks_np,
        template_seqs=templates,
        fixed_residues=fixed_residues,
    )

    out["chain_lengths"] = compiled["chain_lengths"]
    out["segments"] = [
        {
            "chain_id": s.chain_id,
            "kind": s.kind,
            "name": s.name,
            "spans": s.spans,
            "total_length": s.total_length,
            "is_contiguous": s.is_contiguous,
            "start": s.start,
            "end": s.end,
            "props": s.props,
        }
        for s in compiled["segments"]
    ]
    out["blueprint_summary"] = {
        "task_name": "CD25_scFv_Design_Task",
        "chain_order": compiled["chain_order"],
        "chain_lengths": compiled["chain_lengths"],
        "binder_architecture": "VH-Linker-VL",
        "target_name": "CD25",
    }
    out["score_config"] = score_cfg
    return out


if __name__ == "__main__":
    r = run_search(seed=0)
    print("Fast loss:", r["fast_loss"])
    print("Chains:", r.get("chain_lengths"))
    print("Best seqs keys:", list(r.get("seqs", {}).keys()))