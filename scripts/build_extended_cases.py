from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = PROJECT_ROOT / "cases"


PDL1_ECD = (
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYNKINQRILVVDPVTSEHELTCQAEGYPKAEVIWTSSDHQVLSGKTTTTNSKREEKLFNVTSTLRINTTTNEIFYCTFRRLDPEENHTAELVIPELPLAHPPNER"
)
PDL2_ECD_PROXY = (
    "LPDLKAFPVEKINLSDYHYNSRLRVVLNTQGITLQCYGSVLLSNLQLGVPQELDLSKIKNQLTQLVDPNVTCQLKFRFYQDQKTTLPDCTTWEKILEVHGVGYNVTYRQTNTLEVILTSGQVILQAGERVHFSCSVMHEALHNHYTQKSLSLSPGK"
)
PD1_ECD = (
    "NPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDSRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAE"
)
IL2RB_ECD_PROXY = (
    "AVNGTSQFTCFYNSRANISCVWSQDGALQDTSCQVHAWPDRRRWNQTCELLPVSQASWACNLILGAPDSQKLTTVDIVTLRVLCREGVRWRVMAIQDFKPFENLRLMAPISLQVVHVETHRCNISWEISQASHYFERHLEFEARTLSPGHTWEEAPLLTLKQKQEWICLETLTPDTQYEFQVRVKPLQGEFTTWSPWSQPLAFRTKPAALGKDTIPWLGHLLVGLSGAFGFIILVYLLINCRNTGPWLKKVLKCNTPDPSKFFSQLSSEHGGDVQKWLSSPFPSSSFSPG"
)
PDZ_SEQ = (
    "GSPEFLGEEDIPREPRRIVIHRGSTGLGFNIIGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEANSRVNSSGRIVTN"
)
CALMODULIN_SEQ = (
    "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK"
)
MLCK_PEPTIDE = "KRRWKKNFIAVSAANRFKKISSSGAL"


PDL1_AVELUMAB_EPITOPE = [[36, 37], [38, 39], [40, 46], [48, 49], [51, 52], [55, 56], [57, 59], [60, 61], [95, 96], [97, 98], [99, 100], [103, 104], [105, 106], [107, 108]]
PDL1_PD1_INTERFACE = [[1, 2], [5, 6], [8, 9], [36, 37], [38, 39], [40, 41], [48, 49], [58, 59], [95, 96], [97, 98], [99, 100], [101, 108]]
PDZ_GROOVE_SPANS = [[21, 22], [25, 33], [34, 35], [42, 43], [75, 76], [79, 80], [82, 84]]
CALMODULIN_EF_SPANS = [[19, 31], [56, 68], [92, 104], [129, 141]]


PROTEOR1_H = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYDMHWVRQAPGKGLEWVAVIWRDGSNEYYADSVKGRFIISRDNSKNTLYLQMNSLRAEDTAVYYCARRGIIMVRGLLGYWGQGTLVTVSS"
PROTEOR1_L = "DIQMTQSPPSLSASVGDRVTITCRASQGISNYLAWHQQKPGKVPKLLIYTASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQKYNSAPFTFGPGTKVDI"
PROTEOR1_A = "LEVVQLNISAHMDFGEARLDSVTINGNTSYCVTKPYFRLETNFMCTGCTMNLRTDTCSFDLSAVNNGMSFSQFCLSTESGACEMKIIVTYVWNYLLRQRLYVTAVEGQTHTGTT"
PROTEOR1_H_SPANS = [[25, 32], [51, 57], [98, 110]]
PROTEOR1_L_SPANS = [[23, 34], [49, 56], [88, 97]]
PROTEOR1_A_HOTSPOT_SPANS = [[52, 60], [62, 64], [66, 74], [77, 85], [98, 108], [109, 110]]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def replace_evolve_block(template: str, block: str) -> str:
    return re.sub(
        r"# EVOLVE-BLOCK-START.*?# EVOLVE-BLOCK-END",
        block.strip(),
        template,
        flags=re.S,
    )


def scfv_initial(case_id: str, block: str) -> str:
    template = (CASES_ROOT / "cd25_scfv" / "initial_program.py").read_text(encoding="utf-8")
    return replace_evolve_block(template, block).replace("cd25_scfv", case_id)


def generic_initial(case_id: str, block: str) -> str:
    template = (CASES_ROOT / "tetr_dopamine" / "initial_program.py").read_text(encoding="utf-8")
    text = replace_evolve_block(template, block).replace("tetr_dopamine", case_id)
    text = text.replace('"iterations": 240', '"iterations": 360')
    text = text.replace('"external_kb_weight": 0.40', '"external_kb_weight": 0.35')
    text = text.replace('"external_kb_retrieval_enabled": False', '"external_kb_retrieval_enabled": False')
    text = text.replace('"progen_weight": 0.5', '"progen_weight": 0.6')
    return text


def config_yaml(case_id: str, task_label: str, editable_nodes: str, base_case: str = "cd25_scfv") -> str:
    return dedent(
        f"""
        # General settings
        max_iterations: 200
        checkpoint_interval: 10
        log_level: "INFO"
        random_seed: 42
        max_tasks_per_child: 1

        # Evolution settings
        diff_based_evolution: true
        max_code_length: 32000

        llm:
          models:
            - name: "qwen3.7-max"
              weight: 1.0
          api_base: "${{ASTEVOLVE_LLM_API_BASE}}"
          api_key: "${{ASTEVOLVE_LLM_API_KEY}}"
          temperature: 0.85
          top_p: 0.95
          max_tokens: 4096
          timeout: 180
          retries: 1
          retry_delay: 5

        prompt:
          system_message: |
            You are ASTevolve's outer-loop strategy designer for the fixed {task_label} task.

            Modify only the EVOLVE-BLOCK in cases/{case_id}/initial_program.py. Prefer exact SEARCH/REPLACE diffs.
            Your entire response must be one or more exact diff blocks and nothing else:
            <<<<<<< SEARCH
            exact original code
            =======
            replacement code
            >>>>>>> REPLACE

            Preserve propose_strategy(), run_search(seed=None), preview_case(), design_state.json,
            evaluator.py, engine interfaces, and the fixed biological case identity.
            Runtime settings are locked outside the EVOLVE-BLOCK and will be overwritten before evaluation.

            Editable scope is layout/topology only:
            - create, remove, split, merge, reorder, or reweight design_regions;
            - choose binder_domain_order only within the declared case architecture;
            - map design regions to concrete node names: {editable_nodes};
            - tune length budgets/ranges only for mutable loop/CDR/linker-like nodes;
            - set secondary_structure_priors, site_anchors, motif candidates, and residue-class priors;
            - increase large-jump exploration by region design, not by changing iteration or solver parameters.

            Do not hard-code final sequences, copy retrieved proteins verbatim, edit target/decoy sequences,
            bypass evaluation, change model/KB paths, or add top-level strategy settings outside layout_plan.
            Use prompt artifacts such as llm_feedback_summary, layout_summary, strategy_schema_report,
            objective warnings, node pLDDT, and state/interface metrics to decide the next blueprint edit.

        evaluator:
          timeout: 180000
          parallel_evaluations: 1
          enable_artifacts: true
          cascade_evaluation: false
          use_llm_feedback: false
        """
    ).lstrip()


def manifest(case_id: str, name: str, task_type: str) -> dict:
    return {
        "case_id": case_id,
        "name": name,
        "task_type": task_type,
        "design_state_path": "design_state.json",
        "memory_path": "memory.yaml",
        "output_root": f"../../artifacts/{case_id}",
        "entry_program": "initial_program.py",
        "config_path": "config.yaml",
        "notes": "Self-contained ASTevolve case with case-specific typed multistate objectives.",
    }


def default_memory(case_id: str) -> str:
    return dedent(
        f"""
        case_id: {case_id}
        schema_version: ast_case_memory_v1
        best_candidates: []
        recent_runs: []
        node_observations: {{}}
        objective_observations: {{}}
        notes:
          - Generated by scripts/build_extended_cases.py; runtime runs may append/update this file.
        """
    ).lstrip()


def split_chain_by_spans(seq: str, spans: list[list[int]], prefix: str, cdr_names: list[str]) -> list[list[str]]:
    segments: list[list[str]] = []
    cursor = 0
    fr_idx = 1
    for idx, (start, end) in enumerate(spans):
        if cursor < start:
            segments.append([f"{prefix}_FR{fr_idx}", "framework", seq[cursor:start]])
            fr_idx += 1
        segments.append([cdr_names[idx], "cdr", seq[start:end]])
        cursor = end
    if cursor < len(seq):
        segments.append([f"{prefix}_FR{fr_idx}", "framework", seq[cursor:]])
    return segments


def sequence_from_binder(binder: dict) -> str:
    seq = ""
    for domain in binder.get("domain_order", []) or []:
        key = (binder.get("domain_segment_keys") or {}).get(domain)
        if key:
            seq += "".join(str(seg[2]) for seg in binder.get(key, []) if len(seg) >= 3)
    return seq


def replace_segment_sequence(binder: dict, segment_name: str, sequence: str) -> None:
    for key in (binder.get("domain_segment_keys") or {}).values():
        for segment in binder.get(key, []) or []:
            if len(segment) >= 3 and segment[0] == segment_name:
                if len(segment[2]) != len(sequence):
                    raise ValueError(
                        f"{segment_name} replacement length mismatch: {len(segment[2])} != {len(sequence)}"
                    )
                segment[2] = sequence
                return
    raise KeyError(f"Unknown binder segment: {segment_name}")


def add_benchmark_start(
    state: dict,
    *,
    case_id: str,
    start_type: str,
    reference_binder_sequence: str,
    changed_nodes: list[str],
    rationale: str,
) -> None:
    state["benchmark_start"] = {
        "schema_version": "ast_benchmark_start_v1",
        "case_id": case_id,
        "start_type": start_type,
        "initial_seed_sequence": sequence_from_binder(state["binder"]),
        "reference_oracle_sequence": reference_binder_sequence,
        "changed_nodes": changed_nodes,
        "rationale": rationale,
        "interpretation": (
            "The executable binder segments are intentionally degraded relative to the reference/oracle "
            "sequence so evolution has measurable room to improve. The reference sequence is metadata only "
            "and is not used as the starting template."
        ),
    }


def neutral_pattern(length: int, alphabet: str = "SGTNQ") -> str:
    return "".join(alphabet[i % len(alphabet)] for i in range(length))


def replace_spans(seq: str, spans: list[list[int]], replacements: list[str]) -> str:
    if len(spans) != len(replacements):
        raise ValueError("spans/replacements length mismatch")
    pieces: list[str] = []
    cursor = 0
    for (start, end), replacement in zip(spans, replacements):
        if len(replacement) != end - start:
            raise ValueError(f"replacement length mismatch for span {start}:{end}")
        pieces.append(seq[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(seq[cursor:])
    return "".join(pieces)


def degraded_antibody_cdrs() -> dict[str, str]:
    return {
        "VH_CDR1": "CKASGSSGGSSGMH",
        "VH_CDR2": "GSSGSGTGSSTGNQGS",
        "VH_CDR3": "GGGSGGSY",
        "VL_CDR1": "SASSSGSGMH",
        "VL_CDR2": "SGSGTAS",
        "VL_CDR3": "QGSGGSTLT",
    }


def apply_degraded_antibody_cdrs(binder: dict) -> None:
    for name, seq in degraded_antibody_cdrs().items():
        replace_segment_sequence(binder, name, seq)


def neutralize_residue_for_pocket(aa: str) -> str:
    if aa in "DE":
        return "Q"
    if aa in "KRH":
        return "N"
    if aa in "WYF":
        return "S"
    if aa in "LIVM":
        return "A"
    if aa == "C":
        return "S"
    return aa


def neutralize_segment(seq: str) -> str:
    return "".join(neutralize_residue_for_pocket(aa) for aa in seq)


def split_spans(seq: str, spans: list[list[int]], mutable_prefix: str, support_prefix: str, mutable_kind: str) -> list[list[str]]:
    segments: list[list[str]] = []
    cursor = 0
    support_idx = 1
    mutable_idx = 1
    for start, end in spans:
        if cursor < start:
            segments.append([f"{support_prefix}_{support_idx}", "framework", seq[cursor:start]])
            support_idx += 1
        segments.append([f"{mutable_prefix}_{mutable_idx}", mutable_kind, seq[start:end]])
        mutable_idx += 1
        cursor = end
    if cursor < len(seq):
        segments.append([f"{support_prefix}_{support_idx}", "framework", seq[cursor:]])
    return [seg for seg in segments if seg[2]]


def scfv_binder_from_cd25(*, degraded: bool = True) -> dict:
    base = json.loads((CASES_ROOT / "cd25_scfv" / "design_state.json").read_text(encoding="utf-8"))
    binder = deepcopy(base["binder"])
    if degraded:
        apply_degraded_antibody_cdrs(binder)
    return binder


def scfv_regions() -> dict:
    return {
        "cdr_binding_surface": ["VH_CDR1", "VH_CDR2", "VH_CDR3", "VL_CDR1", "VL_CDR2", "VL_CDR3"],
        "primary_cdr_hotspot": ["VH_CDR3", "VH_CDR2", "VL_CDR3"],
        "framework_stability_core": ["VH_FR1", "VH_FR2", "VH_FR3", "VH_FR4", "VL_FR1", "VL_FR2", "VL_FR3", "VL_FR4"],
        "linker_core": ["Linker_core"],
    }


def scfv_mutation_policy() -> dict:
    return {
        "preferred_edit_order": ["VH_CDR3", "VH_CDR2", "VL_CDR3", "VH_CDR1", "VL_CDR1", "VL_CDR2", "Linker_core", "VH_FR2", "VH_FR3", "VL_FR2", "VL_FR3"],
        "always_open_segments": ["VH_CDR1", "VH_CDR2", "VH_CDR3", "VL_CDR1", "VL_CDR2", "VL_CDR3"],
        "conditionally_open_segments": ["Linker_core", "VH_FR2", "VH_FR3", "VL_FR2", "VL_FR3"],
        "fixed_linker_segments": ["Linker_head", "Linker_tail"],
        "generally_frozen": ["FR1", "FR4"],
    }


def scfv_block(case_label: str, primary_target: str, decoy_target: str) -> str:
    return dedent(
        f"""
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
            \"\"\"OpenEvolve edits the {case_label} scFv topology/region blueprint only.\"\"\"
            strategy = base_strategy()
            strategy["layout_plan"] = {{
                "binder_domain_order": ["VH_domain", "Linker_module", "VL_domain"],
                "secondary_structure_priors": {{
                    "VH_CDR1": "loop", "VH_CDR2": "loop", "VH_CDR3": "loop",
                    "VL_CDR1": "loop", "VL_CDR2": "loop", "VL_CDR3": "loop",
                    "VH_FR2": "beta", "VH_FR3": "beta", "VL_FR2": "beta", "VL_FR3": "beta",
                }},
                "design_regions": [
                    {{
                        "name": "primary_selective_hotspot_paratope",
                        "role": "dominant CDR surface for {primary_target} epitope contacts while avoiding {decoy_target}",
                        "position": 1,
                        "bind_to": ["VH_CDR3", "VH_CDR2", "VL_CDR3"],
                        "secondary_structure": "loop",
                        "priority_boost": 1.55,
                        "length_budget": 28,
                        "node_weights": {{"VH_CDR3": 1.8, "VH_CDR2": 1.05, "VL_CDR3": 1.15}},
                        "length_ranges": {{"VH_CDR3": [8, 18], "VH_CDR2": [10, 18], "VL_CDR3": [7, 13]}},
                        "site_anchors": {{
                            "VH_CDR3": {{"relative_ranges": [[2, 8]], "weight": 2.4, "favored_residues": ["Y", "W", "H", "R", "D", "E", "N", "Q"]}},
                            "VH_CDR2": {{"relative_positions": [3, 6, 9], "weight": 1.7, "favored_residues": ["Y", "H", "S", "T", "N", "Q"]}},
                            "VL_CDR3": {{"relative_ranges": [[2, 6]], "weight": 1.8, "favored_residues": ["Y", "H", "S", "T", "N", "Q"]}},
                        }},
                        "mutation_rate": 0.12,
                        "max_mutations_per_step": 10,
                        "operator_phase": "explore",
                        "large_jump": True,
                        "motif_candidates": ["YY", "YW", "YH", "RY", "DY", "STY", "NQY"],
                        "mutation_ops": {{"point": 0.20, "block": 0.20, "site_resample": 0.22, "segment_mutagenesis": 0.20, "motif_graft": 0.15, "swap": 0.03}},
                        "favored_residues": ["Y", "W", "H", "R", "D", "E", "S", "T", "N", "Q", "G"],
                        "disfavored_residues": ["C"],
                        "policy_weight": 1.0,
                    }},
                    {{
                        "name": "secondary_shape_complementarity_surface",
                        "role": "secondary light-chain and CDR1 shaping for local epitope specificity",
                        "position": 2,
                        "bind_to": ["VL_CDR1", "VH_CDR1", "VL_CDR2"],
                        "secondary_structure": "loop",
                        "priority_boost": 1.05,
                        "length_budget": 24,
                        "node_weights": {{"VL_CDR1": 1.1, "VH_CDR1": 0.9, "VL_CDR2": 0.7}},
                        "length_ranges": {{"VL_CDR1": [8, 13], "VH_CDR1": [8, 14], "VL_CDR2": [6, 9]}},
                        "mutation_rate": 0.075,
                        "max_mutations_per_step": 6,
                        "operator_phase": "refine",
                        "large_jump": True,
                        "mutation_ops": {{"point": 0.34, "block": 0.18, "site_resample": 0.22, "segment_mutagenesis": 0.14, "motif_graft": 0.08, "swap": 0.04}},
                        "favored_residues": ["Y", "H", "S", "T", "N", "Q", "R", "D", "G"],
                        "disfavored_residues": ["C"],
                        "policy_weight": 0.76,
                    }},
                    {{
                        "name": "vh_vl_geometry_and_linker_support",
                        "role": "adjust linker and sparse framework-adjacent geometry without replacing the scaffold",
                        "position": 3,
                        "bind_to": ["Linker_core", "VH_FR2", "VH_FR3", "VL_FR2", "VL_FR3"],
                        "secondary_structure": "beta",
                        "priority_boost": 0.55,
                        "length_ranges": {{"Linker_core": [5, 10]}},
                        "length_mutable": True,
                        "mutation_rate": 0.025,
                        "max_mutations_per_step": 2,
                        "operator_phase": "stabilize",
                        "mutation_ops": {{"point": 0.82, "block": 0.08, "segment_resample": 0.05, "swap": 0.05}},
                        "favored_residues": ["G", "S", "A", "T", "N", "Q"],
                        "disfavored_residues": ["C", "W"],
                        "policy_weight": 0.35,
                    }},
                ],
            }}
            return strategy


        # EVOLVE-BLOCK-END
        """
    )


def generic_block(case_label: str, domain_order: list[str], regions: list[dict], ss_priors: dict[str, str]) -> str:
    text = (
        f"""
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
            \"\"\"OpenEvolve edits the {case_label} protein topology/region blueprint only.\"\"\"
            strategy = base_strategy()
            strategy["layout_plan"] = {{
                "binder_domain_order": {pformat(domain_order, width=100)},
                "secondary_structure_priors": {pformat(ss_priors, width=100)},
                "design_regions": {pformat(regions, width=120)},
            }}
            return strategy


        # EVOLVE-BLOCK-END
        """
    )
    return re.sub(r"(?m)^        ", "", text).strip()


def write_case(case_id: str, name: str, task_type: str, design_state: dict, initial_text: str, config_text: str, sheet: dict) -> None:
    case_dir = CASES_ROOT / case_id
    write_json(case_dir / "case.json", manifest(case_id, name, task_type))
    write_json(case_dir / "design_state.json", design_state)
    write_json(case_dir / "case_sheet.json", sheet)
    write_text(case_dir / "initial_program.py", initial_text)
    write_text(case_dir / "config.yaml", config_text)
    if not (case_dir / "memory.yaml").exists():
        write_text(case_dir / "memory.yaml", default_memory(case_id))


def case_sheet(
    case_id: str,
    primary: str,
    positives: list[str],
    failures: list[str],
    thresholds: dict,
    gaps: list[str],
    benchmark_start: dict | None = None,
) -> dict:
    return {
        "schema_version": "ast_case_sheet_v1",
        "case_name": case_id,
        "readiness": {
            "runtime": "ready",
            "biological_specificity": "structured_provisional",
            "primary_reason": "Case includes explicit states, target/decoy entities, node regions, hotspot spans, typed objectives, and confidence/clash guardrails; wet-lab construct details can further tighten thresholds.",
        },
        "design_goal": {
            "primary": primary,
            "formal_success_definition": "Improve the typed positive-state objectives while suppressing typed negative/decoy objectives, preserving local node confidence, and avoiding severe clashes or unfolding.",
        },
        "state_success_criteria": {
            "positive_signals": positives,
            "failure_signals": failures,
        },
        "objective_thresholds": thresholds,
        "benchmark_start": benchmark_start or {},
        "information_gaps": gaps,
    }


def build_cd25_selectivity() -> None:
    base = json.loads((CASES_ROOT / "cd25_scfv" / "design_state.json").read_text(encoding="utf-8"))
    state = deepcopy(base)
    reference_binder_sequence = sequence_from_binder(base["binder"])
    state["binder"] = scfv_binder_from_cd25(degraded=True)
    state["task_name"] = "CD25_scFv_Selectivity_Design_Task"
    state["task_type"] = "selective_scfv_epitope_design"
    state["version"] = "cd25_scfv_selectivity_v1"
    state["case_sheet_path"] = "case_sheet.json"
    state["target"]["name"] = "CD25"
    state["target"]["epitope_name"] = "CD25_basiliximab_epitope"
    state["additional_targets"] = {
        "IL2RB_decoy": {
            "name": "IL2RB_CD122_decoy",
            "sequence": IL2RB_ECD_PROXY,
            "biological_reason": "CD122 is an IL-2 receptor-family off-target decoy; selectivity should favor CD25 over CD122-like receptor surfaces.",
        }
    }
    regions = scfv_regions()
    regions["target_epitope_focus"] = ["CD25_basiliximab_epitope"]
    state["multistate_regions"] = regions
    state["complex_states"] = [
        {"name": "cd25_positive_state", "role": "desired CD25-bound state", "objective": "CDR surface should bind CD25 at the basiliximab-like epitope.", "metric": "plddt", "entities": [{"type": "protein", "id": "scFv", "source_chain": "BB"}, {"type": "protein", "id": "CD25", "source_chain": "T"}]},
        {"name": "il2rb_decoy_state", "role": "negative receptor-family decoy state", "objective": "The same scFv should not form a strong broad interface with CD122/IL2RB.", "metric": "plddt", "entities": [{"type": "protein", "id": "scFv", "source_chain": "BB"}, {"type": "protein", "id": "IL2RB", "sequence": IL2RB_ECD_PROXY}]},
    ]
    state["multistate_objectives"] = [
        {"name": "cd25_epitope_interface_on", "type": "interface_on", "state": "cd25_positive_state", "pair": ["scFv", "CD25"], "left_region": "cdr_binding_surface", "right_region": "CD25_basiliximab_epitope", "weight": 1.15, "contact_target": 64, "residue_pair_target": 15, "coverage_target": 0.35, "off_target_penalty_weight": 0.20, "off_target_contact_tolerance": 130},
        {"name": "cd25_epitope_specificity", "type": "epitope_specificity", "state": "cd25_positive_state", "pair": ["scFv", "CD25"], "left_region": "cdr_binding_surface", "target_region": "CD25_basiliximab_epitope", "weight": 0.85, "target_contact_fraction": 0.55, "contact_target": 60, "coverage_target": 0.35},
        {"name": "il2rb_decoy_interface_off", "type": "interface_off", "state": "il2rb_decoy_state", "pair": ["scFv", "IL2RB"], "left_region": "cdr_binding_surface", "weight": 1.0, "contact_target": 45, "residue_pair_target": 10, "off_target_contact_tolerance": 65},
        {"name": "cd25_over_il2rb_selectivity_delta", "type": "interface_delta", "positive_state": "cd25_positive_state", "negative_state": "il2rb_decoy_state", "positive_pair": ["scFv", "CD25"], "negative_pair": ["scFv", "IL2RB"], "positive_left_region": "cdr_binding_surface", "positive_right_region": "CD25_basiliximab_epitope", "negative_left_region": "cdr_binding_surface", "weight": 1.2, "direction": "decrease", "min_delta": 0.12, "target_delta": 0.45, "contact_delta_target": 35},
        {"name": "cdr_confidence_floor", "type": "region_confidence_floor", "states": ["cd25_positive_state", "il2rb_decoy_state"], "region": "cdr_binding_surface", "metric": "plddt_min", "floor": 45, "target": 72, "weight": 0.55},
        {"name": "bound_state_confidence", "type": "confidence", "states": ["cd25_positive_state", "il2rb_decoy_state"], "metric": "plddt", "weight": 0.45},
    ]
    state["design_points"]["design_intent"] = "Design a CD25-selective scFv: bind the CD25 basiliximab-like epitope, avoid IL2RB/CD122-like receptor-family decoy binding, and keep CDR/framework confidence interpretable."
    state["design_points"]["initial_seed_policy"] = "CDRs are neutralized from the previous CD25 template; framework/linker are preserved. This avoids starting from a likely strong CD25 binder."
    state["mutation_policy"] = scfv_mutation_policy()
    add_benchmark_start(
        state,
        case_id="cd25_scfv_selectivity",
        start_type="degraded_neutral_cdr_seed",
        reference_binder_sequence=reference_binder_sequence,
        changed_nodes=list(degraded_antibody_cdrs()),
        rationale="CD25 selectivity should be measured from a weak generic CDR seed, not from the original CD25 scFv CDRs.",
    )
    block = scfv_block("CD25-selective", "CD25", "IL2RB/CD122")
    sheet = case_sheet(
        "cd25_scfv_selectivity",
        "Design a VH-linker-VL scFv that binds a known CD25 epitope while rejecting an IL2RB/CD122 decoy surface.",
        ["CDR-to-CD25 epitope contacts increase.", "A majority of CD25 contacts stay on the specified epitope spans.", "IL2RB decoy contacts remain weak.", "CDR node confidence remains above the hard floor."],
        ["Decoy IL2RB interface is comparable to CD25.", "Contacts are mostly outside the specified CD25 epitope.", "Interface relies on linker/framework rather than CDRs.", "CDRs collapse or severe clashes dominate."],
        {"cd25_epitope_interface_on": {"contacts": 64, "residue_pairs": 15, "coverage": 0.35}, "selectivity_delta": {"min_delta": 0.12, "target_delta": 0.45, "contact_delta_target": 35}, "cdr_confidence": {"hard_floor": 45, "target": 72}},
        ["Exact wet-lab CD25 construct boundaries.", "Whether basiliximab-like competition is required or just CD25-selective binding.", "Preferred developability liabilities and CDR length bounds."],
        benchmark_start=state["benchmark_start"],
    )
    write_case("cd25_scfv_selectivity", "CD25 selective scFv design", "selective_scfv_epitope_design", state, scfv_initial("cd25_scfv_selectivity", block), config_yaml("cd25_scfv_selectivity", "CD25-selective scFv", "VH_CDR1, VH_CDR2, VH_CDR3, VL_CDR1, VL_CDR2, VL_CDR3, Linker_core, sparse FR2/FR3 support"), sheet)


def build_pdl1_selectivity() -> None:
    state = {
        "task_name": "PDL1_scFv_Selectivity_Design_Task",
        "task_type": "selective_scfv_epitope_design",
        "version": "pdl1_scfv_selectivity_v1",
        "memory_path": "memory.yaml",
        "case_sheet_path": "case_sheet.json",
        "binder": scfv_binder_from_cd25(degraded=True),
        "target": {
            "chain_id": "T",
            "name": "PDL1",
            "uniprot_id": "Q9NZQ7",
            "sequence_numbering": "First residue is mature PD-L1 ECD residue 1 from the local case proxy.",
            "sequence": PDL1_ECD,
            "epitope_name": "PDL1_avelumab_epitope",
            "epitope_source": "RCSB 5GRJ chain C antibody-contact residues, 4.5 A heavy-atom cutoff, converted to zero-based end-exclusive spans.",
            "epitope_spans": PDL1_AVELUMAB_EPITOPE,
            "hotspot_motif": "YRCMISYGGADYKRITVKVN",
        },
        "additional_targets": {
            "PDL2_decoy": {"name": "PDL2", "sequence": PDL2_ECD_PROXY, "biological_reason": "PD-L2 is the closest functional homolog; anti-PD-L1 selectivity should avoid strong PD-L2 binding."},
            "PD1_decoy": {"name": "PD1", "sequence": PD1_ECD, "biological_reason": "PD-1 is included as an off-target protein to discourage receptor-like binding by the designed scFv."},
        },
        "design_points": {
            "schema_version": "ast_design_points_v1",
            "design_intent": "Design an anti-PD-L1 scFv that focuses on avelumab/PD-1-overlap ECD surface while staying selective against PD-L2 and PD-1 off-target proteins.",
            "primary_design_nodes": ["VH_CDR3", "VH_CDR2", "VL_CDR3"],
            "secondary_design_nodes": ["VH_CDR1", "VL_CDR1", "VL_CDR2"],
            "default_open_nodes": ["VH_CDR3", "VH_CDR2", "VL_CDR3", "VH_CDR1", "VL_CDR1", "VL_CDR2"],
            "preserved_nodes": ["VH_FR1", "VH_FR4", "VL_FR1", "VL_FR4", "Linker_head", "Linker_tail"],
        },
        "complex_states": [
            {"name": "pdl1_positive_state", "role": "desired PD-L1-bound state", "objective": "scFv CDRs should bind the PD-L1 ECD at the avelumab-like epitope.", "metric": "plddt", "entities": [{"type": "protein", "id": "scFv", "source_chain": "BB"}, {"type": "protein", "id": "PDL1", "source_chain": "T"}]},
            {"name": "pdl2_decoy_state", "role": "PD-L2 homolog negative state", "objective": "scFv should avoid strong PD-L2 binding.", "metric": "plddt", "entities": [{"type": "protein", "id": "scFv", "source_chain": "BB"}, {"type": "protein", "id": "PDL2", "sequence": PDL2_ECD_PROXY}]},
            {"name": "pd1_offtarget_state", "role": "PD-1 receptor off-target state", "objective": "scFv should avoid receptor-like off-target binding to PD-1.", "metric": "plddt", "entities": [{"type": "protein", "id": "scFv", "source_chain": "BB"}, {"type": "protein", "id": "PD1", "sequence": PD1_ECD}]},
        ],
        "multistate_regions": {**scfv_regions(), "target_epitope_focus": ["PDL1_avelumab_epitope"], "pd1_overlap_surface": ["PDL1_PD1_interface"]},
        "multistate_objectives": [
            {"name": "pdl1_epitope_interface_on", "type": "interface_on", "state": "pdl1_positive_state", "pair": ["scFv", "PDL1"], "left_region": "cdr_binding_surface", "right_region": "PDL1_avelumab_epitope", "weight": 1.2, "contact_target": 58, "residue_pair_target": 14, "coverage_target": 0.32, "off_target_penalty_weight": 0.20, "off_target_contact_tolerance": 120},
            {"name": "pdl1_epitope_specificity", "type": "epitope_specificity", "state": "pdl1_positive_state", "pair": ["scFv", "PDL1"], "left_region": "cdr_binding_surface", "target_region": "PDL1_avelumab_epitope", "weight": 0.9, "target_contact_fraction": 0.55, "contact_target": 55, "coverage_target": 0.32},
            {"name": "pdl2_decoy_interface_off", "type": "interface_off", "state": "pdl2_decoy_state", "pair": ["scFv", "PDL2"], "left_region": "cdr_binding_surface", "weight": 0.95, "contact_target": 42, "residue_pair_target": 9, "off_target_contact_tolerance": 60},
            {"name": "pd1_offtarget_interface_off", "type": "interface_off", "state": "pd1_offtarget_state", "pair": ["scFv", "PD1"], "left_region": "cdr_binding_surface", "weight": 0.65, "contact_target": 35, "residue_pair_target": 8, "off_target_contact_tolerance": 55},
            {"name": "pdl1_over_pdl2_selectivity_delta", "type": "interface_delta", "positive_state": "pdl1_positive_state", "negative_state": "pdl2_decoy_state", "positive_pair": ["scFv", "PDL1"], "negative_pair": ["scFv", "PDL2"], "positive_left_region": "cdr_binding_surface", "positive_right_region": "PDL1_avelumab_epitope", "negative_left_region": "cdr_binding_surface", "weight": 1.15, "direction": "decrease", "min_delta": 0.12, "target_delta": 0.45, "contact_delta_target": 32},
            {"name": "cdr_confidence_floor", "type": "region_confidence_floor", "states": ["pdl1_positive_state", "pdl2_decoy_state", "pd1_offtarget_state"], "region": "cdr_binding_surface", "metric": "plddt_min", "floor": 45, "target": 72, "weight": 0.55},
            {"name": "state_confidence", "type": "confidence", "states": ["pdl1_positive_state", "pdl2_decoy_state", "pd1_offtarget_state"], "metric": "plddt", "weight": 0.45},
        ],
        "mutation_policy": scfv_mutation_policy(),
        "case_information_needed": ["Whether the desired antibody should block PD-1 by binding the PD-1 interface or bind a non-blocking PD-L1 epitope.", "Exact mature PD-L1/PD-L2 construct boundaries for assay matching.", "Allowed human framework families and developability liabilities."],
    }
    pdl1_reference_binder = sequence_from_binder(scfv_binder_from_cd25(degraded=False))
    state["design_points"]["initial_seed_policy"] = "Generic neutralized CDR seed on the same scFv framework; no PD-L1 oracle CDR is used as the executable starting point."
    add_benchmark_start(
        state,
        case_id="pdl1_scfv_selectivity",
        start_type="generic_degraded_neutral_cdr_seed",
        reference_binder_sequence=pdl1_reference_binder,
        changed_nodes=list(degraded_antibody_cdrs()),
        rationale="PD-L1 design starts from generic weak CDRs to test whether AST can discover a PD-L1-selective paratope rather than inherit a tuned binder.",
    )
    block = scfv_block("PD-L1-selective", "PD-L1", "PD-L2/PD-1")
    sheet = case_sheet(
        "pdl1_scfv_selectivity",
        "Design a PD-L1-selective scFv focused on an avelumab-like PD-L1 ECD hotspot while rejecting PD-L2 and PD-1 decoys.",
        ["CDR-to-PD-L1 epitope contacts form.", "PD-L1 contacts are concentrated on the known hotspot spans.", "PD-L2 and PD-1 decoy interfaces remain weak.", "CDR confidence remains interpretable."],
        ["PD-L2 or PD-1 binding approaches PD-L1 binding.", "Contacts move away from the PD-L1 epitope.", "Framework/linker dominates the interface.", "CDRs are low-confidence or clashing."],
        {"pdl1_epitope_interface_on": {"contacts": 58, "residue_pairs": 14, "coverage": 0.32}, "pdl1_over_pdl2_delta": {"min_delta": 0.12, "target_delta": 0.45, "contact_delta_target": 32}, "cdr_confidence": {"hard_floor": 45, "target": 72}},
        state["case_information_needed"],
        benchmark_start=state["benchmark_start"],
    )
    write_case("pdl1_scfv_selectivity", "PD-L1 selective scFv design", "selective_scfv_epitope_design", state, scfv_initial("pdl1_scfv_selectivity", block), config_yaml("pdl1_scfv_selectivity", "PD-L1-selective scFv", "VH_CDR1, VH_CDR2, VH_CDR3, VL_CDR1, VL_CDR2, VL_CDR3, Linker_core, sparse FR2/FR3 support"), sheet)


def build_proteor1_cdr_mask() -> None:
    seed_h = replace_spans(
        PROTEOR1_H,
        PROTEOR1_H_SPANS,
        ["GSSGGSY", "SGSGTY", "GGSGGSGGSGDY"],
    )
    seed_l = replace_spans(
        PROTEOR1_L,
        PROTEOR1_L_SPANS,
        ["SGSGGSGSGSY", "SGSGTAS", "QGSGGSTLT"],
    )
    binder = {
        "chain_id": "BB",
        "architecture": "ProteoR1_HL_scFv_masked_CDR",
        "domain_segment_keys": {"VH_domain": "vh_segments", "Linker_module": "linker_segments", "VL_domain": "vl_segments"},
        "domain_aliases": {"VH": "VH_domain", "heavy": "VH_domain", "Linker": "Linker_module", "VL": "VL_domain", "light": "VL_domain"},
        "domain_order": ["VH_domain", "Linker_module", "VL_domain"],
        "vh_segments": split_chain_by_spans(seed_h, PROTEOR1_H_SPANS, "VH", ["VH_CDR1", "VH_CDR2", "VH_CDR3"]),
        "linker_segments": [["Linker_head", "linker", "GGGGS"], ["Linker_core", "linker", "GGGGS"], ["Linker_tail", "linker", "GGGGS"]],
        "vl_segments": split_chain_by_spans(seed_l, PROTEOR1_L_SPANS, "VL", ["VL_CDR1", "VL_CDR2", "VL_CDR3"]),
    }
    state = {
        "task_name": "ProteoR1_CDR_Mask_Reevolution_Task",
        "task_type": "masked_cdr_scfv_evolution",
        "version": "proteor1_cdr_mask_v1",
        "memory_path": "memory.yaml",
        "case_sheet_path": "case_sheet.json",
        "binder": binder,
        "target": {"chain_id": "T", "name": "ProteoR1_demo_antigen_8r9y_A", "sequence": PROTEOR1_A, "epitope_name": "ProteoR1_spec_mask_hotspot", "epitope_source": "Proteo-R1 demo canonical YAML 8r9y_H_L_A antigen spec_mask runs.", "epitope_spans": PROTEOR1_A_HOTSPOT_SPANS},
        "design_points": {"schema_version": "ast_design_points_v1", "design_intent": "Replicate a Proteo-R1-style masked CDR redesign benchmark without copying its optimization method: evolve the CDR mask under AST typed structural scoring.", "initial_seed_policy": "Executable H/L CDRs are neutral fills at the Proteo-R1 mask spans; the YAML ground_truth CDRs are stored only as oracle metadata.", "primary_design_nodes": ["VH_CDR3", "VH_CDR1", "VL_CDR1", "VL_CDR3"], "secondary_design_nodes": ["VH_CDR2", "VL_CDR2"], "default_open_nodes": ["VH_CDR1", "VH_CDR2", "VH_CDR3", "VL_CDR1", "VL_CDR2", "VL_CDR3"], "preserved_nodes": ["VH_FR1", "VH_FR2", "VH_FR3", "VH_FR4", "VL_FR1", "VL_FR2", "VL_FR3", "VL_FR4"]},
        "complex_states": [{"name": "proteor1_antigen_bound_state", "role": "masked CDR antigen-bound state", "objective": "Redesigned CDRs should recover hotspot-centered binding to the fixed antigen.", "metric": "plddt", "entities": [{"type": "protein", "id": "scFv", "source_chain": "BB"}, {"type": "protein", "id": "Antigen_A", "source_chain": "T"}]}],
        "multistate_regions": {**scfv_regions(), "target_epitope_focus": ["ProteoR1_spec_mask_hotspot"]},
        "multistate_objectives": [
            {"name": "masked_cdr_hotspot_interface_on", "type": "interface_on", "state": "proteor1_antigen_bound_state", "pair": ["scFv", "Antigen_A"], "left_region": "cdr_binding_surface", "right_region": "ProteoR1_spec_mask_hotspot", "weight": 1.25, "contact_target": 70, "residue_pair_target": 18, "coverage_target": 0.38, "off_target_penalty_weight": 0.22, "off_target_contact_tolerance": 140},
            {"name": "cdr3_hotspot_specificity", "type": "epitope_specificity", "state": "proteor1_antigen_bound_state", "pair": ["scFv", "Antigen_A"], "left_region": "primary_cdr_hotspot", "target_region": "ProteoR1_spec_mask_hotspot", "weight": 0.85, "target_contact_fraction": 0.55, "contact_target": 45, "coverage_target": 0.30},
            {"name": "cdr_mask_confidence_floor", "type": "region_confidence_floor", "states": ["proteor1_antigen_bound_state"], "region": "cdr_binding_surface", "metric": "plddt_min", "floor": 45, "target": 72, "weight": 0.6},
            {"name": "bound_state_confidence", "type": "confidence", "states": ["proteor1_antigen_bound_state"], "metric": "plddt", "weight": 0.45},
        ],
        "mutation_policy": scfv_mutation_policy(),
        "case_information_needed": ["Whether the goal is de novo CDR recovery, improvement over ground truth, or robustness to masked inputs.", "External negative antigens for selectivity if desired."],
    }
    add_benchmark_start(
        state,
        case_id="proteor1_cdr_mask",
        start_type="proteor1_masked_cdr_neutral_fill",
        reference_binder_sequence=PROTEOR1_H + "GGGGSGGGGSGGGGS" + PROTEOR1_L,
        changed_nodes=["VH_CDR1", "VH_CDR2", "VH_CDR3", "VL_CDR1", "VL_CDR2", "VL_CDR3"],
        rationale="The Proteo-R1 ground-truth CDRs are not used as the executable start; masked CDR spans are filled with neutral residues to create recovery room.",
    )
    block = scfv_block("ProteoR1-style masked CDR", "8r9y antigen hotspot", "non-hotspot antigen surface")
    sheet = case_sheet(
        "proteor1_cdr_mask",
        "Re-evolve masked H/L CDRs from a Proteo-R1 demo case while preserving framework and focusing the antigen spec-mask hotspot.",
        ["CDR-hotspot contacts recover.", "Contacts remain concentrated on ProteoR1_spec_mask_hotspot.", "Masked CDR node confidence stays above floor."],
        ["Interface is broad but not hotspot-centered.", "Framework replaces CDRs as binder.", "CDR loops become low-confidence."],
        {"hotspot_interface": {"contacts": 70, "residue_pairs": 18, "coverage": 0.38}, "cdr_confidence": {"hard_floor": 45, "target": 72}},
        state["case_information_needed"],
        benchmark_start=state["benchmark_start"],
    )
    write_case("proteor1_cdr_mask", "Proteo-R1 style masked CDR evolution", "masked_cdr_scfv_evolution", state, scfv_initial("proteor1_cdr_mask", block), config_yaml("proteor1_cdr_mask", "Proteo-R1-style masked CDR", "VH_CDR1, VH_CDR2, VH_CDR3, VL_CDR1, VL_CDR2, VL_CDR3, Linker_core, sparse FR2/FR3 support"), sheet)


def build_pdz() -> None:
    pdz_seed = replace_spans(
        PDZ_SEQ,
        PDZ_GROOVE_SPANS,
        [neutral_pattern(end - start, "ASG") for start, end in PDZ_GROOVE_SPANS],
    )
    segments = split_spans(pdz_seed, PDZ_GROOVE_SPANS, "PDZ_groove", "PDZ_scaffold", "framework")
    binder = {
        "name": "PDZ_domain",
        "chain_id": "BB",
        "architecture": "single_domain_PDZ_peptide_binding_domain",
        "domain_segment_keys": {"PDZ_domain": "pdz_segments"},
        "domain_order": ["PDZ_domain"],
        "pdz_segments": segments,
    }
    groove_nodes = [seg[0] for seg in segments if seg[0].startswith("PDZ_groove")]
    scaffold_nodes = [seg[0] for seg in segments if seg[0].startswith("PDZ_scaffold")]
    state = {
        "task_name": "PDZ_Peptide_Selectivity_Design_Task",
        "task_type": "peptide_binding_domain_selectivity",
        "version": "pdz_peptide_selectivity_v1",
        "memory_path": "memory.yaml",
        "case_sheet_path": "case_sheet.json",
        "binder": binder,
        "target": {"chain_id": "T", "name": "target_C_terminal_peptide", "sequence": "KQTSV", "feature_kind": "peptide", "epitope_name": "PDZ_target_peptide", "epitope_source": "1BE9 chain B target peptide KQTSV; full peptide is the desired contact region.", "epitope_spans": [[0, 5]]},
        "additional_targets": {"decoy_peptide": {"name": "decoy_C_terminal_peptide", "sequence": "KKAAA", "biological_reason": "A charge/small-residue decoy peptide lacking the canonical terminal Val motif."}},
        "design_points": {"schema_version": "ast_design_points_v1", "design_intent": "Retune PDZ groove specificity for KQTSV-like target peptide while suppressing a decoy peptide.", "initial_seed_policy": "The executable PDZ scaffold keeps 1BE9 backbone-length segmentation but neutralizes peptide-contact groove residues; native 1BE9 is oracle metadata only.", "primary_design_nodes": groove_nodes, "secondary_design_nodes": scaffold_nodes[:2], "default_open_nodes": groove_nodes, "preserved_nodes": scaffold_nodes},
        "complex_states": [
            {"name": "target_peptide_bound_state", "role": "desired target peptide state", "objective": "PDZ groove should bind KQTSV target peptide.", "metric": "plddt", "entities": [{"type": "protein", "id": "PDZ", "source_chain": "BB"}, {"type": "protein", "id": "target_peptide", "source_chain": "T"}]},
            {"name": "decoy_peptide_state", "role": "negative decoy peptide state", "objective": "PDZ groove should not bind decoy peptide as strongly.", "metric": "plddt", "entities": [{"type": "protein", "id": "PDZ", "source_chain": "BB"}, {"type": "protein", "id": "decoy_peptide", "sequence": "KKAAA"}]},
        ],
        "multistate_regions": {"pdz_groove": groove_nodes, "pdz_scaffold": scaffold_nodes, "target_peptide": ["PDZ_target_peptide"]},
        "multistate_objectives": [
            {"name": "target_peptide_interface_on", "type": "interface_on", "state": "target_peptide_bound_state", "pair": ["PDZ", "target_peptide"], "left_region": "pdz_groove", "right_region": "PDZ_target_peptide", "weight": 1.2, "contact_target": 24, "residue_pair_target": 7, "coverage_target": 0.45, "off_target_penalty_weight": 0.10, "off_target_contact_tolerance": 70},
            {"name": "decoy_peptide_interface_off", "type": "interface_off", "state": "decoy_peptide_state", "pair": ["PDZ", "decoy_peptide"], "left_region": "pdz_groove", "weight": 1.0, "contact_target": 18, "residue_pair_target": 5, "off_target_contact_tolerance": 30},
            {"name": "target_over_decoy_delta", "type": "interface_delta", "positive_state": "target_peptide_bound_state", "negative_state": "decoy_peptide_state", "positive_pair": ["PDZ", "target_peptide"], "negative_pair": ["PDZ", "decoy_peptide"], "positive_left_region": "pdz_groove", "positive_right_region": "PDZ_target_peptide", "negative_left_region": "pdz_groove", "weight": 1.25, "direction": "decrease", "min_delta": 0.10, "target_delta": 0.40, "contact_delta_target": 18},
            {"name": "groove_confidence_floor", "type": "region_confidence_floor", "states": ["target_peptide_bound_state", "decoy_peptide_state"], "region": "pdz_groove", "metric": "plddt_min", "floor": 45, "target": 75, "weight": 0.55},
            {"name": "state_confidence", "type": "confidence", "states": ["target_peptide_bound_state", "decoy_peptide_state"], "metric": "plddt", "weight": 0.45},
        ],
        "mutation_policy": {"preferred_edit_order": groove_nodes + scaffold_nodes[:2], "always_open_segments": groove_nodes, "conditionally_open_segments": scaffold_nodes[:2], "generally_frozen": scaffold_nodes[2:]},
        "case_information_needed": ["Actual target/decoy peptide panel for the biological question.", "Whether backbone remodeling of the PDZ groove is allowed.", "Desired peptide class constraints beyond KQTSV."],
    }
    add_benchmark_start(
        state,
        case_id="pdz_peptide_selectivity",
        start_type="pdz_groove_neutralized_seed",
        reference_binder_sequence=PDZ_SEQ,
        changed_nodes=groove_nodes,
        rationale="Native 1BE9 already binds KQTSV-like peptide, so the executable start neutralizes groove contact residues to create a real selectivity recovery task.",
    )
    regions = [
        {"name": "pdz_target_specificity_groove", "role": "primary peptide-binding groove residues from 1BE9 contacts", "position": 1, "bind_to": groove_nodes, "secondary_structure": "beta", "priority_boost": 1.45, "mutation_rate": 0.09, "max_mutations_per_step": 7, "operator_phase": "explore", "large_jump": True, "mutation_ops": {"point": 0.32, "block": 0.18, "site_resample": 0.24, "segment_mutagenesis": 0.16, "motif_graft": 0.06, "swap": 0.04}, "favored_residues": ["Y", "F", "H", "S", "T", "N", "Q", "R", "K", "D", "E"], "disfavored_residues": ["C"], "policy_weight": 0.95},
        {"name": "pdz_scaffold_guardrail", "role": "sparse support around groove while preserving PDZ fold", "position": 2, "bind_to": scaffold_nodes[:3], "secondary_structure": "beta", "priority_boost": 0.40, "mutation_rate": 0.018, "max_mutations_per_step": 1, "operator_phase": "stabilize", "mutation_ops": {"point": 0.92, "block": 0.03, "swap": 0.05}, "favored_residues": ["A", "S", "T", "N", "Q", "V", "I", "L"], "disfavored_residues": ["C", "P"], "policy_weight": 0.30},
    ]
    block = generic_block("PDZ peptide-selective", ["PDZ_domain"], regions, {name: "beta" for name in groove_nodes})
    sheet = case_sheet(
        "pdz_peptide_selectivity",
        "Retune a PDZ groove to bind KQTSV target peptide and reject KKAAA decoy peptide.",
        ["Groove-target peptide contacts form.", "Decoy peptide contacts stay lower.", "PDZ groove confidence remains stable."],
        ["Decoy binds as strongly as target.", "Contacts are made by scaffold outside the groove.", "Groove mutations unfold the domain."],
        {"target_interface": {"contacts": 24, "residue_pairs": 7, "coverage": 0.45}, "target_over_decoy_delta": {"min_delta": 0.10, "target_delta": 0.40, "contact_delta_target": 18}, "groove_confidence": {"hard_floor": 45, "target": 75}},
        state["case_information_needed"],
        benchmark_start=state["benchmark_start"],
    )
    write_case("pdz_peptide_selectivity", "PDZ peptide selectivity design", "peptide_binding_domain_selectivity", state, generic_initial("pdz_peptide_selectivity", block), config_yaml("pdz_peptide_selectivity", "PDZ peptide-selectivity", ", ".join(groove_nodes + scaffold_nodes[:3])), sheet)


def build_calcium() -> None:
    calcium_seed = replace_spans(
        CALMODULIN_SEQ,
        CALMODULIN_EF_SPANS,
        [neutralize_segment(CALMODULIN_SEQ[start:end]) for start, end in CALMODULIN_EF_SPANS],
    )
    segments = split_spans(calcium_seed, CALMODULIN_EF_SPANS, "EFhand_loop", "CaM_scaffold", "framework")
    loop_nodes = [seg[0] for seg in segments if seg[0].startswith("EFhand_loop")]
    scaffold_nodes = [seg[0] for seg in segments if seg[0].startswith("CaM_scaffold")]
    binder = {
        "name": "Calmodulin_EF_hand_sensor",
        "chain_id": "BB",
        "architecture": "single_chain_four_EF_hand_calcium_switch",
        "domain_segment_keys": {"CaM_EFhand_domain": "cam_segments"},
        "domain_order": ["CaM_EFhand_domain"],
        "cam_segments": segments,
    }
    state = {
        "task_name": "Calcium_EFhand_Switch_Design_Task",
        "task_type": "ion_gated_peptide_binding_switch",
        "version": "calcium_efhand_switch_v1",
        "memory_path": "memory.yaml",
        "case_sheet_path": "case_sheet.json",
        "binder": binder,
        "target": {"chain_id": "T", "name": "MLCK_like_target_peptide", "sequence": MLCK_PEPTIDE, "feature_kind": "peptide", "epitope_name": "MLCK_peptide_binding_surface", "epitope_source": "Canonical calmodulin target-peptide readout proxy; full peptide is treated as the desired binding region.", "epitope_spans": [[0, len(MLCK_PEPTIDE)]]},
        "ligands": {"calcium": {"name": "calcium", "ccd": "CA", "role": "EF-hand activating ion"}},
        "design_points": {"schema_version": "ast_design_points_v1", "design_intent": "Design an EF-hand calcium switch: apo state should avoid strong peptide binding; calcium-loaded state should keep acidic EF pockets and improve peptide interface.", "initial_seed_policy": "The executable CaM scaffold preserves domain length and scaffold segments but neutralizes the canonical EF-hand loop chemistry. Native calmodulin is reference metadata only.", "primary_design_nodes": loop_nodes, "secondary_design_nodes": scaffold_nodes[:3], "default_open_nodes": loop_nodes, "preserved_nodes": scaffold_nodes},
        "complex_states": [
            {"name": "apo_peptide_state", "role": "apo negative peptide-binding state", "objective": "Without calcium, the scaffold should avoid strong MLCK peptide binding while remaining folded.", "metric": "plddt", "entities": [{"type": "protein", "id": "CaM", "source_chain": "BB"}, {"type": "protein", "id": "MLCK_peptide", "source_chain": "T"}]},
            {"name": "calcium_peptide_state", "role": "calcium-loaded positive peptide-binding state", "objective": "Calcium-loaded CaM should bind the MLCK-like peptide more strongly.", "metric": "plddt", "entities": [{"type": "protein", "id": "CaM", "source_chain": "BB"}, {"type": "protein", "id": "MLCK_peptide", "source_chain": "T"}, {"type": "ion", "id": "calcium", "ccd": "CA", "count": 4}]},
            {"name": "calcium_only_state", "role": "calcium pocket sanity state", "objective": "EF-hand loops should remain compatible with calcium coordination without peptide.", "metric": "plddt", "entities": [{"type": "protein", "id": "CaM", "source_chain": "BB"}, {"type": "ion", "id": "calcium", "ccd": "CA", "count": 4}]},
        ],
        "multistate_regions": {"efhand_calcium_loops": loop_nodes, "cam_scaffold_core": scaffold_nodes, "target_peptide": ["MLCK_peptide_binding_surface"]},
        "multistate_objectives": [
            {"name": "calcium_state_peptide_interface_on", "type": "interface_on", "state": "calcium_peptide_state", "pair": ["CaM", "MLCK_peptide"], "left_region": "efhand_calcium_loops", "right_region": "MLCK_peptide_binding_surface", "weight": 1.05, "contact_target": 42, "residue_pair_target": 12, "coverage_target": 0.25, "off_target_penalty_weight": 0.10, "off_target_contact_tolerance": 110},
            {"name": "apo_state_peptide_interface_off", "type": "interface_off", "state": "apo_peptide_state", "pair": ["CaM", "MLCK_peptide"], "left_region": "efhand_calcium_loops", "weight": 0.95, "contact_target": 32, "residue_pair_target": 8, "off_target_contact_tolerance": 55},
            {"name": "calcium_vs_apo_peptide_delta", "type": "interface_delta", "positive_state": "calcium_peptide_state", "negative_state": "apo_peptide_state", "positive_pair": ["CaM", "MLCK_peptide"], "negative_pair": ["CaM", "MLCK_peptide"], "positive_left_region": "efhand_calcium_loops", "positive_right_region": "MLCK_peptide_binding_surface", "negative_left_region": "efhand_calcium_loops", "weight": 1.15, "direction": "increase", "min_delta": 0.10, "target_delta": 0.40, "contact_delta_target": 25},
            {"name": "efhand_acidic_calcium_pocket", "type": "ligand_pocket_pharmacophore", "state": "calcium_only_state", "pair": ["CaM", "calcium"], "left_region": "efhand_calcium_loops", "required_classes": {"acidic": 0.32, "polar": 0.45}, "weight": 0.95, "contact_target": 24, "residue_pair_target": 8, "coverage_target": 0.25},
            {"name": "efhand_loop_confidence_floor", "type": "region_confidence_floor", "states": ["apo_peptide_state", "calcium_peptide_state", "calcium_only_state"], "region": "efhand_calcium_loops", "metric": "plddt_min", "floor": 45, "target": 72, "weight": 0.55},
            {"name": "state_confidence", "type": "confidence", "states": ["apo_peptide_state", "calcium_peptide_state", "calcium_only_state"], "metric": "plddt", "weight": 0.45},
        ],
        "mutation_policy": {"preferred_edit_order": loop_nodes + scaffold_nodes[:3], "always_open_segments": loop_nodes, "conditionally_open_segments": scaffold_nodes[:3], "generally_frozen": scaffold_nodes[3:]},
        "case_information_needed": ["Exact target peptide or protein readout for the calcium sensor.", "Allowed edits in EF-loop acidic residues versus scaffold helices.", "Whether the goal is calcium affinity tuning, peptide affinity switching, or both."],
    }
    add_benchmark_start(
        state,
        case_id="calcium_efhand_switch",
        start_type="efhand_loop_chemistry_neutralized_seed",
        reference_binder_sequence=CALMODULIN_SEQ,
        changed_nodes=loop_nodes,
        rationale="Native calmodulin is already a calcium-gated peptide binder; the executable start neutralizes EF-hand loop chemistry so recovery of calcium/peptide behavior is measurable.",
    )
    regions = [
        {"name": "efhand_calcium_coordination_loops", "role": "retune acidic/polar Ca2+ pocket loops without losing EF-hand geometry", "position": 1, "bind_to": loop_nodes, "secondary_structure": "loop", "priority_boost": 1.45, "mutation_rate": 0.075, "max_mutations_per_step": 6, "operator_phase": "explore", "large_jump": True, "mutation_ops": {"point": 0.32, "block": 0.16, "site_resample": 0.24, "segment_mutagenesis": 0.18, "motif_graft": 0.06, "swap": 0.04}, "favored_residues": ["D", "E", "N", "Q", "S", "T", "G"], "disfavored_residues": ["C", "W", "F"], "policy_weight": 0.95},
        {"name": "cam_lobe_scaffold_guardrail", "role": "sparse helix/scaffold support for calcium-gated peptide binding", "position": 2, "bind_to": scaffold_nodes[:4], "secondary_structure": "helix", "priority_boost": 0.45, "mutation_rate": 0.018, "max_mutations_per_step": 1, "operator_phase": "stabilize", "mutation_ops": {"point": 0.92, "block": 0.03, "swap": 0.05}, "favored_residues": ["A", "L", "I", "V", "S", "T", "N", "Q", "E", "K"], "disfavored_residues": ["C", "P"], "policy_weight": 0.30},
    ]
    block = generic_block("calcium EF-hand switch", ["CaM_EFhand_domain"], regions, {name: "loop" for name in loop_nodes})
    sheet = case_sheet(
        "calcium_efhand_switch",
        "Design a CaM/EF-hand-like switch where calcium-loaded state binds MLCK-like peptide better than apo state.",
        ["Calcium-peptide interface increases.", "Apo-peptide interface stays weak.", "EF-hand loops preserve acidic/polar calcium pocket chemistry.", "Scaffold remains confident."],
        ["Peptide binds equally in apo state.", "Calcium pocket loses acidic/polar chemistry.", "Switch arises from unfolding or clashes.", "Scaffold helices destabilize."],
        {"calcium_peptide_interface": {"contacts": 42, "residue_pairs": 12, "coverage": 0.25}, "calcium_vs_apo_delta": {"min_delta": 0.10, "target_delta": 0.40, "contact_delta_target": 25}, "efhand_pocket": {"acidic_fraction": 0.32, "polar_fraction": 0.45}, "loop_confidence": {"hard_floor": 45, "target": 72}},
        state["case_information_needed"],
        benchmark_start=state["benchmark_start"],
    )
    write_case("calcium_efhand_switch", "Calcium EF-hand peptide-binding switch", "ion_gated_peptide_binding_switch", state, generic_initial("calcium_efhand_switch", block), config_yaml("calcium_efhand_switch", "calcium EF-hand switch", ", ".join(loop_nodes + scaffold_nodes[:4])), sheet)


def upgrade_tetr() -> None:
    path = CASES_ROOT / "tetr_dopamine" / "design_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    reference_binder_sequence = state["binder"].get("source_sequence") or sequence_from_binder(state["binder"])
    for segment in state["binder"].get("ted_ligand_dimer_segments", []):
        if len(segment) >= 3 and segment[0] == "ligand_pocket_core":
            segment[2] = neutralize_segment(segment[2])
            break
    state.setdefault("design_points", {})
    state["design_points"]["initial_seed_policy"] = (
        "The executable TetR seed preserves the native HTH, dimer scaffold, and allosteric topology, "
        "but starts from a dopamine-naive ligand pocket with hydrophobic/aromatic/charged chemistry "
        "neutralized. The native TetR-like source sequence is reference metadata only."
    )
    add_benchmark_start(
        state,
        case_id="tetr_dopamine",
        start_type="dopamine_pocket_neutralized_seed",
        reference_binder_sequence=reference_binder_sequence,
        changed_nodes=["ligand_pocket_core"],
        rationale=(
            "TetR should not start from a fully optimized ligand pocket. The initial executable pocket is "
            "chemically softened while the DNA-binding head and dimer core stay intact, giving the evaluator "
            "room to reward dopamine-pocket recovery plus apo/holo DNA-release behavior."
        ),
    )
    objectives = state.get("multistate_objectives", [])
    names = {obj.get("name") for obj in objectives if isinstance(obj, dict)}
    additions = [
        {"name": "dopamine_pocket_pharmacophore", "type": "ligand_pocket_pharmacophore", "state": "dopamine_bound_state_B", "pair": ["TetR_dimer", "dopamine"], "left_region": "ligand_pocket", "required_classes": {"acidic": 0.18, "polar": 0.35, "aromatic": 0.18}, "weight": 0.85, "contact_target": 18, "residue_pair_target": 5, "coverage_target": 0.12},
        {"name": "apo_vs_holo_dna_release_delta", "type": "interface_delta", "positive_state": "tetO_bound_state_A", "negative_state": "dopamine_tetO_competition_state_C", "positive_pair": ["TetR_dimer", ["TetO_plus", "TetO_minus"]], "negative_pair": ["TetR_dimer", ["TetO_plus", "TetO_minus"]], "positive_left_region": "dna_binding_head", "negative_left_region": "dna_binding_head", "weight": 1.1, "direction": "decrease", "min_delta": 0.10, "target_delta": 0.42, "contact_delta_target": 35},
        {"name": "allosteric_pathway_confidence_floor", "type": "region_confidence_floor", "states": ["tetO_bound_state_A", "dopamine_bound_state_B", "dopamine_tetO_competition_state_C"], "region": "allosteric_pathway", "metric": "plddt_min", "floor": 42, "target": 70, "weight": 0.45},
    ]
    for obj in additions:
        if obj["name"] not in names:
            objectives.append(obj)
    state["multistate_objectives"] = objectives
    state.setdefault("case_information_needed", [])
    state["case_information_needed"] = list(dict.fromkeys(state["case_information_needed"] + [
        "Residue-level dopamine pocket pharmacophore positions from a reference TetR-like pocket.",
        "Exact catecholamine selectivity panel if dopamine-specific behavior is required.",
    ]))
    write_json(path, state)
    sheet_path = CASES_ROOT / "tetr_dopamine" / "case_sheet.json"
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    sheet.setdefault("objective_thresholds", {})
    sheet["objective_thresholds"]["dopamine_pocket_pharmacophore"] = {"acidic_fraction": 0.18, "polar_fraction": 0.35, "aromatic_fraction": 0.18}
    sheet["objective_thresholds"]["apo_vs_holo_dna_release_delta"] = {"min_delta": 0.10, "target_delta": 0.42, "contact_delta_target": 35}
    sheet["benchmark_start"] = state["benchmark_start"]
    write_json(sheet_path, sheet)


def main() -> int:
    build_cd25_selectivity()
    build_pdl1_selectivity()
    build_proteor1_cdr_mask()
    build_pdz()
    build_calcium()
    upgrade_tetr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
