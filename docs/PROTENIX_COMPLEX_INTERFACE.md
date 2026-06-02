# Protenix Complex Interface

`astevolve/apis/protenix.py` now supports both the original protein-only multichain path
and a general complex path for state-specific evaluation.

## Protein-Only Compatibility

Existing callers can keep using:

```python
from astevolve.apis.protenix import run_protenix_confidence_multichain

out = run_protenix_confidence_multichain(
    pred_name="protein_pair",
    chains=[("A", seq_a), ("B", seq_b)],
    conda_env="pytorch",
)
```

## General Complex Input

Use `run_protenix_confidence_complex` for protein-ligand, protein-DNA,
protein-RNA, ions, and mixed complexes:

```python
from astevolve.apis.protenix import run_protenix_confidence_complex

out = run_protenix_confidence_complex(
    pred_name="protein_ligand_state",
    entities=[
        {"type": "protein", "id": "A", "sequence": protein_seq},
        {"type": "ligand", "id": "L", "smiles": "CCO"},
    ],
    conda_env="pytorch",
    model_name="protenix_mini_esm_v0.5.0",
)
```

Supported entity forms:

```python
{"type": "protein", "id": "A", "sequence": "...", "count": 1}
{"type": "dna", "id": "D1", "sequence": "ATGC"}
{"type": "rna", "id": "R1", "sequence": "AUGC"}
{"type": "ligand", "id": "L", "smiles": "..."}
{"type": "ligand", "id": "L", "ccd": "ATP"}
{"type": "ligand", "id": "L", "file": "path/to/ligand.sdf"}
{"type": "ion", "id": "MG", "ion": "MG"}
```

For CCD ligands, the wrapper automatically converts `"ATP"` to `"CCD_ATP"`.
For file ligands, it automatically converts a path to `"FILE_<path>"`.

## Low-Cost Smoke Settings

For testing input acceptance rather than serious structure scoring:

```python
out = run_protenix_confidence_complex(
    pred_name="smoke",
    entities=[...],
    conda_env="pytorch",
    use_msa=False,
    cycle=1,
    step=1,
    sample=1,
    use_default_params=False,
    timeout=180,
)
```

These settings are intentionally cheap and should not be used for final
biological scoring.

## Returned Shape

The complex API returns the same high-level structure used by the protein-only
adapter, with extra complex metadata:

```python
{
    "metrics": {"plddt": ..., "ptm": ..., "iptm": ..., "ranking_score": ...},
    "chain_metrics": {...},
    "residue_plddt": {...},
    "entities": [...],
    "input_json": ".../input.json",
    "summary_json": "...summary_confidence_sample_0.json",
    "out_dir": ".../output",
}
```

## ASTevolve Registry

The model registry exposes:

```python
from astevolve.models.registry import run_structure_confidence_complex

out = run_structure_confidence_complex(
    provider="protenix",
    pred_name="state_a",
    entities=[...],
)
```

