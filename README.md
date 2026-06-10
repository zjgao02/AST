# ASTevolve

ASTevolve is the current protein-design search workspace for:

- `tetr_dopamine`: TetR dopamine-responsive redesign.
- `cd25_scfv`: CD25-targeting scFv optimization.
- `cd25_scfv_selectivity`: CD25 epitope scFv with IL2RB/CD122 decoy rejection.
- `pdl1_scfv_selectivity`: PD-L1 epitope scFv with PD-L2 and PD-1 decoy rejection.
- `proteor1_cdr_mask`: Proteo-R1-style masked CDR re-evolution.
- `pdz_peptide_selectivity`: PDZ groove peptide-selectivity design.
- `dlg1_pdz1_to_pdz2`: DLG1/SAP97 PDZ1-to-PDZ2 semantic-graph specificity transfer.
- `calcium_efhand_switch`: CaM/EF-hand calcium-gated peptide switch.

The current outer loop uses OpenEvolve to edit each case's `EVOLVE-BLOCK`.
The editable design layer is the AST layout/topology DSL: domains, design
regions, nodes, motifs, site anchors, and region-level mutation priors.
Runtime choices such as outer iterations, inner iterations, MCTS settings,
Protenix, ProGen, KB paths, and score weights are locked outside the
`EVOLVE-BLOCK` and are submitted through launcher parameters or environment
variables.

Full run standard: `docs/RUN_QUICK_START.md`.
Linux migration guide: `docs/LINUX_MIGRATION.md`.
Case starting seeds: `docs/CASE_STARTING_SEEDS.md`.
Weight/download manifest: `model_weights/WEIGHTS_MANIFEST.txt`.

## Quick Commands

Preview TetR:

```bash
bash scripts/submit_ast_run.sh --case tetr_dopamine --profile smoke --stage preview
```

Preview scFv:

```bash
bash scripts/submit_ast_run.sh --case cd25_scfv --profile smoke --stage preview
```

Preview every prepared case:

```bash
for case in tetr_dopamine cd25_scfv_selectivity pdl1_scfv_selectivity proteor1_cdr_mask pdz_peptide_selectivity dlg1_pdz1_to_pdz2 calcium_efhand_switch; do
  bash scripts/submit_ast_run.sh --case "$case" --profile smoke --stage preview --no-conda
done
```

Formal TetR exploratory run:

```bash
export ASTEVOLVE_LLM_API_KEY="..."
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile formal \
  --stage formal \
  --outer-iterations 200 \
  --inner-iterations 240 \
  --run-name tetr_formal_001
```

Formal scFv exploratory run:

```bash
export ASTEVOLVE_LLM_API_KEY="..."
bash scripts/submit_ast_run.sh \
  --case cd25_scfv \
  --profile formal \
  --stage formal \
  --outer-iterations 200 \
  --inner-iterations 1200 \
  --run-name scfv_formal_001
```

Detached Slurm GPU submission on the Linux cluster:

```bash
CASE=pdl1_scfv_selectivity STAGE=outer PROFILE=formal RUN_NAME=pdl1_full_001 \
OUTER_ITERATIONS=120 INNER_ITERATIONS=720 TIME=48:00:00 \
bash scripts/submit_ast_gpu.sh
```

Windows PowerShell uses the same profile/stage idea:

```powershell
.\scripts\submit_ast_run.ps1 -Case tetr_dopamine -Profile formal -Stage formal -OuterIterations 200 -InnerIterations 240 -RunName tetr_formal_001
```

Recommended formal-run gate:

```text
assets -> preview -> inner-smoke -> outer
```
