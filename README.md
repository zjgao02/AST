# ASTevolve

ASTevolve is the current protein-design search workspace for:

- `tetr_dopamine`: TetR dopamine-responsive redesign.
- `cd25_scfv`: CD25-targeting scFv optimization.

The current outer loop uses OpenEvolve to edit each case's `EVOLVE-BLOCK`.
The editable design layer is the AST layout/topology DSL: domains, design
regions, nodes, motifs, site anchors, and region-level mutation priors.
Runtime choices such as outer iterations, inner iterations, MCTS settings,
Protenix, ProGen, KB paths, and score weights are locked outside the
`EVOLVE-BLOCK` and are submitted through launcher parameters or environment
variables.

Full run standard: `docs/RUN_QUICK_START.md`.
Linux migration guide: `docs/LINUX_MIGRATION.md`.
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

Windows PowerShell uses the same profile/stage idea:

```powershell
.\scripts\submit_ast_run.ps1 -Case tetr_dopamine -Profile formal -Stage formal -OuterIterations 200 -InnerIterations 240 -RunName tetr_formal_001
```

Recommended formal-run gate:

```text
assets -> preview -> inner-smoke -> outer
```
