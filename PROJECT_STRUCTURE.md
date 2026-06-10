# ASTevolve Project Structure

The root directory now contains only thin entry points, shared package code,
case folders, data, and the pruned vendored OpenEvolve runtime.

```text
evaluator.py                # thin OpenEvolve evaluator shim
requirements.txt

astevolve/
  core/                     # AST primitives and constraint terms
    protein_lang.py
    constraints.py
  search/
    inner_opt.py            # shared inner-loop MCTS/SA optimizer
  evaluation/
    open_evolve.py          # shared OpenEvolve evaluator implementation
  apis/                     # model/backend adapters
    progen.py
    protenix.py
    esmfold2.py
  knowledge/
    providers/
      sabdab_antibody.py    # SAbDab/CDR external KB provider
  cases/                    # case registry
  models/                   # model interface registry
  metrics/                  # structure metric summaries
  runtime/                  # current-case helpers
  semantic_graph/           # Protein Semantic Graph summaries for LLM feedback

engine/                     # strategy -> AST/input compilation and memory update
cases/                      # case-specific state, memory, strategy entry, config
data/                       # KB source artifacts
artifacts/                  # generated outputs and local caches, ignored by git
docs/                       # shared interface/metric notes
openevolve/                 # pruned vendored OpenEvolve runtime
```

Case folders are self-contained for outer-loop runs:

```text
cases/cd25_scfv/
  case.json
  initial_program.py
  config.yaml
  design_state.json
  memory.yaml
  INNER_LOOP_PROCESS.md

cases/tetr_dopamine/
  case.json
  initial_program.py
  config.yaml
  design_state.json
  memory.yaml
```

Run CD25 scFv:

```powershell
$env:ASTEVOLVE_LLM_API_KEY="..."
conda run -n <ast-or-pytorch> python openevolve\openevolve-run.py cases\cd25_scfv\initial_program.py evaluator.py --config cases\cd25_scfv\config.yaml
```

Run TetR dopamine:

```powershell
$env:ASTEVOLVE_LLM_API_KEY="..."
conda run -n <ast-or-pytorch> python openevolve\openevolve-run.py cases\tetr_dopamine\initial_program.py evaluator.py --config cases\tetr_dopamine\config.yaml
```

The vendored `openevolve/` directory intentionally keeps only `openevolve-run.py`,
`LICENSE`, and the `openevolve/` Python package needed by the CLI runtime.
Examples, tests, visualizer assets, docs, and its nested git checkout were removed.