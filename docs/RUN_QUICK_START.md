# ASTevolve Run Quick Start

This document is the current run/submission standard for ASTevolve case runs.
It covers the prepared active cases:

- `tetr_dopamine`: TetR dopamine-responsive redesign.
- `cd25_scfv`: CD25-targeting scFv optimization.
- `cd25_scfv_selectivity`: CD25 epitope binding with IL2RB/CD122 decoy rejection.
- `pdl1_scfv_selectivity`: PD-L1 epitope binding with PD-L2 and PD-1 decoy rejection.
- `proteor1_cdr_mask`: Proteo-R1-style masked CDR re-evolution.
- `pdz_peptide_selectivity`: PDZ target peptide versus decoy peptide selectivity.
- `calcium_efhand_switch`: calcium-gated EF-hand peptide-binding switch.

The intended split is:

- OpenEvolve outer loop edits only the case `initial_program.py` `EVOLVE-BLOCK`,
  mainly `strategy["layout_plan"]`: domain order, design regions, nodes, motifs,
  site anchors, region-level mutation priors.
- Runtime settings are submitted through the launcher or environment variables:
  outer iterations, inner iterations, Protenix on/off, external KB on/off,
  retrieval on/off, ProGen weight, run name, and artifact location.
- Do not let LLM diffs choose global optimizer settings such as `iterations`,
  `mcts_*`, model paths, KB paths, or `score_config`. The two active cases now
  lock those fields outside the `EVOLVE-BLOCK`.

## 1. Recommended Launcher

Linux/bash:

```bash
bash scripts/submit_ast_run.sh --case tetr_dopamine --profile smoke --stage preview
```

Windows/PowerShell:

```powershell
.\scripts\submit_ast_run.ps1 -Case tetr_dopamine -Profile smoke -Stage preview
```

The launcher runs from the project root and calls the existing project entry
points:

- `scripts/check_assets.py`
- `cases/<case>/initial_program.py`
- `scripts/smoke_case.py`
- `openevolve/openevolve-run.py`

For detached GPU jobs on the Linux cluster, use:

```bash
CASE=pdl1_scfv_selectivity STAGE=outer PROFILE=formal RUN_NAME=pdl1_full_001 \
OUTER_ITERATIONS=120 INNER_ITERATIONS=720 TIME=48:00:00 \
bash scripts/submit_ast_gpu.sh
```

`submit_ast_gpu.sh` writes the Slurm batch file and logs under
`artifacts/runs/logs/`; it defaults to `sbatch`, so the job continues after the
terminal disconnects.

## 2. Cases

| Case | Entry program | OpenEvolve config | Formal inner default |
|---|---|---|---:|
| `tetr_dopamine` | `cases/tetr_dopamine/initial_program.py` | `cases/tetr_dopamine/config.yaml` | 240 |
| `cd25_scfv` | `cases/cd25_scfv/initial_program.py` | `cases/cd25_scfv/config.yaml` | 1200 |
| `cd25_scfv_selectivity` | `cases/cd25_scfv_selectivity/initial_program.py` | `cases/cd25_scfv_selectivity/config.yaml` | 1200 |
| `pdl1_scfv_selectivity` | `cases/pdl1_scfv_selectivity/initial_program.py` | `cases/pdl1_scfv_selectivity/config.yaml` | 1200 |
| `proteor1_cdr_mask` | `cases/proteor1_cdr_mask/initial_program.py` | `cases/proteor1_cdr_mask/config.yaml` | 1200 |
| `pdz_peptide_selectivity` | `cases/pdz_peptide_selectivity/initial_program.py` | `cases/pdz_peptide_selectivity/config.yaml` | 360 |
| `calcium_efhand_switch` | `cases/calcium_efhand_switch/initial_program.py` | `cases/calcium_efhand_switch/config.yaml` | 360 |

Both configs currently use `max_iterations: 200` as their default outer-loop
budget. The launcher can override it with `--outer-iterations` or
`-OuterIterations`.

## 3. Profiles

| Profile | Purpose | Outer iterations | Inner iterations | Protenix | External KB | Retrieval | ProGen |
|---|---|---:|---:|---|---|---|---|
| `smoke` | Fast sanity check | 1 | 1 | off | off | off | 0.0 |
| `cheap` | Search logic check without structure cost | 10 | 10 | off | on | case default | case default |
| `formal` | Main exploratory run | 200 | case default | on | on | case default | case default |

Case defaults:

- TetR: retrieval off, ProGen weight `0.5`, formal inner iterations `240`.
- scFv/antibody-style cases: retrieval on, ProGen weight `1.0`, formal inner iterations `1200`.
- PDZ and EF-hand single-domain cases: retrieval off, ProGen weight `0.6`, formal inner iterations `360`.

Any profile value can be overridden at submission time.

## 4. Stages

| Stage | What it does | Needs LLM key |
|---|---|---|
| `assets` | Check case KB/model asset paths without loading models | no |
| `preview` | Compile the case and print layout/masks/config summary | no |
| `inner-smoke` | Run one evaluator call with chosen inner-loop settings | no |
| `outer` | Run OpenEvolve outer loop | yes |
| `formal` | Run `assets`, `preview`, `inner-smoke`, then `outer` | yes |
| `all` | Same as `formal` | yes |

The normal gate before a formal run is:

```text
assets -> preview -> inner-smoke -> outer
```

## 5. Main Submission Parameters

Bash names:

```bash
--case CASE_ID
--stage assets|preview|inner-smoke|outer|formal|all
--profile smoke|cheap|formal
--outer-iterations N
--inner-iterations N
--use-protenix auto|on|off
--external-kb auto|on|off
--external-retrieval auto|on|off
--progen-weight FLOAT
--run-name NAME
--conda-env pytorch
--no-conda
--dry-run
```

PowerShell names:

```powershell
-Case CASE_ID
-Stage assets|preview|inner-smoke|outer|formal|all
-Profile smoke|cheap|formal
-OuterIterations N
-InnerIterations N
-Protenix auto|on|off
-ExternalKb auto|on|off
-ExternalRetrieval auto|on|off
-ProgenWeight FLOAT
-RunName NAME
-CondaEnv pytorch
-NoConda
-DryRun
```

## 5.1 Artifact Layout

Formal run outputs should be under:

```text
artifacts/runs/<case>/<run_name>/
```

Legacy or fallback one-off outputs that use a UUID as the artifact id are
automatically placed under:

```text
artifacts/transient/<uuid>/
```

This keeps the top-level `artifacts/` directory from accumulating many UUID
folders. Override the transient folder name if needed:

```bash
export ASTEVOLVE_TRANSIENT_ARTIFACT_DIR=transient
```

## 6. Current Bash Commands

Fast preview for TetR:

```bash
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile smoke \
  --stage preview
```

Fast inner-loop smoke for TetR, no expensive models:

```bash
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile smoke \
  --stage inner-smoke \
  --run-name tetr_smoke_001
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

Fast preview for scFv:

```bash
bash scripts/submit_ast_run.sh \
  --case cd25_scfv \
  --profile smoke \
  --stage preview
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

Cheap search check with KB and ProGen but no Protenix:

```bash
bash scripts/submit_ast_run.sh \
  --case tetr_dopamine \
  --profile cheap \
  --stage inner-smoke \
  --use-protenix off \
  --external-kb on \
  --external-retrieval off \
  --progen-weight 0.5 \
  --run-name tetr_cheap_001
```

Dry-run the exact command without executing:

```bash
bash scripts/submit_ast_run.sh \
  --case cd25_scfv \
  --profile formal \
  --stage outer \
  --dry-run
```

## 7. Direct Commands Without Launcher

Use these if you want to control everything manually.

TetR:

```bash
export ASTEVOLVE_LLM_API_KEY="..."
export ASTEVOLVE_CASE_ID=tetr_dopamine
export ASTEVOLVE_INNER_ITERATIONS=240
export ASTEVOLVE_ENABLE_PROTENIX=1
export ASTEVOLVE_ENABLE_EXTERNAL_KB=1
export ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL=0
export ASTEVOLVE_PROGEN_WEIGHT=0.5
export ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA=0
export ASTEVOLVE_PROTENIX_COMPLEX_CYCLE=1
export ASTEVOLVE_PROTENIX_COMPLEX_STEP=1
export ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE=1
export ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS=0
export ASTEVOLVE_MCTS_OUTPUT_DIR=artifacts/runs/tetr_dopamine/tetr_formal_001/inner
export ASTEVOLVE_PROTENIX_TMP=artifacts/runs/tetr_dopamine/tetr_formal_001/protenix_tmp

conda run -n pytorch python openevolve/openevolve-run.py \
  cases/tetr_dopamine/initial_program.py \
  evaluator.py \
  --config cases/tetr_dopamine/config.yaml \
  --iterations 200 \
  --output artifacts/runs/tetr_dopamine/tetr_formal_001/openevolve
```

scFv:

```bash
export ASTEVOLVE_LLM_API_KEY="..."
export ASTEVOLVE_CASE_ID=cd25_scfv
export ASTEVOLVE_INNER_ITERATIONS=1200
export ASTEVOLVE_ENABLE_PROTENIX=1
export ASTEVOLVE_ENABLE_EXTERNAL_KB=1
export ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL=1
export ASTEVOLVE_PROGEN_WEIGHT=1.0
export ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA=0
export ASTEVOLVE_PROTENIX_COMPLEX_CYCLE=1
export ASTEVOLVE_PROTENIX_COMPLEX_STEP=1
export ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE=1
export ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS=0
export ASTEVOLVE_MCTS_OUTPUT_DIR=artifacts/runs/cd25_scfv/scfv_formal_001/inner
export ASTEVOLVE_PROTENIX_TMP=artifacts/runs/cd25_scfv/scfv_formal_001/protenix_tmp

conda run -n pytorch python openevolve/openevolve-run.py \
  cases/cd25_scfv/initial_program.py \
  evaluator.py \
  --config cases/cd25_scfv/config.yaml \
  --iterations 200 \
  --output artifacts/runs/cd25_scfv/scfv_formal_001/openevolve
```

## 8. Model And Runtime Environment Variables

The launchers set these for each run:

| Variable | Meaning |
|---|---|
| `ASTEVOLVE_INNER_ITERATIONS` | Inner-loop MCTS/SA budget per evaluator call |
| `ASTEVOLVE_ENABLE_PROTENIX` | `1` enables structure evaluation; `0` skips it |
| `ASTEVOLVE_PROGEN_WEIGHT` | ProGen sequence prior weight; `0.0` disables it |
| `ASTEVOLVE_ENABLE_EXTERNAL_KB` | Enables case external KB prior |
| `ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL` | Enables embedding retrieval from the external KB |
| `ASTEVOLVE_MCTS_OUTPUT_DIR` | Inner-loop artifacts for this run |
| `ASTEVOLVE_PROTENIX_TMP` | Protenix temp/output root for this run |
| `ASTEVOLVE_LLM_API_KEY` | Required for OpenEvolve outer-loop LLM calls |

Optional Protenix controls:

| Variable | Default used by launcher | Meaning |
|---|---:|---|
| `ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA` | `0` | Disable/enable MSA use |
| `ASTEVOLVE_PROTENIX_COMPLEX_CYCLE` | `1` | Protenix cycles for complex prediction |
| `ASTEVOLVE_PROTENIX_COMPLEX_STEP` | `1` | Protenix steps |
| `ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE` | `1` | Number of samples |
| `ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS` | `0` | Use Protenix defaults instead of cheap overrides |
| `ASTEVOLVE_PROTENIX_COMPLEX_TIMEOUT` | unset | Per-call timeout override |

For expensive final validation, set higher Protenix values before launching, for
example:

```bash
export ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS=1
export ASTEVOLVE_PROTENIX_COMPLEX_TIMEOUT=1800
```

## 9. What To Submit For A Formal Run

A formal run submission should record:

- case id: `tetr_dopamine` or `cd25_scfv`
- run name
- launcher command or direct command
- profile and stage
- outer iterations
- inner iterations
- model switches: Protenix, ProGen weight, external KB, retrieval
- any non-default Protenix settings
- `ASTEVOLVE_LLM_API_KEY` presence, but never the key value
- asset check result
- preview output
- inner-smoke result
- final OpenEvolve output directory

Generated artifacts should stay under `artifacts/` or the configured
`ASTEVOLVE_ARTIFACT_ROOT`; they should not be committed with source changes.
