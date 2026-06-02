# CD25 scFv Case

This folder contains the CD25-targeting scFv design case.

```text
case.json
initial_program.py      # OpenEvolve strategy entry for this case
config.yaml             # OpenEvolve run config for this case
design_state.json       # CD25/scFv AST state
memory.yaml             # case-specific internal memory
```

Run from the project root with the `pytorch` conda environment:

```powershell
$env:ASTEVOLVE_LLM_API_KEY="..."
conda run -n pytorch python openevolve\openevolve-run.py cases\cd25_scfv\initial_program.py evaluator.py --config cases\cd25_scfv\config.yaml
```
