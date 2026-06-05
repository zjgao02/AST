# ASTevolve Cases

Each case folder is self-contained for outer-loop runs: it owns the biological
state, memory store, OpenEvolve entry program, and OpenEvolve config.

```text
cases/
  calcium_efhand_switch/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
  cd25_scfv/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
  cd25_scfv_selectivity/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
  pdl1_scfv_selectivity/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
  pdz_peptide_selectivity/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
  proteor1_cdr_mask/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
  tetr_dopamine/
    case.json
    initial_program.py
    config.yaml
    design_state.json
    memory.yaml
```

Root-level Python files are shared runtime modules, not case-specific strategy
entry files.

The currently prepared mechanistic/typed-objective cases are:

- `tetr_dopamine`: TetR dopamine allosteric switch with apo DNA-on, dopamine holo, and dopamine/DNA competition states.
- `cd25_scfv_selectivity`: CD25 epitope-binding scFv with IL2RB/CD122 decoy rejection.
- `pdl1_scfv_selectivity`: PD-L1 epitope-binding scFv with PD-L2 and PD-1 off-target rejection.
- `proteor1_cdr_mask`: Proteo-R1-style masked H/L CDR re-evolution against the 8r9y demo antigen hotspot.
- `pdz_peptide_selectivity`: PDZ groove retuning for KQTSV target peptide over a KKAAA decoy.
- `calcium_efhand_switch`: CaM/EF-hand calcium-gated peptide-binding switch with apo-off and calcium-on states.
