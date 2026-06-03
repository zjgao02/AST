# ASTevolve Cases

Each case folder is self-contained for outer-loop runs: it owns the biological
state, memory store, OpenEvolve entry program, and OpenEvolve config.

```text
cases/
  cd25_scfv/
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
