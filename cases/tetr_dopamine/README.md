# TetR Dopamine Case

This case redesigns a TetR-family homodimer so that the ligand-binding/allosteric
domain recognizes dopamine instead of tetracycline while preserving a
DNA-bound TetO-compatible state.

## Biological Setup

```text
Designed protein:
  TetR monomer, evaluated as a homodimer

State A:
  TetR dimer + TetO DNA duplex

State B:
  TetR dimer + dopamine ligand
```

TetR sequence length: 207 aa.

Domain annotations from the user-provided InterPro/AlphaFold view:

```text
1-66: homeodomain-like HTH DNA-binding domain
68-205: TED consensus ligand/dimerization domain
```

TetO plus strand:

```text
TCCCTATCAGTGATAGAGA
```

Dopamine SMILES:

```text
NCCc1ccc(O)c(O)c1
```

## Main Mutable Nodes

```text
ligand_pocket_core
allosteric_hinge
relay_helix
domain_connector
HTH_exit_loop
dimerization_interface
HTH_recognition_helix
```

The intended first-stage design objective is not to prove in vitro function,
but to search for pocket/relay mutations that make the dopamine-bound complex
structurally plausible while avoiding loss of the TetO-bound state.


## Run

```powershell
$env:ASTEVOLVE_LLM_API_KEY="..."
conda run -n pytorch python openevolve\openevolve-run.py cases\tetr_dopamine\initial_program.py evaluator.py --config cases\tetr_dopamine\config.yaml
```
