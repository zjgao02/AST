# Case Starting Seeds

ASTevolve cases should not start from the best known or native solution for the
editable design region. Each case stores two separate sequences:

- `benchmark_start.initial_seed_sequence`: the executable sequence used by the
  inner loop and OpenEvolve evaluator.
- `benchmark_start.reference_oracle_sequence`: metadata only, used to document
  what was degraded or masked. It is not used as the starting program.

The current prepared cases keep the broad fold or framework fixed while
degrading the regions that the search is expected to improve.

| Case | Executable start | Changed nodes | Approx. identity to reference |
| --- | --- | --- | --- |
| `tetr_dopamine` | Dopamine-naive TetR pocket seed; HTH and dimer scaffold preserved | `ligand_pocket_core` | 0.87 |
| `cd25_scfv_selectivity` | Neutralized CDR seed on the CD25 scFv framework | all VH/VL CDRs | 0.85 |
| `pdl1_scfv_selectivity` | Generic neutralized CDR seed; no PD-L1 oracle CDRs | all VH/VL CDRs | 0.85 |
| `proteor1_cdr_mask` | Proteo-R1 masked CDR spans filled with neutral motifs | all VH/VL CDRs | 0.84 |
| `pdz_peptide_selectivity` | Native-length PDZ scaffold with neutralized peptide-contact groove | PDZ groove nodes | 0.89 |
| `calcium_efhand_switch` | Calmodulin-length scaffold with EF-hand loop chemistry neutralized | EF-hand loops | 0.82 |

This setup makes long runs easier to interpret: improvement should come from
recovering or reinventing local binding/allosteric features, not from starting at
a known positive control.
