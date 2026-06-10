# PDZ clustering and transfer-pair shortlist

Inputs:
- Cases: C:\Users\13058\Documents\AST\ast_patch\data\pdz_case_sequence_info_v0.csv
- Structures: C:\Users\13058\Documents\AST\ast_patch\data\pdz_structures\pdb

Method:
- Sequence: global pairwise identity over RCSB representative PDZ construct sequence.
- Structure: first-model C-alpha shape RMSD by PCA-initialized ICP; sequence-aligned RMSD is retained in the pairwise CSV as a diagnostic only.
- Binding motif: C-terminal peptide tail classified from p-2 and p0 residues.

Top transfer/counterselection pairs:

| rank | relation | A | motif A | B | motif B | seq id | CA shape RMSD | motif dist | direction |
|---:|---|---|---|---|---|---:|---:|---:|---|
| 1 | specificity_transfer | Erbin PDZ | class_II_X-Phi-X-Phi | TIP-1 PDZ | class_I_X-S/T-X-Phi | 0.4565 | 2.120 | 3 | Erbin PDZ (EYLGLDVPV, class_II_X-Phi-X-Phi) -> TIP-1 PDZ (ANSRWPTSII, class_I_X-S/T-X-Phi) |
| 2 | specificity_transfer | Syntenin-1 PDZ2 | class_II_X-Phi-X-Phi | Tamalin/GRASP PDZ | class_I_X-S/T-X-Phi | 0.3418 | 1.634 | 3 | Syntenin-1 PDZ2 (TNEFYA, class_II_X-Phi-X-Phi) -> Tamalin/GRASP PDZ (SSSSL, class_I_X-S/T-X-Phi) |
| 3 | specificity_transfer | GRIP1 PDZ6 | noncanonical_Cterm | CAL/GOPC PDZ | class_I_X-S/T-X-Phi | 0.3837 | 2.160 | 3 | GRIP1 PDZ6 (ATVRTYSC, noncanonical_Cterm) -> CAL/GOPC PDZ (ANSRWPTSII, class_I_X-S/T-X-Phi) |
| 4 | specificity_transfer | PICK1 PDZ | class_II_X-Phi-X-Phi | CAL/GOPC PDZ | class_I_X-S/T-X-Phi | 0.3571 | 1.709 | 2 | PICK1 PDZ (ESVKI, class_II_X-Phi-X-Phi) -> CAL/GOPC PDZ (ANSRWPTSII, class_I_X-S/T-X-Phi) |
| 5 | specificity_transfer | PICK1 PDZ | class_II_X-Phi-X-Phi | DLG1/SAP97 PDZ2 | class_I_X-S/T-X-Phi | 0.3605 | 2.275 | 3 | PICK1 PDZ (ESVKI, class_II_X-Phi-X-Phi) -> DLG1/SAP97 PDZ2 (SIPCMSHSSGMPLGATGL, class_I_X-S/T-X-Phi) |
| 6 | specificity_transfer | GRIP1 PDZ6 | noncanonical_Cterm | DLG1/SAP97 PDZ1 | class_I_X-S/T-X-Phi | 0.3021 | 1.967 | 3 | GRIP1 PDZ6 (ATVRTYSC, noncanonical_Cterm) -> DLG1/SAP97 PDZ1 (RHSGSYLVTSV, class_I_X-S/T-X-Phi) |
| 7 | specificity_transfer | GRIP1 PDZ6 | noncanonical_Cterm | DLG1/SAP97 PDZ2 | class_I_X-S/T-X-Phi | 0.3263 | 2.140 | 4 | GRIP1 PDZ6 (ATVRTYSC, noncanonical_Cterm) -> DLG1/SAP97 PDZ2 (SIPCMSHSSGMPLGATGL, class_I_X-S/T-X-Phi) |
| 8 | specificity_transfer | CAL/GOPC PDZ | class_I_X-S/T-X-Phi | DLG1/SAP97 PDZ2 | class_I_X-S/T-X-Phi | 0.3953 | 1.621 | 2 | CAL/GOPC PDZ (ANSRWPTSII, class_I_X-S/T-X-Phi) -> DLG1/SAP97 PDZ2 (SIPCMSHSSGMPLGATGL, class_I_X-S/T-X-Phi) |
| 9 | specificity_transfer | Erbin PDZ | class_II_X-Phi-X-Phi | Par-3 PDZ3 | class_I_X-S/T-X-Phi | 0.3804 | 2.508 | 3 | Erbin PDZ (EYLGLDVPV, class_II_X-Phi-X-Phi) -> Par-3 PDZ3 (DEDQHSQITKV, class_I_X-S/T-X-Phi) |
| 10 | specificity_transfer | GRIP1 PDZ6 | noncanonical_Cterm | Afadin/AF-6 PDZ | class_I_X-S/T-X-Phi | 0.3708 | 2.507 | 3 | GRIP1 PDZ6 (ATVRTYSC, noncanonical_Cterm) -> Afadin/AF-6 PDZ (LFSTEV, class_I_X-S/T-X-Phi) |
