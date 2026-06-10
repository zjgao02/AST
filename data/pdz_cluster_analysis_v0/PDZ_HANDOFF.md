# PDZ Protein Semantic Graph handoff

This thread built a first-stage PDZ case library for testing:

Protein Semantic Graph + OpenEvolve outer loop + inner-loop protein designer.

The project decision is to start with PDZ specificity redesign rather than scFv, because PDZ has clearer semantic nodes, cheaper complex evaluation, and cleaner evidence for node-level search.

## Core hypothesis

The goal is not just to show that AI can design proteins.

The goal is to test whether proteins can be represented as editable semantic graphs, and whether search at the semantic-node level is more effective and interpretable than direct residue/sequence search.

For PDZ, useful graph nodes are:

- Recognition Node: binding groove
- Specificity Node: alphaB helix / betaB strand region
- Allosteric Node: alpha3 helix or regulatory coupling region
- Stability Node: hydrophobic core

## Generated files

Main candidate library:

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_case_candidates_v0.csv`
  - 20 PDZ cases with representative PDB, peptide, priority, and task value.

Sequence metadata:

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_case_sequence_info_v0.csv`
  - 20 rows, one per PDZ case.
  - Includes PDZ construct sequence, peptide sequence, entity IDs, chain IDs, UniProt IDs, citation fields, and source URL.

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_case_sequences_v0.fasta`
  - 40 FASTA records.
  - Each case has one PDZ construct sequence and one peptide sequence.

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_case_sequence_info_raw_rcsb_v0.json`
  - Raw RCSB GraphQL metadata.

Downloaded representative PDB files:

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_structures\pdb\`
  - 20 `.pdb` coordinate files from RCSB.

Clustering and pair-selection outputs:

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_cluster_analysis_v0\pdz_case_cluster_features_v0.csv`
  - Per-case sequence / structure / binding motif features.

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_cluster_analysis_v0\pdz_case_pairwise_similarity_v0.csv`
  - 190 pairwise comparisons.

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_cluster_analysis_v0\pdz_specificity_transfer_pairs_v0.csv`
  - 25 algorithmic candidate transfer pairs.

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_cluster_analysis_v0\pdz_transfer_tasks_v0.csv`
  - 8 manually curated, task-shaped A-to-B design candidates.

- `C:\Users\13058\Documents\AST\ast_patch\data\pdz_cluster_analysis_v0\pdz_case_clustering_report_v0.md`
  - Short report.

Reusable script:

- `C:\Users\13058\Documents\AST\ast_patch\scripts\analyze_pdz_clusters.py`
  - Recomputes sequence identity, motif classes, C-alpha shape RMSD proxy, pairwise similarity, and transfer-task candidates.

## Current best tasks

Recommended first task:

1. `DLG1_PDZ1_to_DLG1_PDZ2`
   - Source: DLG1/SAP97 PDZ1, PDB `3RL7`, peptide `RHSGSYLVTSV`
   - Target: DLG1/SAP97 PDZ2, PDB `2G2L`, peptide `SIPCMSHSSGMPLGATGL`
   - Sequence identity: 0.5051
   - C-alpha shape RMSD proxy: 1.357 A
   - Why: Same family/scaffold, cleanest specificity-transfer case.

Other high-value tasks:

2. `Erbin_to_TIP1`
   - Erbin PDZ, `1MFG`, peptide `EYLGLDVPV`, class II-like
   - TIP-1 PDZ, `3SFJ`, peptide `ANSRWPTSII`, class I-like
   - Good class-II to class-I transfer.

3. `Syntenin_to_Tamalin`
   - Syntenin-1 PDZ2, `1OBY`, peptide `TNEFYA`, class II-like
   - Tamalin/GRASP PDZ, `2EGN`, peptide `SSSSL`, class I-like
   - Strong motif contrast and close structure proxy.

4. `CAL_vs_TIP1_counterselection`
   - CAL/GOPC PDZ, `4E34`, peptide `ANSRWPTSII`
   - TIP-1 PDZ, `3SFJ`, same peptide `ANSRWPTSII`
   - Better for selectivity/counterselection than motif rewriting.

Useful but second-tier:

- `PICK1_to_CAL`
- `GRIP1_to_Afadin`
- `PSD95_to_Erbin`
- `CAL_to_DLG1_PDZ2`

## Important caveats

- The PDZ sequences are representative PDB construct sequences from RCSB, not necessarily full UniProt canonical sequences.
- They may contain expression tags, engineered residues, truncations, missing residues, or noncanonical residue placeholders.
- Structure clustering used a lightweight C-alpha shape RMSD proxy, not formal TM-align.
- Before final case selection, the next step should add:
  - literature mutation data,
  - formal structure alignment if available,
  - residue-to-node mapping for binding groove / alphaB / betaB / core,
  - evaluation definitions for positive peptide and negative peptide states.

## Recommended next action

Start with `DLG1_PDZ1_to_DLG1_PDZ2` and build an AST case definition:

- Source state: DLG1/SAP97 PDZ1 bound to APC peptide.
- Target state: DLG1/SAP97 PDZ2-like recognition of GluR-A peptide.
- Design objective:
  - improve target peptide binding score,
  - retain PDZ fold stability,
  - penalize loss of source scaffold integrity,
  - optionally include negative/counterselection peptides.

The first implementation task should convert `pdz_transfer_tasks_v0.csv` into a case folder under:

`C:\Users\13058\Documents\AST\ast_patch\cases\pdz_specificity_transfer\`

with an `initial_program.py`, `memory.yaml`, and config that define graph nodes and editable regions.
