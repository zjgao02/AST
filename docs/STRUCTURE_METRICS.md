# Structure Metrics In ASTevolve

The inner loop now separates structure confidence into explicit, reportable
metrics instead of treating all structure feedback as one pLDDT-like number.

## Metrics Collected

From Protenix or any compatible structure backend:

```text
plddt
ptm
iptm
gpde
ranking_score
has_clash
disorder
```

From residue-level confidence:

```text
chain_plddt
node_plddt
node_plddt_mean
node_plddt_min
low_confidence_nodes
```

From predicted CIF coordinates:

```text
interface_contact_count
interface_residue_pair_count
interface_plddt_mean
interface_plddt_min
clash_count
contact_examples
```

## DockQ

True DockQ is only available when a native/reference complex is provided. For
ordinary de novo evolution runs, ASTevolve only reports:

```text
dockq.available = false
```

No DockQ-like value is included in the default evaluator score, because it can
be confused with true DockQ. Use interface contact count, interface pLDDT, ipTM,
and clash count as no-reference diagnostics until a native/reference complex is
available.

## Inner Loop Artifacts

The best structure-evaluated candidates contain:

```text
confidence_metrics
structure_metrics
chain_plddt
node_plddt
protenix_summary_json
protenix_out_dir
```

For MCTS runs, top-K candidates that received expensive structure evaluation
are also written to:

```text
inner_loop/structure_evaluated_variants.json
```

The round summary includes:

```text
search_artifacts.structure_evaluation_summary
```

## Evaluator Metrics

`evaluator.py` now exposes these OpenEvolve metrics:

```text
combined_score
struct_score
fast_score
plddt
plddt_delta
ptm
iptm
ranking_score
interface_plddt_mean
interface_contact_count
interface_residue_pair_count
clash_count
node_plddt_mean
node_plddt_min
```
