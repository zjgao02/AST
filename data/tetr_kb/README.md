# TetR InterPro KB

This directory is the TetR-family allosteric transcription factor external
knowledge repository for ASTevolve.

It is intentionally separate from the broad Magneton/InterPro cache and from
the antibody KB. The records here are generated from InterPro/Pfam TetR-family
protein matches, then split into node-level fragments that match the
`cases/tetr_dopamine` design graph.

## Files

- `tetr_interpro_records.jsonl`: annotated TetR-family sequence fragments.
- `tetr_interpro_records.manifest.json`: download/build manifest.
- `tetr_interpro_external_prior_cache.json`: ASTevolve prior cache generated
  from the TetR records.
- `embedding_metadata_esm2_t6_8M_tetr_interpro.jsonl`: metadata rows aligned
  to the embedding matrix.
- `embeddings_esm2_t6_8M_tetr_interpro.npy`: ESM2 mean-pooled fragment
  embeddings.
- `embedding_manifest_esm2_t6_8M_tetr_interpro.json`: manifest consumed by
  the external KB provider.
- `tetr_dopamine_external_prior_cache.json`: older smoke/static prior kept only
  as a fallback reference; the TetR case now points to
  `tetr_interpro_external_prior_cache.json`.

## Rebuild

Small sampled rebuild:

```powershell
& 'D:\Programs\Anaconda3\Scripts\conda.exe' run -n pytorch python data\scripts\build_tetr_interpro_kb_v0.py --max-proteins 80 --page-size 40 --sleep 0.2 --include-full
& 'D:\Programs\Anaconda3\Scripts\conda.exe' run -n pytorch python data\scripts\embed_tetr_interpro_records_v0.py --batch-size 16
```

Larger build:

```powershell
& 'D:\Programs\Anaconda3\Scripts\conda.exe' run -n pytorch python data\scripts\build_tetr_interpro_kb_v0.py --max-proteins 5000 --page-size 100 --sleep 0.2 --include-full
& 'D:\Programs\Anaconda3\Scripts\conda.exe' run -n pytorch python data\scripts\embed_tetr_interpro_records_v0.py --batch-size 16
```

Use `--max-proteins 0` only if you intend to pull the full PF00440 match set.
PF00440 is large, so full retrieval and embedding should be treated as a batch
job, not an ordinary smoke test.
