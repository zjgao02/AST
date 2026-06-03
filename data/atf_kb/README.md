# ATF InterPro KB

This directory is the allosteric transcription factor (ATF) external knowledge
repository for ASTevolve.

It is broader than the old TetR-only KB. Records are generated from curated
InterPro/Pfam seeds for major bacterial ligand-responsive or allosterically
regulated transcription factor families, then split into annotated fragments
that can act as MCTS priors.

## Current Family Seeds

- `tetr`: TetR/AcrR, `PF00440`
- `laci`: LacI/GalR, `PF00356`
- `lysr`: LysR, `PF00126`
- `gntr`: GntR, `PF00392`
- `marr`: MarR, `PF01047`
- `arac`: AraC/XylS, `PF00165`
- `merr`: MerR, `PF00376`
- `luxr`: LuxR/FixJ, `PF00196`
- `iclr`: IclR, `PF01614`
- `crp`: CRP/FNR, `PF00325`
- `arsr`: ArsR/SmtB, `PF01022`
- `ompr`: OmpR/PhoB response regulator, `PF00486`

## Files

- `atf_interpro_records.jsonl`: annotated ATF-family sequence fragments.
- `atf_interpro_records.manifest.json`: download/build manifest.
- `atf_interpro_external_prior_cache.json`: ASTevolve prior cache generated
  from the ATF records.
- `embedding_metadata_esm2_t6_8M_atf_interpro.jsonl`: metadata rows aligned to
  the embedding matrix.
- `embeddings_esm2_t6_8M_atf_interpro.npy`: ESM2 mean-pooled fragment
  embeddings.
- `embedding_manifest_esm2_t6_8M_atf_interpro.json`: manifest consumed by the
  external KB provider.

## Rebuild

Small sampled rebuild:

```powershell
& 'D:\Programs\Anaconda3\Scripts\conda.exe' run -n pytorch python data\scripts\build_atf_interpro_kb_v0.py --max-proteins-per-family 10 --page-size 50 --sleep 0.2 --include-full
& 'D:\Programs\Anaconda3\Scripts\conda.exe' run -n pytorch python data\scripts\embed_atf_interpro_records_v0.py --batch-size 16
```

Representative build for routine ASTevolve runs:

```powershell
Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Downloads\ast\data\scripts\run_atf_representative_kb_v0.ps1','-MaxProteinsPerFamily','200') -WindowStyle Hidden
```

Use `--max-proteins-per-family 0` only for an intentional full pull. Some
families contain hundreds of thousands of matches, so full retrieval and
embedding should be treated as a batch job.

Background full rebuild:

```powershell
Start-Process -FilePath 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Downloads\ast\data\scripts\run_atf_full_kb_v0.ps1') -WindowStyle Hidden
```

Monitor:

```powershell
Get-Content D:\Downloads\ast\artifacts\logs\atf_kb_representative_latest_status.json
Get-ChildItem D:\Downloads\ast\artifacts\logs -Filter 'atf_kb_representative_*'
```
