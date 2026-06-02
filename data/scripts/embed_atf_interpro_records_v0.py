from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AA = "ACDEFGHIKLMNPQRSTVWY"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_sequence(seq: str) -> str:
    return "".join(aa if aa in AA else "X" for aa in str(seq or "").upper())


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed ATF InterPro fragment records with an ESM2 model."
    )
    kb_dir = project_root() / "data" / "atf_kb"
    parser.add_argument("--records", type=Path, default=kb_dir / "atf_interpro_records.jsonl")
    parser.add_argument("--embeddings", type=Path, default=kb_dir / "embeddings_esm2_t6_8M_atf_interpro.npy")
    parser.add_argument("--metadata", type=Path, default=kb_dir / "embedding_metadata_esm2_t6_8M_atf_interpro.jsonl")
    parser.add_argument("--manifest", type=Path, default=kb_dir / "embedding_manifest_esm2_t6_8M_atf_interpro.json")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    selected: list[dict[str, Any]] = []
    skipped_no_sequence = 0
    skipped_long = 0
    for rec in iter_jsonl(args.records):
        seq = normalize_sequence(rec.get("sequence_fragment") or "")
        if not seq:
            skipped_no_sequence += 1
            continue
        if len(seq) > args.max_length:
            skipped_long += 1
            continue
        rec = dict(rec)
        rec["_embedding_sequence"] = seq
        selected.append(rec)
        if args.max_records and len(selected) >= args.max_records:
            break

    if not selected:
        raise RuntimeError(f"No embeddable records selected from {args.records}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    embeddings = []
    metadata_rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(selected), args.batch_size), desc="embedding ATF InterPro"):
            batch = selected[start : start + args.batch_size]
            seqs = [rec["_embedding_sequence"] for rec in batch]
            encoded = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            out = model(**encoded)
            hidden = out.last_hidden_state.detach().cpu().numpy()
            attention = encoded["attention_mask"].detach().cpu().numpy()

            for i, rec in enumerate(batch):
                seq_len = len(rec["_embedding_sequence"])
                residue_embeddings = hidden[i, 1 : 1 + seq_len, :]
                if residue_embeddings.shape[0] != seq_len:
                    valid_len = int(attention[i].sum())
                    residue_embeddings = hidden[i, 1 : max(1, valid_len - 1), :]
                embeddings.append(residue_embeddings.mean(axis=0).astype(np.float32))
                metadata_rows.append(
                    {
                        "embedding_index": len(metadata_rows),
                        "record_id": rec.get("record_id"),
                        "family_id": rec.get("family_id"),
                        "family_name": rec.get("family_name"),
                        "protein_id": rec.get("protein_id"),
                        "protein_name": rec.get("protein_name"),
                        "substructure_class": rec.get("substructure_class"),
                        "substructure_id": rec.get("substructure_id"),
                        "node_name": rec.get("node_name"),
                        "node_kind": rec.get("node_kind"),
                        "entry_accession": rec.get("entry_accession"),
                        "match_id": rec.get("match_id"),
                        "substructure_name": rec.get("substructure_name"),
                        "span_length": rec.get("span_length"),
                        "sequence_length": seq_len,
                        "residue_spans": rec.get("residue_spans"),
                    }
                )

    emb = np.vstack(embeddings).astype(np.float32)
    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings, emb)
    with args.metadata.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = args.manifest.parent
    manifest = {
        "records": relative_or_absolute(args.records, base),
        "embeddings": relative_or_absolute(args.embeddings, base),
        "metadata": relative_or_absolute(args.metadata, base),
        "model": args.model,
        "device": device,
        "embedding_shape": list(emb.shape),
        "pooling": "mean_residue_embedding",
        "selected_records": len(selected),
        "skipped_no_sequence": skipped_no_sequence,
        "skipped_long": skipped_long,
        "max_length": args.max_length,
        "max_records": args.max_records,
        "source": "atf_interpro_records",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
