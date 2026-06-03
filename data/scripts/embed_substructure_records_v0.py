from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))
DEFAULT_RECORDS = DATA_ROOT / "external_kb" / "substructure_records.jsonl"
DEFAULT_EMBEDDINGS = DATA_ROOT / "external_kb" / "embeddings.npy"
DEFAULT_METADATA = DATA_ROOT / "external_kb" / "embedding_metadata.jsonl"
DEFAULT_MANIFEST = DATA_ROOT / "external_kb" / "embedding_manifest.json"


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_sequence(seq: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa if aa in allowed else "X" for aa in seq.upper())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed Magneton substructure records with an ESM2 model."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-records", type=int, default=3000)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[records] {args.records}")
    print(f"[model] {args.model}")
    print(f"[device] {device}")
    print(f"[max-records] {args.max_records}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    selected = []
    skipped_no_seq = 0
    skipped_long = 0

    for rec in iter_records(args.records):
        seq = normalize_sequence(rec.get("sequence_fragment") or "")
        if not seq:
            skipped_no_seq += 1
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
        raise RuntimeError("No records selected for embedding.")

    embeddings = []
    metadata_rows = []

    with torch.no_grad():
        for start in tqdm(range(0, len(selected), args.batch_size), desc="embedding"):
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
                # ESM tokenization uses special tokens around the residue sequence.
                # For ESM2 via HF, residue tokens are positions [1, 1 + seq_len).
                residue_embeddings = hidden[i, 1 : 1 + seq_len, :]
                if residue_embeddings.shape[0] != seq_len:
                    valid_len = int(attention[i].sum())
                    residue_embeddings = hidden[i, 1 : max(1, valid_len - 1), :]
                pooled = residue_embeddings.mean(axis=0).astype(np.float32)
                embeddings.append(pooled)

                metadata_rows.append({
                    "embedding_index": len(metadata_rows),
                    "record_id": rec.get("record_id"),
                    "protein_id": rec.get("protein_id"),
                    "substructure_class": rec.get("substructure_class"),
                    "substructure_id": rec.get("substructure_id"),
                    "match_id": rec.get("match_id"),
                    "substructure_name": rec.get("substructure_name"),
                    "span_length": rec.get("span_length"),
                    "sequence_length": seq_len,
                    "residue_spans": rec.get("residue_spans"),
                })

    emb = np.vstack(embeddings).astype(np.float32)
    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings, emb)

    with args.metadata.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "records": str(args.records),
        "embeddings": str(args.embeddings),
        "metadata": str(args.metadata),
        "model": args.model,
        "device": device,
        "embedding_shape": list(emb.shape),
        "pooling": "mean_residue_embedding",
        "selected_records": len(selected),
        "skipped_no_sequence": skipped_no_seq,
        "skipped_long": skipped_long,
        "max_length": args.max_length,
        "max_records": args.max_records,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
