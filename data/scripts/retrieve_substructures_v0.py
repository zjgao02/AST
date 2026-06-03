from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))
DEFAULT_RECORDS = DATA_ROOT / "external_kb" / "substructure_records.jsonl"
DEFAULT_EMBEDDINGS = DATA_ROOT / "external_kb" / "embeddings.npy"
DEFAULT_METADATA = DATA_ROOT / "external_kb" / "embedding_metadata.jsonl"


AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("KRDE")
POLAR = set("STNQY")
AROMATIC = set("FWYH")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records_by_id(path: Path) -> dict[str, dict]:
    return {rec["record_id"]: rec for rec in iter_jsonl(path)}


def summarize_hits(hits: list[dict]) -> dict:
    class_counts = Counter(hit["metadata"].get("substructure_class") for hit in hits)
    aa_counts = Counter()
    lengths = []

    for hit in hits:
        rec = hit.get("record") or {}
        seq = rec.get("sequence_fragment") or ""
        aa_counts.update(aa for aa in seq if aa in AA)
        if rec.get("span_length") is not None:
            lengths.append(float(rec["span_length"]))

    aa_total = sum(aa_counts.values())
    aa_frequency = {aa: aa_counts[aa] / aa_total if aa_total else 0.0 for aa in AA}

    return {
        "class_counts": dict(class_counts),
        "length_prior": {
            "min": min(lengths) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "mean": float(np.mean(lengths)) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "residue_class_bias": {
            "hydrophobic": sum(aa_frequency[aa] for aa in HYDROPHOBIC),
            "charged": sum(aa_frequency[aa] for aa in CHARGED),
            "polar": sum(aa_frequency[aa] for aa in POLAR),
            "aromatic": sum(aa_frequency[aa] for aa in AROMATIC),
        },
        "favored_residues": [
            aa for aa, _ in aa_counts.most_common(8)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect nearest neighbors in ASTevolve Magneton external KB embeddings."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--class-filter", default="")
    args = parser.parse_args()

    emb = np.load(args.embeddings).astype(np.float32)
    meta = list(iter_jsonl(args.metadata))
    records = load_records_by_id(args.records)

    if len(meta) != emb.shape[0]:
        raise ValueError(f"Metadata count {len(meta)} != embeddings rows {emb.shape[0]}")
    if not (0 <= args.query_index < emb.shape[0]):
        raise IndexError(args.query_index)

    normalized = normalize_rows(emb)
    query = normalized[args.query_index]
    scores = normalized @ query

    order = np.argsort(-scores)
    hits = []
    for idx in order:
        if int(idx) == args.query_index:
            continue
        row = meta[int(idx)]
        if args.class_filter and row.get("substructure_class") != args.class_filter:
            continue
        record = records.get(row["record_id"])
        hits.append({
            "rank": len(hits) + 1,
            "index": int(idx),
            "score": float(scores[int(idx)]),
            "metadata": row,
            "record": record,
        })
        if len(hits) >= args.top_k:
            break

    query_meta = meta[args.query_index]
    print("[query]")
    print(json.dumps(query_meta, ensure_ascii=False, indent=2))
    print("\n[hits]")
    for hit in hits:
        row = hit["metadata"]
        print(
            f"{hit['rank']:>2}. score={hit['score']:.4f} "
            f"class={row.get('substructure_class')} "
            f"id={row.get('substructure_id')} "
            f"name={row.get('substructure_name')}"
        )

    print("\n[prior summary]")
    print(json.dumps(summarize_hits(hits), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
