from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_RAW_ROOT = Path(r"D:\Downloads\data\magneton_raw")
DEFAULT_OUT = Path(r"D:\Downloads\data\external_kb\substructure_records.jsonl")

DEFAULT_CLASSES = (
    "Active_site",
    "Binding_site",
    "Conserved_site",
    "Domain",
    "Homologous_superfamily",
)


def iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_needed_ids(jsonl_path: Path, max_proteins: int = 0) -> set[str]:
    needed: set[str] = set()
    for i, protein in enumerate(iter_jsonl_gz(jsonl_path), start=1):
        protein_id = protein.get("uniprot_id")
        if protein_id:
            needed.add(str(protein_id))
        if max_proteins and i >= max_proteins:
            break
    return needed


def infer_columns(fields: list[str]) -> tuple[str, str]:
    lower_to_original = {field.lower(): field for field in fields}

    id_candidates = (
        "uniprot_id",
        "entry",
        "entry name",
        "accession",
        "primaryaccession",
        "uniprot",
        "id",
    )
    seq_candidates = ("sequence", "seq", "protein_sequence")

    id_col = next((lower_to_original[name] for name in id_candidates if name in lower_to_original), None)
    seq_col = next((lower_to_original[name] for name in seq_candidates if name in lower_to_original), None)

    if id_col is None or seq_col is None:
        raise ValueError(f"Could not infer ID/sequence columns from TSV fields: {fields}")
    return id_col, seq_col


def parse_fasta_id(header: str) -> str:
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]
    token = header.split()[0]
    parts = token.split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return token


def load_sequences_from_fasta(path: Path, needed_ids: set[str]) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_parts
        if current_id and current_id in needed_ids:
            sequences[current_id] = "".join(current_parts)
        current_id = None
        current_parts = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                parsed_id = parse_fasta_id(line)
                current_id = parsed_id if parsed_id in needed_ids else None
                current_parts = []
            elif current_id is not None:
                current_parts.append(line)
        flush()

    return sequences


def load_sequences_from_tsv(path: Path, needed_ids: set[str]) -> dict[str, str] | None:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        try:
            id_col, seq_col = infer_columns(fields)
        except ValueError:
            return None

        seqs: dict[str, str] = {}
        for row in reader:
            protein_id = (row.get(id_col) or "").strip()
            sequence = (row.get(seq_col) or "").strip()
            if protein_id in needed_ids and sequence:
                seqs[protein_id] = sequence

    return seqs


def load_sequences(seq_tsv: Path, fasta_gz: Path, needed_ids: set[str]) -> dict[str, str]:
    from_tsv = load_sequences_from_tsv(seq_tsv, needed_ids)
    if from_tsv is not None:
        return from_tsv
    return load_sequences_from_fasta(fasta_gz, needed_ids)


def extract_fragment(sequence: str | None, spans: list[list[int]]) -> str | None:
    if not sequence:
        return None

    pieces: list[str] = []
    length = len(sequence)
    for start, end in spans:
        start = int(start)
        end = int(end)

        # Magneton positions appear to be 0-based, half-open intervals.
        if start < 0 or end < start or end > length:
            return None

        pieces.append(sequence[start:end])
    return "".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Magneton protein records into ASTevolve substructure records."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-proteins", type=int, default=0)
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated Magneton element_type values to keep.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root
    jsonl_path = raw_root / "interpro_103.0" / "debug_subset" / "swissprot.with_ss.0.jsonl.gz"
    seq_tsv = raw_root / "sequences" / "swissprot_subset.tsv"
    fasta_gz = raw_root / "sequences" / "uniprot_sprot.fasta.gz"
    keep_classes = {item.strip() for item in args.classes.split(",") if item.strip()}

    print(f"[raw-root] {raw_root}")
    print(f"[jsonl] {jsonl_path}")
    print(f"[seq-tsv] {seq_tsv}")
    print(f"[fasta-gz] {fasta_gz}")
    print(f"[out] {args.out}")
    print(f"[classes] {sorted(keep_classes)}")

    needed_ids = collect_needed_ids(jsonl_path, max_proteins=args.max_proteins)
    print(f"[needed protein ids] {len(needed_ids)}")

    sequences = load_sequences(seq_tsv, fasta_gz, needed_ids)
    print(f"[loaded needed sequences] {len(sequences)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    proteins_read = 0
    records_written = 0
    records_with_sequence = 0
    class_counts: Counter[str] = Counter()

    with args.out.open("w", encoding="utf-8") as out_handle:
        for protein in iter_jsonl_gz(jsonl_path):
            proteins_read += 1
            if args.max_proteins and proteins_read > args.max_proteins:
                break

            protein_id = protein.get("uniprot_id")
            sequence = sequences.get(protein_id)

            for entry_idx, entry in enumerate(protein.get("entries", [])):
                element_type = entry.get("element_type")
                if element_type not in keep_classes:
                    continue

                spans = entry.get("positions") or []
                if not spans:
                    continue

                fragment = extract_fragment(sequence, spans)
                span_length = sum(int(end) - int(start) for start, end in spans)

                record = {
                    "record_id": (
                        f"{protein_id}|{element_type}|{entry.get('id')}|"
                        f"{entry.get('match_id')}|{entry_idx}"
                    ),
                    "source": "magneton_interpro_103.0_debug",
                    "protein_id": protein_id,
                    "kb_id": protein.get("kb_id"),
                    "protein_name": protein.get("name"),
                    "protein_length": protein.get("length"),
                    "substructure_class": element_type,
                    "substructure_id": entry.get("id"),
                    "match_id": entry.get("match_id"),
                    "substructure_name": entry.get("element_name"),
                    "representative": bool(entry.get("representative", False)),
                    "residue_spans": spans,
                    "span_indexing": "assumed_0_based_half_open",
                    "is_discontinuous": len(spans) > 1,
                    "span_length": span_length,
                    "sequence_fragment": fragment,
                    "has_sequence": fragment is not None,
                    "metadata": {
                        "parsed_entries": protein.get("parsed_entries"),
                        "total_entries": protein.get("total_entries"),
                    },
                }
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                records_written += 1
                class_counts[element_type] += 1
                if fragment is not None:
                    records_with_sequence += 1

    manifest = {
        "raw_root": str(raw_root),
        "jsonl_path": str(jsonl_path),
        "seq_tsv": str(seq_tsv),
        "fasta_gz": str(fasta_gz),
        "out": str(args.out),
        "needed_ids": len(needed_ids),
        "proteins_read": proteins_read,
        "records_written": records_written,
        "records_with_sequence": records_with_sequence,
        "class_counts": dict(class_counts),
        "kept_classes": sorted(keep_classes),
    }

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[done]")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
