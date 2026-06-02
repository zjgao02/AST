from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CDR_ALIASES = {
    "VH_CDR1": ["h1", "cdrh1", "h_cdr1", "heavy_cdr1", "cdr_h1"],
    "VH_CDR2": ["h2", "cdrh2", "h_cdr2", "heavy_cdr2", "cdr_h2"],
    "VH_CDR3": ["h3", "cdrh3", "h_cdr3", "heavy_cdr3", "cdr_h3"],
    "VL_CDR1": ["l1", "cdrl1", "l_cdr1", "light_cdr1", "cdr_l1"],
    "VL_CDR2": ["l2", "cdrl2", "l_cdr2", "light_cdr2", "cdr_l2"],
    "VL_CDR3": ["l3", "cdrl3", "l_cdr3", "light_cdr3", "cdr_l3"],
}

FIELD_ALIASES = {
    "pdb": ["pdb", "pdb_code", "pdb_id"],
    "hchain": ["hchain", "h_chain", "heavy_chain", "heavy"],
    "lchain": ["lchain", "l_chain", "light_chain", "light"],
    "model": ["model"],
    "antigen_chain": ["antigen_chain", "antigenchain", "ag_chain", "agchain"],
    "antigen_type": ["antigen_type", "ag_type", "agtype"],
    "antigen_name": ["antigen_name", "antigen", "ag_name"],
    "antigen_species": ["antigen_species", "ag_species"],
    "heavy_species": ["heavy_species", "h_species"],
    "light_species": ["light_species", "l_species"],
    "method": ["method", "experimental_method"],
    "resolution": ["resolution"],
    "affinity": ["affinity", "kd", "k_d"],
    "affinity_method": ["affinity_method"],
    "pmid": ["pmid", "pubmed"],
    "scfv": ["scfv", "single_chain", "single_chain_fv"],
    "engineered": ["engineered"],
    "heavy_subclass": ["heavy_subclass", "h_subclass"],
    "light_subclass": ["light_subclass", "l_subclass"],
    "light_ctype": ["light_ctype", "l_ctype"],
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))


def _norm_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _get(row: Dict[str, str], aliases: Iterable[str], default: str = "") -> str:
    for alias in aliases:
        key = _norm_header(alias)
        if key in row and str(row[key]).strip() not in {"", "NA", "None", "none", "nan"}:
            return str(row[key]).strip()
    return default


def _split_chains(value: str) -> List[str]:
    if not value:
        return []
    for sep in ["|", ",", ";", " "]:
        value = value.replace(sep, ",")
    return [x.strip() for x in value.split(",") if x.strip()]


def _clean_seq(value: str) -> str:
    value = "".join(c for c in value.strip().upper() if c.isalpha())
    return value if value and value not in {"NA", "NONE"} else ""


def _to_float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _base_metadata(row: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for key, aliases in FIELD_ALIASES.items():
        out[key] = _get(row, aliases)
    out["resolution_float"] = _to_float(out.get("resolution", ""))
    return out


def _record(
    *,
    record_id: str,
    pdb: str,
    substructure_class: str,
    substructure_id: str,
    substructure_name: str,
    sequence_fragment: str,
    metadata: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "source": source,
        "protein_id": pdb,
        "kb_id": f"sabdab|{pdb}",
        "protein_name": metadata.get("antigen_name") or pdb,
        "protein_length": None,
        "substructure_class": substructure_class,
        "substructure_id": substructure_id,
        "match_id": substructure_id,
        "substructure_name": substructure_name,
        "representative": False,
        "residue_spans": [],
        "span_indexing": "not_applicable_summary_tsv",
        "is_discontinuous": False,
        "span_length": len(sequence_fragment) if sequence_fragment else None,
        "sequence_fragment": sequence_fragment,
        "has_sequence": bool(sequence_fragment),
        "metadata": metadata,
    }


def build_records(summary_tsv: Path, out_jsonl: Path, source: str, max_rows: Optional[int]) -> Dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    records_written = 0
    class_counts: Counter[str] = Counter()
    antigen_type_counts: Counter[str] = Counter()

    with summary_tsv.open("r", encoding="utf-8", errors="ignore", newline="") as f, out_jsonl.open(
        "w", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {summary_tsv}")
        reader.fieldnames = [_norm_header(x) for x in reader.fieldnames]

        for row in reader:
            rows_read += 1
            if max_rows is not None and rows_read > max_rows:
                break
            row = {_norm_header(k): (v or "") for k, v in row.items()}
            meta = _base_metadata(row)
            pdb = meta["pdb"].lower()
            if not pdb:
                continue

            antigen_type = (meta.get("antigen_type") or "unknown").lower()
            antigen_type_counts[antigen_type] += 1
            hchain = meta.get("hchain", "")
            lchain = meta.get("lchain", "")
            agchains = _split_chains(meta.get("antigen_chain", ""))
            pair_id = f"{pdb}_{hchain or '-'}_{lchain or '-'}_{meta.get('model') or '0'}"

            antibody_record = _record(
                record_id=f"{pair_id}|Antibody_pair",
                pdb=pdb,
                substructure_class="Antibody_pair",
                substructure_id=pair_id,
                substructure_name="SAbDab antibody heavy-light pairing",
                sequence_fragment="",
                metadata={**meta, "antigen_chains": agchains},
                source=source,
            )
            out.write(json.dumps(antibody_record, ensure_ascii=False) + "\n")
            records_written += 1
            class_counts["Antibody_pair"] += 1

            if agchains:
                antigen_record = _record(
                    record_id=f"{pair_id}|Antigen_chain|{','.join(agchains)}",
                    pdb=pdb,
                    substructure_class="Antigen_chain",
                    substructure_id=",".join(agchains),
                    substructure_name=meta.get("antigen_name") or "SAbDab antigen chain",
                    sequence_fragment="",
                    metadata={**meta, "antigen_chains": agchains},
                    source=source,
                )
                out.write(json.dumps(antigen_record, ensure_ascii=False) + "\n")
                records_written += 1
                class_counts["Antigen_chain"] += 1

            for node_name, aliases in CDR_ALIASES.items():
                seq = _clean_seq(_get(row, aliases))
                if not seq:
                    continue
                record = _record(
                    record_id=f"{pair_id}|{node_name}|{seq}",
                    pdb=pdb,
                    substructure_class="Antibody_CDR",
                    substructure_id=node_name,
                    substructure_name=f"SAbDab {node_name}",
                    sequence_fragment=seq,
                    metadata={**meta, "antigen_chains": agchains, "node_name": node_name},
                    source=source,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1
                class_counts["Antibody_CDR"] += 1

    return {
        "summary_tsv": str(summary_tsv),
        "out": str(out_jsonl),
        "rows_read": rows_read,
        "records_written": records_written,
        "class_counts": dict(class_counts),
        "antigen_type_counts": dict(antigen_type_counts),
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ASTevolve antibody_kb records from SAbDab summary TSV.")
    parser.add_argument("--summary", default=str(DATA_ROOT / "sabdab_raw" / "sabdab_summary.tsv"))
    parser.add_argument("--out", default=str(DATA_ROOT / "antibody_kb" / "sabdab_records.jsonl"))
    parser.add_argument("--manifest", default=str(DATA_ROOT / "antibody_kb" / "sabdab_records.manifest.json"))
    parser.add_argument("--source", default="sabdab_summary_tsv")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    manifest = build_records(
        summary_tsv=Path(args.summary),
        out_jsonl=Path(args.out),
        source=args.source,
        max_rows=args.max_rows,
    )
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
