from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from build_sabdab_records_v0 import FIELD_ALIASES, _get, _norm_header, _split_chains


SABDAB_PDB_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb}/"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

CHOTHIA_CDR_RANGES = {
    "VH_CDR1": ("H", 26, 32),
    "VH_CDR2": ("H", 52, 56),
    "VH_CDR3": ("H", 95, 102),
    "VL_CDR1": ("L", 24, 34),
    "VL_CDR2": ("L", 50, 56),
    "VL_CDR3": ("L", 89, 97),
}


def _base_metadata(row: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for key, aliases in FIELD_ALIASES.items():
        out[key] = _get(row, aliases)
    out["antigen_chains"] = _split_chains(out.get("antigen_chain", ""))
    return out


def _download_text(url: str, timeout: int = 60) -> str:
    with urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def fetch_pdb(pdb: str, pdb_dir: Path) -> Optional[Path]:
    pdb = pdb.lower()
    pdb_dir.mkdir(parents=True, exist_ok=True)
    out = pdb_dir / f"{pdb}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out

    for url in (SABDAB_PDB_URL.format(pdb=pdb), RCSB_PDB_URL.format(pdb=pdb.upper())):
        try:
            text = _download_text(url)
            if text.startswith("ATOM") or "\nATOM" in text or "\nSEQRES" in text:
                out.write_text(text, encoding="utf-8")
                return out
        except (HTTPError, URLError, TimeoutError, OSError):
            continue
    return None


def _resseq_int(raw: str) -> Optional[int]:
    raw = raw.strip()
    digits = "".join(c for c in raw if c.isdigit() or c == "-")
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_atom_chain_sequences(pdb_path: Path) -> Dict[str, List[Tuple[str, str, int, str]]]:
    chains: Dict[str, List[Tuple[str, str, int, str]]] = {}
    seen = set()
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        resname = line[17:20].strip()
        aa = THREE_TO_ONE.get(resname)
        if not aa:
            continue
        chain = line[21].strip() or "_"
        resseq_raw = line[22:27].strip()
        ins = line[26].strip()
        resseq = _resseq_int(resseq_raw)
        if resseq is None:
            continue
        key = (chain, resseq_raw, ins)
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain, []).append((aa, resseq_raw, resseq, ins))
    return chains


def _chain_seq(entries: List[Tuple[str, str, int, str]]) -> str:
    return "".join(x[0] for x in entries)


def _cdr_seq(entries: List[Tuple[str, str, int, str]], start: int, end: int) -> Tuple[str, List[str]]:
    selected = [x for x in entries if start <= x[2] <= end]
    return "".join(x[0] for x in selected), [x[1] for x in selected]


def _record(
    *,
    record_id: str,
    pdb: str,
    substructure_class: str,
    substructure_id: str,
    substructure_name: str,
    sequence_fragment: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "source": "sabdab_chothia_pdb",
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
        "span_indexing": "chothia_numbering_for_antibody_chains",
        "is_discontinuous": False,
        "span_length": len(sequence_fragment) if sequence_fragment else None,
        "sequence_fragment": sequence_fragment,
        "has_sequence": bool(sequence_fragment),
        "metadata": metadata,
    }


def iter_summary_rows(summary_tsv: Path, max_rows: Optional[int]) -> Iterable[Dict[str, str]]:
    with summary_tsv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return
        reader.fieldnames = [_norm_header(x) for x in reader.fieldnames]
        for i, row in enumerate(reader, start=1):
            if max_rows is not None and i > max_rows:
                break
            yield {_norm_header(k): (v or "") for k, v in row.items()}


def build_structure_records(
    summary_tsv: Path,
    pdb_dir: Path,
    out_jsonl: Path,
    max_rows: Optional[int],
    max_structures: Optional[int],
) -> Dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    structures_attempted = 0
    structures_parsed = 0
    records_written = 0
    seen_pdb = set()

    with out_jsonl.open("w", encoding="utf-8") as out:
        for row in iter_summary_rows(summary_tsv, max_rows):
            rows_read += 1
            meta = _base_metadata(row)
            pdb = str(meta.get("pdb") or "").lower()
            if not pdb:
                continue
            if max_structures is not None and pdb not in seen_pdb and len(seen_pdb) >= max_structures:
                continue

            if pdb not in seen_pdb:
                seen_pdb.add(pdb)
                structures_attempted += 1
            pdb_path = fetch_pdb(pdb, pdb_dir)
            if pdb_path is None:
                continue

            chains = parse_atom_chain_sequences(pdb_path)
            if not chains:
                continue
            structures_parsed += 1

            pair_id = f"{pdb}_{meta.get('hchain') or '-'}_{meta.get('lchain') or '-'}_{meta.get('model') or '0'}"
            chain_roles = [
                ("VH_chain", meta.get("hchain", ""), "Variable heavy chain"),
                ("VL_chain", meta.get("lchain", ""), "Variable light chain"),
            ]
            for role, chain_id, name in chain_roles:
                if not chain_id or chain_id == "NA" or chain_id not in chains:
                    continue
                seq = _chain_seq(chains[chain_id])
                rec = _record(
                    record_id=f"{pair_id}|{role}|{chain_id}",
                    pdb=pdb,
                    substructure_class=role,
                    substructure_id=chain_id,
                    substructure_name=name,
                    sequence_fragment=seq,
                    metadata={**meta, "chain_id": chain_id},
                )
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_written += 1

            for ag_chain in meta.get("antigen_chains", []):
                if ag_chain not in chains:
                    continue
                seq = _chain_seq(chains[ag_chain])
                rec = _record(
                    record_id=f"{pair_id}|Antigen_chain_sequence|{ag_chain}",
                    pdb=pdb,
                    substructure_class="Antigen_chain_sequence",
                    substructure_id=ag_chain,
                    substructure_name=meta.get("antigen_name") or "Antigen chain sequence",
                    sequence_fragment=seq,
                    metadata={**meta, "chain_id": ag_chain},
                )
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_written += 1

            for node_name, (chain_role, start, end) in CHOTHIA_CDR_RANGES.items():
                chain_id = meta.get("hchain") if chain_role == "H" else meta.get("lchain")
                if not chain_id or chain_id == "NA" or chain_id not in chains:
                    continue
                seq, residue_ids = _cdr_seq(chains[chain_id], start, end)
                if not seq:
                    continue
                rec = _record(
                    record_id=f"{pair_id}|{node_name}|{chain_id}|{start}-{end}",
                    pdb=pdb,
                    substructure_class="Antibody_CDR",
                    substructure_id=node_name,
                    substructure_name=f"SAbDab Chothia {node_name}",
                    sequence_fragment=seq,
                    metadata={
                        **meta,
                        "node_name": node_name,
                        "chain_id": chain_id,
                        "chothia_start": start,
                        "chothia_end": end,
                        "chothia_residue_ids": residue_ids,
                    },
                )
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_written += 1

    return {
        "summary_tsv": str(summary_tsv),
        "pdb_dir": str(pdb_dir),
        "out": str(out_jsonl),
        "rows_read": rows_read,
        "structures_attempted": structures_attempted,
        "structures_parsed": structures_parsed,
        "records_written": records_written,
        "source": "sabdab_chothia_pdb",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sequence-bearing SAbDab records from Chothia-numbered PDB files.")
    parser.add_argument("--summary", default=str(DATA_ROOT / "sabdab_raw" / "sabdab_summary.tsv"))
    parser.add_argument("--pdb-dir", default=str(DATA_ROOT / "sabdab_raw" / "pdb"))
    parser.add_argument("--out", default=str(DATA_ROOT / "antibody_kb" / "sabdab_structure_records.jsonl"))
    parser.add_argument("--manifest", default=str(DATA_ROOT / "antibody_kb" / "sabdab_structure_records.manifest.json"))
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--max-structures", type=int, default=100)
    args = parser.parse_args()

    manifest = build_structure_records(
        summary_tsv=Path(args.summary),
        pdb_dir=Path(args.pdb_dir),
        out_jsonl=Path(args.out),
        max_rows=args.max_rows,
        max_structures=args.max_structures,
    )
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
