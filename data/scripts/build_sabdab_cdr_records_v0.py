from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


CDR_LINK_RE = re.compile(
    r"cdrviewer/\?pdb=(?P<pdb>[0-9a-zA-Z]{4})&loop=CDR(?P<chain_type>[HL])(?P<cdr_no>[123])&chain=(?P<chain>[^&'\"]+)&CDRdef=(?P<cdr_def>[^'\"]+)['\"][^>]*>(?P<seq>[A-Za-z]+)</a>",
    re.IGNORECASE,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))


def _strip_tags(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def _cells(row_html: str) -> List[str]:
    return re.findall(r"<td\b[^>]*>(.*?)</td>", row_html, flags=re.I | re.S)


def _pdb_from_row(row_html: str) -> str:
    m = re.search(r"structureviewer/\?pdb=([0-9a-zA-Z]{4})", row_html, flags=re.I)
    return m.group(1).lower() if m else ""


def _resolution_float(value: str):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _node_name(chain_type: str, cdr_no: str) -> str:
    prefix = "VH" if chain_type.upper() == "H" else "VL"
    return f"{prefix}_CDR{cdr_no}"


def _record(
    *,
    record_id: str,
    pdb: str,
    node_name: str,
    seq: str,
    chain: str,
    cdr_def: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "record_id": record_id,
        "source": f"sabdab_cdrsearch_{cdr_def.lower()}",
        "protein_id": pdb,
        "kb_id": f"sabdab|{pdb}",
        "protein_name": pdb,
        "protein_length": None,
        "substructure_class": "Antibody_CDR",
        "substructure_id": node_name,
        "match_id": f"{node_name}:{chain}",
        "substructure_name": f"SAbDab {cdr_def} {node_name}",
        "representative": False,
        "residue_spans": [],
        "span_indexing": f"{cdr_def.lower()}_cdr_definition",
        "is_discontinuous": False,
        "span_length": len(seq),
        "sequence_fragment": seq,
        "has_sequence": bool(seq),
        "metadata": metadata,
    }


def parse_cdr_html(html_path: Path, out_jsonl: Path, manifest_path: Path) -> Dict[str, Any]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.I | re.S)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    pdb_count = set()
    node_counts: Dict[str, int] = {}

    with out_jsonl.open("w", encoding="utf-8") as out:
        for row_idx, row in enumerate(rows):
            if "cdrviewer" not in row.lower():
                continue
            cells = _cells(row)
            pdb = _pdb_from_row(row)
            if not pdb:
                continue
            species = _strip_tags(cells[1]) if len(cells) > 1 else ""
            method = _strip_tags(cells[2]) if len(cells) > 2 else ""
            resolution = _strip_tags(cells[3]) if len(cells) > 3 else ""
            pdb_count.add(pdb)

            for match_idx, m in enumerate(CDR_LINK_RE.finditer(row)):
                cdr_def = html.unescape(m.group("cdr_def"))
                node_name = _node_name(m.group("chain_type"), m.group("cdr_no"))
                chain = html.unescape(m.group("chain"))
                seq = html.unescape(m.group("seq")).upper()
                metadata = {
                    "pdb": pdb,
                    "species": species,
                    "method": method,
                    "resolution": resolution,
                    "resolution_float": _resolution_float(resolution),
                    "node_name": node_name,
                    "chain_id": chain,
                    "cdr_definition": cdr_def,
                    "row_index": row_idx,
                }
                rec = _record(
                    record_id=f"{pdb}|{node_name}|{chain}|{match_idx}",
                    pdb=pdb,
                    node_name=node_name,
                    seq=seq,
                    chain=chain,
                    cdr_def=cdr_def,
                    metadata=metadata,
                )
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_written += 1
                node_counts[node_name] = node_counts.get(node_name, 0) + 1

    manifest = {
        "html_path": str(html_path),
        "out": str(out_jsonl),
        "records_written": records_written,
        "unique_pdbs": len(pdb_count),
        "node_counts": node_counts,
        "source": "sabdab_cdrsearch",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SAbDab all-CDR HTML into ASTevolve Antibody_CDR records.")
    parser.add_argument("--html", default=str(DATA_ROOT / "sabdab_raw" / "cdrs_chothia.html"))
    parser.add_argument("--out", default=str(DATA_ROOT / "antibody_kb" / "sabdab_cdr_records.jsonl"))
    parser.add_argument("--manifest", default=str(DATA_ROOT / "antibody_kb" / "sabdab_cdr_records.manifest.json"))
    args = parser.parse_args()

    manifest = parse_cdr_html(Path(args.html), Path(args.out), Path(args.manifest))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
