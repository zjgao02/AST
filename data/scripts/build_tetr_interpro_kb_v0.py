from __future__ import annotations

import argparse
import json
import ssl
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib import request
from urllib.error import HTTPError, URLError


API_ROOT = "https://www.ebi.ac.uk/interpro/api"
AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWYV")
CHARGED = set("KRDE")
POLAR = set("STNQY")
AROMATIC = set("FWYH")

TETR_ENTRY_SEEDS = (
    ("pfam", "PF00440", "TetR_N"),
)

TETR_DOMAIN_ENTRIES = {
    "PF00440": {
        "name": "TetR_N",
        "node_name": "HTH_recognition_helix",
        "node_kind": "dna_contact",
        "substructure_class": "TetR_HTH_domain",
    },
    "PF02909": {
        "name": "TetR_C_1",
        "node_name": "ligand_pocket_core",
        "node_kind": "pocket",
        "substructure_class": "TetR_C_domain",
    },
    "IPR001647": {
        "name": "HTH_TetR",
        "node_name": "HTH_recognition_helix",
        "node_kind": "dna_contact",
        "substructure_class": "TetR_HTH_domain",
    },
    "IPR004111": {
        "name": "Repressor_TetR_C",
        "node_name": "ligand_pocket_core",
        "node_kind": "pocket",
        "substructure_class": "TetR_C_domain",
    },
    "IPR023772": {
        "name": "DNA-bd_HTH_TetR-type_CS",
        "node_name": "HTH_recognition_helix",
        "node_kind": "dna_contact",
        "substructure_class": "TetR_HTH_conserved_site",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fetch_json(url: str, *, sleep_seconds: float, retries: int = 4) -> dict[str, Any]:
    context = ssl._create_unverified_context()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, context=context, timeout=90) as response:
                if response.status == 204:
                    return {}
                if response.status == 408:
                    time.sleep(max(61.0, sleep_seconds))
                    continue
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code in {408, 429, 500, 502, 503, 504} and attempt + 1 < retries:
                time.sleep(max(5.0, sleep_seconds) * (attempt + 1))
                continue
            raise
        except URLError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(max(5.0, sleep_seconds) * (attempt + 1))
                continue
            raise
    if last_error:
        raise last_error
    return {}


def iter_entry_proteins(
    db: str,
    accession: str,
    *,
    page_size: int,
    max_proteins: int,
    sleep_seconds: float,
) -> Iterable[dict[str, Any]]:
    url = f"{API_ROOT}/protein/UniProt/entry/{db}/{accession}/?page_size={page_size}"
    yielded = 0
    while url:
        payload = fetch_json(url, sleep_seconds=sleep_seconds)
        for item in payload.get("results", []) or []:
            yield item
            yielded += 1
            if max_proteins and yielded >= max_proteins:
                return
        url = payload.get("next")
        if url:
            time.sleep(sleep_seconds)


def fetch_protein(accession: str, sleep_seconds: float) -> dict[str, Any]:
    url = f"{API_ROOT}/protein/uniprot/{accession}"
    return fetch_json(url, sleep_seconds=sleep_seconds)


def fetch_pfam_payload(accession: str, sleep_seconds: float) -> dict[str, Any]:
    protein_payload = fetch_protein(accession, sleep_seconds)
    protein_meta = dict((protein_payload.get("metadata") or {}))
    entries_url = protein_payload.get("entries_url")
    entries: list[dict[str, Any]] = []

    while entries_url:
        entries_payload = fetch_json(str(entries_url), sleep_seconds=sleep_seconds)
        for result in entries_payload.get("results", []) or []:
            meta = result.get("metadata", {}) or {}
            protein_rows = result.get("proteins", []) or []
            locations = []
            for row in protein_rows:
                row_acc = str(row.get("accession") or "").upper()
                if row_acc and row_acc != accession.upper():
                    continue
                locations.extend(row.get("entry_protein_locations", []) or [])
            if not locations:
                continue
            entries.append(
                {
                    "accession": meta.get("accession"),
                    "entry_protein_locations": locations,
                    "protein_length": protein_meta.get("length"),
                    "source_database": meta.get("source_database"),
                    "entry_type": meta.get("type"),
                    "entry_integrated": meta.get("integrated"),
                    "entry_name": meta.get("name"),
                }
            )
        entries_url = entries_payload.get("next")
        if entries_url:
            time.sleep(sleep_seconds)

    return {"metadata": protein_meta, "entry_subset": entries}


def metadata_accession(item: dict[str, Any]) -> str | None:
    meta = item.get("metadata", {}) if isinstance(item, dict) else {}
    value = meta.get("accession") or item.get("accession")
    return str(value) if value else None


def normalize_sequence(seq: str) -> str:
    return "".join(aa for aa in str(seq or "").upper() if aa in AA)


def iter_locations(entry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for location in entry.get("entry_protein_locations", []) or []:
        fragments = []
        for fragment in location.get("fragments", []) or []:
            start = fragment.get("start")
            end = fragment.get("end")
            if start is None or end is None:
                continue
            fragments.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "dc_status": fragment.get("dc-status") or fragment.get("dc_status"),
                    "representative": bool(fragment.get("representative", False)),
                }
            )
        if fragments:
            yield {
                "fragments": fragments,
                "representative": bool(location.get("representative", False)),
                "model": location.get("model"),
                "score": location.get("score"),
            }


def fragment_sequence(sequence: str, fragments: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for fragment in fragments:
        start = int(fragment["start"])
        end = int(fragment["end"])
        if start < 1 or end < start or end > len(sequence):
            return ""
        pieces.append(sequence[start - 1 : end])
    return "".join(pieces)


def span_length(fragments: list[dict[str, Any]]) -> int:
    return sum(int(item["end"]) - int(item["start"]) + 1 for item in fragments)


def record_id(protein_id: str, cls: str, node_name: str, entry_accession: str, index: int) -> str:
    return f"{protein_id}|{cls}|{node_name}|{entry_accession}|{index}"


def make_record(
    *,
    source: str,
    protein_meta: dict[str, Any],
    sequence: str,
    entry: dict[str, Any],
    location: dict[str, Any],
    node_name: str,
    node_kind: str,
    substructure_class: str,
    record_index: int,
) -> dict[str, Any] | None:
    protein_id = str(protein_meta.get("accession") or "")
    if not protein_id:
        return None
    entry_accession = str(entry.get("accession") or "")
    fragments = list(location.get("fragments", []) or [])
    seq_fragment = fragment_sequence(sequence, fragments)
    if not seq_fragment:
        return None
    return {
        "record_id": record_id(protein_id, substructure_class, node_name, entry_accession, record_index),
        "source": source,
        "protein_id": protein_id,
        "protein_name": protein_meta.get("id") or protein_meta.get("name"),
        "protein_description": protein_meta.get("description"),
        "source_database": protein_meta.get("source_database"),
        "source_organism": protein_meta.get("source_organism"),
        "protein_length": protein_meta.get("length") or len(sequence),
        "substructure_class": substructure_class,
        "substructure_id": node_name,
        "node_name": node_name,
        "node_kind": node_kind,
        "entry_accession": entry_accession,
        "entry_database": entry.get("source_database"),
        "entry_type": entry.get("entry_type"),
        "entry_integrated": entry.get("entry_integrated"),
        "match_id": location.get("model") or entry_accession,
        "substructure_name": TETR_DOMAIN_ENTRIES.get(entry_accession, {}).get("name") or entry_accession,
        "representative": bool(location.get("representative", False)),
        "score": location.get("score"),
        "residue_spans": [[int(f["start"]), int(f["end"])] for f in fragments],
        "span_indexing": "interpro_1_based_closed",
        "is_discontinuous": len(fragments) > 1,
        "span_length": span_length(fragments),
        "sequence_fragment": seq_fragment,
        "has_sequence": True,
    }


def synthetic_region_record(
    *,
    source: str,
    protein_meta: dict[str, Any],
    sequence: str,
    start: int,
    end: int,
    node_name: str,
    node_kind: str,
    substructure_class: str,
    entry_accession: str,
    record_index: int,
) -> dict[str, Any] | None:
    if start < 1 or end < start or end > len(sequence):
        return None
    fragment = sequence[start - 1 : end]
    if not fragment:
        return None
    protein_id = str(protein_meta.get("accession") or "")
    return {
        "record_id": record_id(protein_id, substructure_class, node_name, entry_accession, record_index),
        "source": source,
        "protein_id": protein_id,
        "protein_name": protein_meta.get("id") or protein_meta.get("name"),
        "protein_description": protein_meta.get("description"),
        "source_database": protein_meta.get("source_database"),
        "source_organism": protein_meta.get("source_organism"),
        "protein_length": protein_meta.get("length") or len(sequence),
        "substructure_class": substructure_class,
        "substructure_id": node_name,
        "node_name": node_name,
        "node_kind": node_kind,
        "entry_accession": entry_accession,
        "entry_database": "derived",
        "entry_type": "derived_region",
        "entry_integrated": None,
        "match_id": entry_accession,
        "substructure_name": node_name,
        "representative": False,
        "score": None,
        "residue_spans": [[start, end]],
        "span_indexing": "interpro_1_based_closed",
        "is_discontinuous": False,
        "span_length": end - start + 1,
        "sequence_fragment": fragment,
        "has_sequence": True,
    }


def top_residues(freq: dict[str, float], n: int = 10) -> list[str]:
    return [aa for aa, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:n]]


def aa_frequency(records: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter()
    total = 0
    for rec in records:
        for aa in rec.get("sequence_fragment", ""):
            if aa in AA:
                counts[aa] += 1
                total += 1
    if total <= 0:
        return {}
    return {aa: counts[aa] / total for aa in AA}


def class_bias(freq: dict[str, float]) -> list[str]:
    if not freq:
        return []
    out = []
    if sum(freq.get(aa, 0.0) for aa in AROMATIC) >= 0.10:
        out.append("aromatic")
    if sum(freq.get(aa, 0.0) for aa in POLAR) >= 0.20:
        out.append("polar_uncharged")
    if sum(freq.get(aa, 0.0) for aa in CHARGED) >= 0.14:
        out.append("contextual_charge")
    if sum(freq.get(aa, 0.0) for aa in HYDROPHOBIC) >= 0.35:
        out.append("helix_compatible")
    return out


def build_prior_cache(records: list[dict[str, Any]], embedding_manifest: str) -> dict[str, Any]:
    by_kind_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_node_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_kind_records[str(rec.get("node_kind") or "unknown")].append(rec)
        by_node_records[str(rec.get("node_name") or rec.get("substructure_id") or "unknown")].append(rec)

    kind_priority = {
        "pocket": 1.12,
        "hinge": 1.05,
        "relay": 1.04,
        "dna_contact": 0.68,
        "dimer_interface": 0.72,
        "support": 0.42,
    }
    by_kind: dict[str, Any] = {}
    for kind, items in sorted(by_kind_records.items()):
        freq = aa_frequency(items)
        by_kind[kind] = {
            "priority_boost": kind_priority.get(kind, 1.0),
            "confidence": min(0.65, 0.25 + 0.01 * len(items)),
            "aa_weights": freq,
            "favored_residues": top_residues(freq),
            "favored_residue_classes": class_bias(freq),
            "disfavored_residues": ["C"],
            "source": f"interpro_tetr_records:{kind}",
            "support_count": len(items),
        }

    nodes: dict[str, Any] = {}
    node_priority = {
        "ligand_pocket_core": 1.22,
        "allosteric_hinge": 1.10,
        "relay_helix": 1.06,
        "domain_connector": 1.04,
        "HTH_exit_loop": 0.88,
        "HTH_recognition_helix": 0.55,
        "dimerization_interface": 0.62,
    }
    for node, items in sorted(by_node_records.items()):
        freq = aa_frequency(items)
        prior = {
            "priority_boost": node_priority.get(node, 1.0),
            "confidence": min(0.70, 0.30 + 0.01 * len(items)),
            "aa_weights": freq,
            "favored_residues": top_residues(freq),
            "favored_residue_classes": class_bias(freq),
            "disfavored_residues": ["C"],
            "source": f"interpro_tetr_records:{node}",
            "support_count": len(items),
        }
        if node == "ligand_pocket_core":
            prior["favored_residues"] = sorted(
                set(prior["favored_residues"]) | {"D", "E", "Y", "H", "S", "T", "N", "Q", "W", "F"}
            )
            prior["favored_residue_classes"] = sorted(
                set(prior["favored_residue_classes"]) | {"aromatic", "polar_uncharged", "contextual_charge"}
            )
            prior["source"] = "interpro_tetr_c_domain_records_plus_dopamine_chemical_overlay"
            prior["confidence"] = min(0.55, float(prior["confidence"]))
        nodes[node] = prior

    return {
        "metadata": {
            "provider_schema": "astevolve_external_prior_v1",
            "source": "downloaded_interpro_tetr_records",
            "description": "Generated from InterPro/Pfam TetR protein matches and annotated domain fragments. Dopamine-specific terms are only a low-confidence chemical overlay on top of TetR C-terminal domain statistics.",
            "do_not_copy_sequences": True,
            "record_count": len(records),
        },
        "retrieval": {
            "embedding_manifest": embedding_manifest,
            "match_node_name": False,
            "node_kind_filters": {
                "pocket": {"class_filter": ["TetR_C_domain"], "match_node_name": False},
                "hinge": {"class_filter": ["TetR_connector", "TetR_C_domain"], "match_node_name": False},
                "relay": {"class_filter": ["TetR_C_domain"], "match_node_name": False},
                "dna_contact": {"class_filter": ["TetR_HTH_domain", "TetR_HTH_conserved_site"], "match_node_name": False},
                "dimer_interface": {"class_filter": ["TetR_C_domain", "TetR_full_regulator"], "match_node_name": False},
                "support": {"class_filter": ["TetR_full_regulator"], "match_node_name": False},
            },
            "node_filters": {
                node: {"class_filter": sorted({r["substructure_class"] for r in items}), "match_node_name": False}
                for node, items in sorted(by_node_records.items())
            },
        },
        "defaults": {
            "priority_boost": 1.0,
            "confidence": 0.25,
            "disfavored_residues": ["C"],
        },
        "by_kind": by_kind,
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TetR-family InterPro/Pfam matches and build ASTevolve KB records."
    )
    parser.add_argument("--out-dir", type=Path, default=project_root() / "data" / "tetr_kb")
    parser.add_argument("--records-name", default="tetr_interpro_records.jsonl")
    parser.add_argument("--prior-name", default="tetr_interpro_external_prior_cache.json")
    parser.add_argument("--manifest-name", default="tetr_interpro_records.manifest.json")
    parser.add_argument("--embedding-manifest-name", default="embedding_manifest_esm2_t6_8M_tetr_interpro.json")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--min-length", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=280)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--include-full", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / args.records_name
    prior_path = args.out_dir / args.prior_name
    manifest_path = args.out_dir / args.manifest_name

    protein_seen: set[str] = set()
    records: list[dict[str, Any]] = []
    skipped = Counter()

    for db, accession, _name in TETR_ENTRY_SEEDS:
        for item in iter_entry_proteins(
            db,
            accession,
            page_size=args.page_size,
            max_proteins=args.max_proteins,
            sleep_seconds=args.sleep,
        ):
            protein_id = metadata_accession(item)
            if not protein_id or protein_id in protein_seen:
                continue
            protein_seen.add(protein_id)

            try:
                pfam_payload = fetch_pfam_payload(protein_id, args.sleep)
            except Exception:
                skipped["pfam_fetch_failed"] += 1
                continue
            protein_meta = dict((pfam_payload.get("metadata") or {}))
            sequence = normalize_sequence(protein_meta.get("sequence") or "")
            if not sequence:
                skipped["missing_sequence"] += 1
                continue

            length = int(protein_meta.get("length") or len(sequence))
            if args.reviewed_only and protein_meta.get("source_database") != "reviewed":
                skipped["not_reviewed"] += 1
                continue
            if length < args.min_length or length > args.max_length:
                skipped["length_filter"] += 1
                continue

            protein_meta.setdefault("accession", protein_id)
            entries = list(pfam_payload.get("entry_subset", []) or [])
            if not entries:
                entries = list(item.get("entries", []) or [])
            pf00440_entries = [entry for entry in entries if entry.get("accession") == "PF00440"]
            pf02909_entries = [entry for entry in entries if entry.get("accession") == "PF02909"]
            if not pf00440_entries:
                skipped["missing_pf00440_in_detail"] += 1
                continue

            record_start = len(records)
            hth_end = 0
            for entry in pf00440_entries + pf02909_entries:
                cfg = TETR_DOMAIN_ENTRIES.get(str(entry.get("accession") or ""))
                if not cfg:
                    continue
                for location in iter_locations(entry):
                    rec = make_record(
                        source="interpro_api_tetr_family",
                        protein_meta=protein_meta,
                        sequence=sequence,
                        entry=entry,
                        location=location,
                        node_name=cfg["node_name"],
                        node_kind=cfg["node_kind"],
                        substructure_class=cfg["substructure_class"],
                        record_index=len(records),
                    )
                    if rec is None:
                        continue
                    records.append(rec)
                    if entry.get("accession") == "PF00440":
                        hth_end = max(hth_end, max(span[1] for span in rec["residue_spans"]))

            if hth_end > 0:
                connector = synthetic_region_record(
                    source="interpro_api_tetr_family",
                    protein_meta=protein_meta,
                    sequence=sequence,
                    start=hth_end + 1,
                    end=min(len(sequence), hth_end + 12),
                    node_name="domain_connector",
                    node_kind="hinge",
                    substructure_class="TetR_connector",
                    entry_accession="derived_connector_after_PF00440",
                    record_index=len(records),
                )
                if connector:
                    records.append(connector)

                if not pf02909_entries and hth_end + 16 <= len(sequence):
                    c_term = synthetic_region_record(
                        source="interpro_api_tetr_family",
                        protein_meta=protein_meta,
                        sequence=sequence,
                        start=hth_end + 1,
                        end=len(sequence),
                        node_name="TetR_C_domain_context",
                        node_kind="support",
                        substructure_class="TetR_C_domain",
                        entry_accession="derived_c_terminal_after_PF00440",
                        record_index=len(records),
                    )
                    if c_term:
                        records.append(c_term)

                c_start = min(len(sequence), hth_end + 13)
                derived_layout = [
                    ("ligand_pocket_core", "pocket", "TetR_pocket_context", 52),
                    ("allosteric_hinge", "hinge", "TetR_hinge_context", 18),
                    ("relay_helix", "relay", "TetR_relay_context", 28),
                    ("dimerization_interface", "dimer_interface", "TetR_dimer_context", 28),
                ]
                cursor = c_start
                for node_name, node_kind, cls, width in derived_layout:
                    if cursor > len(sequence):
                        break
                    end = min(len(sequence), cursor + width - 1)
                    if end - cursor + 1 >= max(8, min(width, 12)):
                        derived = synthetic_region_record(
                            source="interpro_api_tetr_family",
                            protein_meta=protein_meta,
                            sequence=sequence,
                            start=cursor,
                            end=end,
                            node_name=node_name,
                            node_kind=node_kind,
                            substructure_class=cls,
                            entry_accession=f"derived_{node_name}_after_PF00440",
                            record_index=len(records),
                        )
                        if derived:
                            records.append(derived)
                    cursor = end + 1

                if cursor <= len(sequence) and len(sequence) - cursor + 1 >= 8:
                    support = synthetic_region_record(
                        source="interpro_api_tetr_family",
                        protein_meta=protein_meta,
                        sequence=sequence,
                        start=cursor,
                        end=len(sequence),
                        node_name="C_terminal_support",
                        node_kind="support",
                        substructure_class="TetR_C_terminal_support",
                        entry_accession="derived_c_terminal_support_after_PF00440",
                        record_index=len(records),
                    )
                    if support:
                        records.append(support)

            if args.include_full:
                full = synthetic_region_record(
                    source="interpro_api_tetr_family",
                    protein_meta=protein_meta,
                    sequence=sequence,
                    start=1,
                    end=len(sequence),
                    node_name="TetR_full_regulator",
                    node_kind="support",
                    substructure_class="TetR_full_regulator",
                    entry_accession="derived_full_sequence",
                    record_index=len(records),
                )
                if full:
                    records.append(full)

            if len(records) == record_start:
                skipped["no_records_written_for_protein"] += 1

    with records_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    prior = build_prior_cache(records, args.embedding_manifest_name)
    prior_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "repository_scope": "TetR-family allosteric transcription factor scaffold KB",
        "repository_intent": "Provide TetR-specific annotated sequence fragments and residue priors for ASTevolve external-KB guidance; evaluator metrics remain responsible for dopamine binding and allosteric switch selection.",
        "source": "InterPro API",
        "api_root": API_ROOT,
        "seed_entries": [
            {"database": db, "accession": accession, "name": name}
            for db, accession, name in TETR_ENTRY_SEEDS
        ],
        "records": str(records_path),
        "prior": str(prior_path),
        "embedding_manifest_expected": str(args.out_dir / args.embedding_manifest_name),
        "max_proteins": args.max_proteins,
        "page_size": args.page_size,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "reviewed_only": args.reviewed_only,
        "include_full": args.include_full,
        "proteins_seen": len(protein_seen),
        "records_written": len(records),
        "class_counts": dict(Counter(rec["substructure_class"] for rec in records)),
        "node_kind_counts": dict(Counter(rec["node_kind"] for rec in records)),
        "skipped": dict(skipped),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
