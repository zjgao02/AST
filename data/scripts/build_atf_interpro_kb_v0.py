from __future__ import annotations

import argparse
import json
import ssl
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ATFFamily:
    family_id: str
    family_name: str
    seed_db: str
    seed_accession: str
    seed_name: str
    orientation: str = "n_terminal_dna_binding"
    default_min_length: int = 70
    default_max_length: int = 1200

    @property
    def class_prefix(self) -> str:
        return self.family_id.replace("-", "_")


# Curated major bacterial ligand-responsive/allosteric TF families.
# These are Pfam seeds; the script records the actual InterPro/Pfam metadata
# returned by the API in the manifest so this list remains auditable.
ATF_FAMILIES: dict[str, ATFFamily] = {
    "tetr": ATFFamily("tetr", "TetR/AcrR", "pfam", "PF00440", "TetR_N", "n_terminal_dna_binding", 140, 320),
    "laci": ATFFamily("laci", "LacI/GalR", "pfam", "PF00356", "LacI_HTH", "n_terminal_dna_binding", 220, 520),
    "lysr": ATFFamily("lysr", "LysR", "pfam", "PF00126", "HTH_LysR", "n_terminal_dna_binding", 220, 420),
    "gntr": ATFFamily("gntr", "GntR", "pfam", "PF00392", "GntR_HTH", "n_terminal_dna_binding", 150, 520),
    "marr": ATFFamily("marr", "MarR", "pfam", "PF01047", "MarR", "single_domain", 90, 260),
    "arac": ATFFamily("arac", "AraC/XylS", "pfam", "PF00165", "AraC_HTH", "c_terminal_dna_binding", 180, 520),
    "merr": ATFFamily("merr", "MerR", "pfam", "PF00376", "MerR", "n_terminal_dna_binding", 90, 260),
    "luxr": ATFFamily("luxr", "LuxR/FixJ", "pfam", "PF00196", "LuxR/GerE_HTH", "c_terminal_dna_binding", 130, 520),
    "iclr": ATFFamily("iclr", "IclR", "pfam", "PF01614", "IclR", "n_terminal_dna_binding", 180, 420),
    "crp": ATFFamily("crp", "CRP/FNR", "pfam", "PF00325", "CRP/FNR", "c_terminal_dna_binding", 140, 360),
    "arsr": ATFFamily("arsr", "ArsR/SmtB", "pfam", "PF01022", "ArsR", "single_domain", 70, 180),
    "ompr": ATFFamily("ompr", "OmpR/PhoB response regulator", "pfam", "PF00486", "Trans_reg_C", "c_terminal_dna_binding", 170, 320),
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


def node_name_for_family(family: ATFFamily, role: str) -> str:
    if family.family_id == "tetr":
        return {
            "dna": "HTH_recognition_helix",
            "connector": "domain_connector",
            "pocket": "ligand_pocket_core",
            "hinge": "allosteric_hinge",
            "relay": "relay_helix",
            "dimer": "dimerization_interface",
            "support": "C_terminal_support",
            "full": "TetR_full_regulator",
        }.get(role, f"tetr_{role}")
    return f"{family.family_id}_{role}"


def substructure_class(family: ATFFamily, role: str) -> str:
    return f"{family.class_prefix}_{role}_context"


def make_record(
    *,
    source: str,
    family: ATFFamily,
    protein_meta: dict[str, Any],
    sequence: str,
    node_name: str,
    node_kind: str,
    cls: str,
    entry_accession: str,
    entry_name: str,
    fragments: list[dict[str, Any]],
    record_index: int,
    representative: bool = False,
    model: str | None = None,
    score: Any = None,
    entry_database: str = "derived",
    entry_type: str = "derived_region",
    entry_integrated: Any = None,
) -> dict[str, Any] | None:
    protein_id = str(protein_meta.get("accession") or "")
    if not protein_id:
        return None
    seq_fragment = fragment_sequence(sequence, fragments)
    if not seq_fragment:
        return None

    return {
        "record_id": f"{protein_id}|{family.family_id}|{cls}|{node_name}|{entry_accession}|{record_index}",
        "source": source,
        "family_id": family.family_id,
        "family_name": family.family_name,
        "seed_accession": family.seed_accession,
        "seed_name": family.seed_name,
        "protein_id": protein_id,
        "protein_name": protein_meta.get("id") or protein_meta.get("name"),
        "protein_description": protein_meta.get("description"),
        "source_database": protein_meta.get("source_database"),
        "source_organism": protein_meta.get("source_organism"),
        "protein_length": protein_meta.get("length") or len(sequence),
        "substructure_class": cls,
        "substructure_id": node_name,
        "node_name": node_name,
        "node_kind": node_kind,
        "entry_accession": entry_accession,
        "entry_database": entry_database,
        "entry_type": entry_type,
        "entry_integrated": entry_integrated,
        "match_id": model or entry_accession,
        "substructure_name": entry_name or node_name,
        "representative": bool(representative),
        "score": score,
        "residue_spans": [[int(f["start"]), int(f["end"])] for f in fragments],
        "span_indexing": "interpro_1_based_closed",
        "is_discontinuous": len(fragments) > 1,
        "span_length": span_length(fragments),
        "sequence_fragment": seq_fragment,
        "has_sequence": True,
    }


def make_contiguous_record(
    *,
    source: str,
    family: ATFFamily,
    protein_meta: dict[str, Any],
    sequence: str,
    start: int,
    end: int,
    node_name: str,
    node_kind: str,
    cls: str,
    entry_accession: str,
    entry_name: str,
    record_index: int,
) -> dict[str, Any] | None:
    if start < 1 or end < start or end > len(sequence):
        return None
    return make_record(
        source=source,
        family=family,
        protein_meta=protein_meta,
        sequence=sequence,
        node_name=node_name,
        node_kind=node_kind,
        cls=cls,
        entry_accession=entry_accession,
        entry_name=entry_name,
        fragments=[{"start": int(start), "end": int(end)}],
        record_index=record_index,
    )


def non_dna_region(family: ATFFamily, sequence: str, dna_start: int, dna_end: int) -> tuple[int, int]:
    length = len(sequence)
    if family.orientation == "c_terminal_dna_binding":
        return 1, max(0, dna_start - 1)
    if family.orientation == "single_domain":
        return 1, length
    if family.orientation == "n_terminal_dna_binding":
        return min(length + 1, dna_end + 1), length
    if dna_end < int(0.55 * length):
        return min(length + 1, dna_end + 1), length
    return 1, max(0, dna_start - 1)


def add_derived_records(
    records: list[dict[str, Any]],
    *,
    family: ATFFamily,
    protein_meta: dict[str, Any],
    sequence: str,
    dna_start: int,
    dna_end: int,
    include_full: bool,
) -> None:
    source = "interpro_api_atf_family"
    length = len(sequence)
    reg_start, reg_end = non_dna_region(family, sequence, dna_start, dna_end)

    if family.orientation == "c_terminal_dna_binding":
        connector_start = max(1, dna_start - 12)
        connector_end = dna_start - 1
    else:
        connector_start = dna_end + 1
        connector_end = min(length, dna_end + 12)

    if connector_start <= connector_end and connector_end - connector_start + 1 >= 6:
        rec = make_contiguous_record(
            source=source,
            family=family,
            protein_meta=protein_meta,
            sequence=sequence,
            start=connector_start,
            end=connector_end,
            node_name=node_name_for_family(family, "connector"),
            node_kind="hinge",
            cls=substructure_class(family, "connector"),
            entry_accession=f"derived_{family.family_id}_connector",
            entry_name=f"{family.family_name} connector",
            record_index=len(records),
        )
        if rec:
            records.append(rec)

    if reg_start <= reg_end and reg_end - reg_start + 1 >= 12:
        reg_len = reg_end - reg_start + 1
        if reg_len < 70:
            layout = [("pocket", "pocket", reg_len)]
        else:
            layout = [
                ("pocket", "pocket", min(60, max(24, int(reg_len * 0.40)))),
                ("hinge", "hinge", min(28, max(12, int(reg_len * 0.15)))),
                ("relay", "relay", min(40, max(18, int(reg_len * 0.22)))),
                ("dimer", "dimer_interface", min(45, max(18, int(reg_len * 0.23)))),
            ]

        cursor = reg_start
        for role, kind, width in layout:
            if cursor > reg_end:
                break
            end = min(reg_end, cursor + width - 1)
            if end - cursor + 1 >= 8:
                rec = make_contiguous_record(
                    source=source,
                    family=family,
                    protein_meta=protein_meta,
                    sequence=sequence,
                    start=cursor,
                    end=end,
                    node_name=node_name_for_family(family, role),
                    node_kind=kind,
                    cls=substructure_class(family, role),
                    entry_accession=f"derived_{family.family_id}_{role}",
                    entry_name=f"{family.family_name} {role} context",
                    record_index=len(records),
                )
                if rec:
                    records.append(rec)
            cursor = end + 1

        if cursor <= reg_end and reg_end - cursor + 1 >= 8:
            rec = make_contiguous_record(
                source=source,
                family=family,
                protein_meta=protein_meta,
                sequence=sequence,
                start=cursor,
                end=reg_end,
                node_name=node_name_for_family(family, "support"),
                node_kind="support",
                cls=substructure_class(family, "support"),
                entry_accession=f"derived_{family.family_id}_support",
                entry_name=f"{family.family_name} support context",
                record_index=len(records),
            )
            if rec:
                records.append(rec)

    if include_full:
        rec = make_contiguous_record(
            source=source,
            family=family,
            protein_meta=protein_meta,
            sequence=sequence,
            start=1,
            end=length,
            node_name=node_name_for_family(family, "full"),
            node_kind="support",
            cls=substructure_class(family, "full"),
            entry_accession=f"derived_{family.family_id}_full_sequence",
            entry_name=f"{family.family_name} full regulator",
            record_index=len(records),
        )
        if rec:
            records.append(rec)


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


def aa_frequency_from_counts(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
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


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _add_summary(bucket: dict[str, Any], rec: dict[str, Any]) -> None:
    bucket["count"] += 1
    cls = rec.get("substructure_class")
    if cls:
        bucket["classes"].add(str(cls))
    for aa in rec.get("sequence_fragment", ""):
        if aa in AA:
            bucket["aa_counts"][aa] += 1


def build_prior_cache_from_records(records_path: Path, embedding_manifest: str) -> dict[str, Any]:
    by_kind_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "classes": set(), "aa_counts": Counter()}
    )
    by_node_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "classes": set(), "aa_counts": Counter()}
    )
    family_counts: Counter[str] = Counter()
    record_count = 0

    for rec in iter_records(records_path):
        record_count += 1
        family_counts[str(rec.get("family_id") or "unknown")] += 1
        kind = str(rec.get("node_kind") or "unknown")
        node = str(rec.get("node_name") or rec.get("substructure_id") or "unknown")
        _add_summary(by_kind_summary[kind], rec)
        _add_summary(by_node_summary[node], rec)

    kind_priority = {
        "pocket": 1.12,
        "hinge": 1.05,
        "relay": 1.04,
        "dna_contact": 0.68,
        "dimer_interface": 0.72,
        "support": 0.42,
    }
    by_kind: dict[str, Any] = {}
    for kind, summary in sorted(by_kind_summary.items()):
        freq = aa_frequency_from_counts(summary["aa_counts"])
        by_kind[kind] = {
            "priority_boost": kind_priority.get(kind, 1.0),
            "confidence": min(0.65, 0.25 + 0.002 * summary["count"]),
            "aa_weights": freq,
            "favored_residues": top_residues(freq),
            "favored_residue_classes": class_bias(freq),
            "disfavored_residues": ["C"],
            "source": f"interpro_atf_records:{kind}",
            "support_count": summary["count"],
        }

    nodes: dict[str, Any] = {}
    node_priority = {
        "ligand_pocket_core": 1.20,
        "allosteric_hinge": 1.10,
        "relay_helix": 1.06,
        "domain_connector": 1.04,
        "HTH_exit_loop": 0.88,
        "HTH_recognition_helix": 0.55,
        "dimerization_interface": 0.62,
    }
    for node, summary in sorted(by_node_summary.items()):
        freq = aa_frequency_from_counts(summary["aa_counts"])
        prior = {
            "priority_boost": node_priority.get(node, 1.0),
            "confidence": min(0.70, 0.30 + 0.002 * summary["count"]),
            "aa_weights": freq,
            "favored_residues": top_residues(freq),
            "favored_residue_classes": class_bias(freq),
            "disfavored_residues": ["C"],
            "source": f"interpro_atf_records:{node}",
            "support_count": summary["count"],
        }
        if node == "ligand_pocket_core":
            prior["favored_residues"] = sorted(
                set(prior["favored_residues"]) | {"D", "E", "Y", "H", "S", "T", "N", "Q", "W", "F"}
            )
            prior["favored_residue_classes"] = sorted(
                set(prior["favored_residue_classes"]) | {"aromatic", "polar_uncharged", "contextual_charge"}
            )
            prior["source"] = "interpro_atf_tetr_pocket_records_plus_dopamine_chemical_overlay"
            prior["confidence"] = min(0.55, float(prior["confidence"]))
        nodes[node] = prior

    class_by_kind = {
        kind: sorted(summary["classes"])
        for kind, summary in by_kind_summary.items()
    }
    return {
        "metadata": {
            "provider_schema": "astevolve_external_prior_v1",
            "source": "downloaded_interpro_atf_records",
            "description": "Generated from InterPro/Pfam matches for curated bacterial allosteric transcription factor families. These are search priors; case-specific evaluator metrics remain responsible for function.",
            "do_not_copy_sequences": True,
            "record_count": record_count,
            "family_counts": dict(family_counts),
        },
        "retrieval": {
            "embedding_manifest": embedding_manifest,
            "match_node_name": False,
            "node_kind_filters": {
                kind: {"class_filter": classes, "match_node_name": False}
                for kind, classes in sorted(class_by_kind.items())
            },
            "node_filters": {
                node: {"class_filter": sorted(summary["classes"]), "match_node_name": False}
                for node, summary in sorted(by_node_summary.items())
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


def selected_families(value: str) -> list[ATFFamily]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(ATF_FAMILIES.values())
    unknown = [name for name in names if name not in ATF_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown ATF families: {unknown}; valid={sorted(ATF_FAMILIES)}")
    return [ATF_FAMILIES[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download InterPro/Pfam matches for curated allosteric transcription factor families."
    )
    parser.add_argument("--out-dir", type=Path, default=project_root() / "data" / "atf_kb")
    parser.add_argument("--records-name", default="atf_interpro_records.jsonl")
    parser.add_argument("--prior-name", default="atf_interpro_external_prior_cache.json")
    parser.add_argument("--manifest-name", default="atf_interpro_records.manifest.json")
    parser.add_argument("--embedding-manifest-name", default="embedding_manifest_esm2_t6_8M_atf_interpro.json")
    parser.add_argument("--families", default="all", help=f"Comma list or all. Valid: {','.join(sorted(ATF_FAMILIES))}")
    parser.add_argument("--max-proteins-per-family", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument(
        "--fetch-all-pfam-entries",
        action="store_true",
        help="Fetch each protein's full Pfam entry list. Slower; seed-only mode is enough for ATF KB fragments.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    families = selected_families(args.families)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / args.records_name
    prior_path = args.out_dir / args.prior_name
    manifest_path = args.out_dir / args.manifest_name

    global_seen: set[tuple[str, str]] = set()
    skipped = Counter()
    family_seen = Counter()
    family_records = Counter()
    records_written = 0
    class_counts: Counter[str] = Counter()
    node_kind_counts: Counter[str] = Counter()
    family_errors: dict[str, str] = {}

    with records_path.open("w", encoding="utf-8") as handle:
        for family in families:
            try:
                items = iter_entry_proteins(
                    family.seed_db,
                    family.seed_accession,
                    page_size=args.page_size,
                    max_proteins=args.max_proteins_per_family,
                    sleep_seconds=args.sleep,
                )
                for item in items:
                    protein_id = metadata_accession(item)
                    if not protein_id:
                        skipped[f"{family.family_id}:missing_protein_id"] += 1
                        continue
                    key = (family.family_id, protein_id)
                    if key in global_seen:
                        skipped[f"{family.family_id}:duplicate"] += 1
                        continue
                    global_seen.add(key)
                    family_seen[family.family_id] += 1

                    try:
                        if args.fetch_all_pfam_entries:
                            pfam_payload = fetch_pfam_payload(protein_id, args.sleep)
                            protein_meta = dict((pfam_payload.get("metadata") or {}))
                            entries = list(pfam_payload.get("entry_subset", []) or [])
                        else:
                            protein_payload = fetch_protein(protein_id, args.sleep)
                            protein_meta = dict((protein_payload.get("metadata") or {}))
                            entries = list(item.get("entries", []) or [])
                    except Exception as exc:
                        skipped[f"{family.family_id}:protein_fetch_failed"] += 1
                        family_errors.setdefault(family.family_id, str(exc))
                        continue

                    sequence = normalize_sequence(protein_meta.get("sequence") or "")
                    if not sequence:
                        skipped[f"{family.family_id}:missing_sequence"] += 1
                        continue

                    length = int(protein_meta.get("length") or len(sequence))
                    min_len = args.min_length or family.default_min_length
                    max_len = args.max_length or family.default_max_length
                    if args.reviewed_only and protein_meta.get("source_database") != "reviewed":
                        skipped[f"{family.family_id}:not_reviewed"] += 1
                        continue
                    if length < min_len or length > max_len:
                        skipped[f"{family.family_id}:length_filter"] += 1
                        continue

                    protein_meta.setdefault("accession", protein_id)
                    seed_entries = [entry for entry in entries if entry.get("accession") == family.seed_accession]
                    if not seed_entries:
                        seed_entries = [
                            entry
                            for entry in item.get("entries", []) or []
                            if entry.get("accession") == family.seed_accession
                        ]
                    if not seed_entries:
                        skipped[f"{family.family_id}:missing_seed_in_detail"] += 1
                        continue

                    protein_records: list[dict[str, Any]] = []
                    seed_start = length
                    seed_end = 1
                    for entry in seed_entries:
                        for location in iter_locations(entry):
                            rec = make_record(
                                source="interpro_api_atf_family",
                                family=family,
                                protein_meta=protein_meta,
                                sequence=sequence,
                                node_name=node_name_for_family(family, "dna"),
                                node_kind="dna_contact",
                                cls=substructure_class(family, "dna"),
                                entry_accession=str(entry.get("accession") or family.seed_accession),
                                entry_name=str(entry.get("entry_name") or family.seed_name),
                                fragments=list(location.get("fragments", []) or []),
                                record_index=len(protein_records),
                                representative=bool(location.get("representative", False)),
                                model=location.get("model"),
                                score=location.get("score"),
                                entry_database=str(entry.get("source_database") or family.seed_db),
                                entry_type=str(entry.get("entry_type") or "domain"),
                                entry_integrated=entry.get("entry_integrated"),
                            )
                            if rec is None:
                                continue
                            protein_records.append(rec)
                            starts = [span[0] for span in rec["residue_spans"]]
                            ends = [span[1] for span in rec["residue_spans"]]
                            seed_start = min(seed_start, min(starts))
                            seed_end = max(seed_end, max(ends))

                    if seed_start <= seed_end:
                        add_derived_records(
                            protein_records,
                            family=family,
                            protein_meta=protein_meta,
                            sequence=sequence,
                            dna_start=seed_start,
                            dna_end=seed_end,
                            include_full=args.include_full,
                        )

                    if not protein_records:
                        skipped[f"{family.family_id}:no_records_written_for_protein"] += 1
                        continue

                    for rec in protein_records:
                        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        records_written += 1
                        family_records[family.family_id] += 1
                        class_counts[str(rec["substructure_class"])] += 1
                        node_kind_counts[str(rec["node_kind"])] += 1
                    if args.progress_every > 0 and family_seen[family.family_id] % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "progress": "atf_download",
                                    "family": family.family_id,
                                    "proteins_seen": family_seen[family.family_id],
                                    "family_records": family_records[family.family_id],
                                    "total_records": records_written,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                print(
                    json.dumps(
                        {
                            "progress": "family_complete",
                            "family": family.family_id,
                            "proteins_seen": family_seen[family.family_id],
                            "family_records": family_records[family.family_id],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                family_errors.setdefault(family.family_id, str(exc))
                skipped[f"{family.family_id}:family_failed"] += 1

    prior = build_prior_cache_from_records(records_path, args.embedding_manifest_name)
    prior_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "repository_scope": "Allosteric transcription factor KB",
        "repository_intent": "Provide family-specific annotated sequence fragments and residue priors for ASTevolve external-KB guidance. Functional case evaluators should remain the main optimization objective.",
        "source": "InterPro API",
        "api_root": API_ROOT,
        "seed_entries": [
            {
                "family_id": family.family_id,
                "family_name": family.family_name,
                "database": family.seed_db,
                "accession": family.seed_accession,
                "name": family.seed_name,
                "orientation": family.orientation,
            }
            for family in families
        ],
        "records": str(records_path),
        "prior": str(prior_path),
        "embedding_manifest_expected": str(args.out_dir / args.embedding_manifest_name),
        "families": [family.family_id for family in families],
        "max_proteins_per_family": args.max_proteins_per_family,
        "page_size": args.page_size,
        "min_length_override": args.min_length,
        "max_length_override": args.max_length,
        "reviewed_only": args.reviewed_only,
        "include_full": args.include_full,
        "fetch_all_pfam_entries": args.fetch_all_pfam_entries,
        "progress_every": args.progress_every,
        "proteins_seen_by_family": dict(family_seen),
        "records_written": records_written,
        "records_by_family": dict(family_records),
        "class_counts": dict(class_counts),
        "node_kind_counts": dict(node_kind_counts),
        "skipped": dict(skipped),
        "family_errors": family_errors,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
