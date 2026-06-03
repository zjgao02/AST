from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("KRDE")
POLAR = set("STNQY")
AROMATIC = set("FWYH")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def residue_features(seq: str) -> dict[str, float]:
    seq = "".join(aa for aa in seq if aa in AA_SET)
    length = len(seq)
    if length == 0:
        return {
            "hydrophobic_frac": 0.0,
            "charged_frac": 0.0,
            "polar_frac": 0.0,
            "aromatic_frac": 0.0,
        }
    return {
        "hydrophobic_frac": sum(aa in HYDROPHOBIC for aa in seq) / length,
        "charged_frac": sum(aa in CHARGED for aa in seq) / length,
        "polar_frac": sum(aa in POLAR for aa in seq) / length,
        "aromatic_frac": sum(aa in AROMATIC for aa in seq) / length,
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ASTevolve Magneton substructure records into class priors."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DATA_ROOT / "external_kb" / "substructure_records.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "external_kb" / "substructure_summary.json",
    )
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    length_by_class: dict[str, list[float]] = defaultdict(list)
    aa_counts_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    feature_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    representative_counts: Counter[str] = Counter()
    unique_types: dict[str, set[str]] = defaultdict(set)

    total = 0
    with_sequence = 0
    for rec in iter_records(args.records):
        total += 1
        cls = rec.get("substructure_class") or "Unknown"
        counts[cls] += 1
        if rec.get("representative"):
            representative_counts[cls] += 1
        if rec.get("substructure_id"):
            unique_types[cls].add(str(rec["substructure_id"]))

        span_len = rec.get("span_length")
        if span_len is not None:
            length_by_class[cls].append(float(span_len))

        seq = rec.get("sequence_fragment") or ""
        if seq:
            with_sequence += 1
            aa_counts_by_class[cls].update(aa for aa in seq if aa in AA_SET)
            feats = residue_features(seq)
            for key, value in feats.items():
                feature_values[cls][key].append(value)

    class_summaries = {}
    for cls in sorted(counts):
        aa_total = sum(aa_counts_by_class[cls].values())
        aa_freq = {
            aa: (aa_counts_by_class[cls][aa] / aa_total if aa_total else 0.0)
            for aa in AA
        }
        class_summaries[cls] = {
            "count": counts[cls],
            "representative_count": representative_counts[cls],
            "unique_substructure_ids": len(unique_types[cls]),
            "length": percentile_summary(length_by_class[cls]),
            "aa_frequency": aa_freq,
            "feature_summary": {
                feature: percentile_summary(values)
                for feature, values in sorted(feature_values[cls].items())
            },
        }

    summary = {
        "records_path": str(args.records),
        "total_records": total,
        "records_with_sequence": with_sequence,
        "classes": class_summaries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
