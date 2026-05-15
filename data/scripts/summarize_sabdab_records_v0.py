from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


AA = "ACDEFGHIKLMNPQRSTVWY"
AROMATIC = set("FYW")
CHARGED = set("KRHDE")
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQY")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _features(seq: str) -> Dict[str, float]:
    if not seq:
        return {}
    n = len(seq)
    return {
        "aromatic_frac": sum(c in AROMATIC for c in seq) / n,
        "charged_frac": sum(c in CHARGED for c in seq) / n,
        "hydrophobic_frac": sum(c in HYDROPHOBIC for c in seq) / n,
        "polar_frac": sum(c in POLAR for c in seq) / n,
    }


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(float(x) for x in values)

    def q(frac: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round(frac * (len(vals) - 1)))))
        return vals[idx]

    return {
        "min": vals[0],
        "p10": q(0.10),
        "median": q(0.50),
        "mean": float(mean(vals)),
        "p90": q(0.90),
        "max": vals[-1],
    }


def _aa_frequency(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {aa: float(counter.get(aa, 0) / total) for aa in AA}


def summarize(records_path: Path, out_summary: Path, out_prior: Path) -> Dict[str, Any]:
    total_records = 0
    records_with_sequence = 0
    classes: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "with_sequence": 0,
            "lengths": [],
            "aa_counter": Counter(),
            "features": defaultdict(list),
        }
    )
    node_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "lengths": [],
            "aa_counter": Counter(),
            "features": defaultdict(list),
            "antigen_types": Counter(),
            "antigen_names": Counter(),
        }
    )
    antigen_type_counts: Counter[str] = Counter()

    for rec in _read_jsonl(records_path):
        total_records += 1
        cls = str(rec.get("substructure_class") or "unknown")
        seq = str(rec.get("sequence_fragment") or "")
        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}

        block = classes[cls]
        block["count"] += 1
        antigen_type = str(meta.get("antigen_type") or "unknown").lower()
        antigen_type_counts[antigen_type] += 1

        if seq:
            records_with_sequence += 1
            block["with_sequence"] += 1
            block["lengths"].append(len(seq))
            block["aa_counter"].update(seq)
            for k, v in _features(seq).items():
                block["features"][k].append(v)

        node_name = str(meta.get("node_name") or rec.get("substructure_id") or "")
        if cls == "Antibody_CDR" and node_name:
            node = node_stats[node_name]
            node["count"] += 1
            node["antigen_types"][antigen_type] += 1
            antigen_name = str(meta.get("antigen_name") or "")
            if antigen_name:
                node["antigen_names"][antigen_name] += 1
            if seq:
                node["lengths"].append(len(seq))
                node["aa_counter"].update(seq)
                for k, v in _features(seq).items():
                    node["features"][k].append(v)

    class_summary = {}
    for cls, block in classes.items():
        class_summary[cls] = {
            "count": block["count"],
            "records_with_sequence": block["with_sequence"],
            "length": _quantiles(block["lengths"]),
            "aa_frequency": _aa_frequency(block["aa_counter"]),
            "feature_summary": {
                key: _quantiles(vals)
                for key, vals in block["features"].items()
            },
        }

    node_summary = {}
    for node_name, block in node_stats.items():
        node_summary[node_name] = {
            "count": block["count"],
            "length": _quantiles(block["lengths"]),
            "aa_frequency": _aa_frequency(block["aa_counter"]),
            "feature_summary": {
                key: _quantiles(vals)
                for key, vals in block["features"].items()
            },
            "antigen_type_counts": dict(block["antigen_types"].most_common(20)),
            "top_antigen_names": dict(block["antigen_names"].most_common(20)),
        }

    summary = {
        "records_path": str(records_path),
        "total_records": total_records,
        "records_with_sequence": records_with_sequence,
        "classes": class_summary,
        "cdr_nodes": node_summary,
        "antigen_type_counts": dict(antigen_type_counts.most_common(50)),
    }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    prior = make_external_prior(summary, records_path)
    out_prior.parent.mkdir(parents=True, exist_ok=True)
    out_prior.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _top_residues(freq: Dict[str, float], n: int = 10) -> List[str]:
    return [aa for aa, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]


def _classes_from_feature(feature_summary: Dict[str, Any]) -> List[str]:
    out = []
    aromatic = feature_summary.get("aromatic_frac", {}).get("mean", 0.0)
    polar = feature_summary.get("polar_frac", {}).get("mean", 0.0)
    charged = feature_summary.get("charged_frac", {}).get("mean", 0.0)
    hydro = feature_summary.get("hydrophobic_frac", {}).get("mean", 0.0)
    if aromatic >= 0.10:
        out.append("aromatic")
    if polar >= 0.18:
        out.append("polar_uncharged")
    if charged >= 0.16:
        out.append("contextual_charge")
    if hydro >= 0.45:
        out.append("hydrophobic")
    return out or ["polar_uncharged"]


def make_external_prior(summary: Dict[str, Any], records_path: Path) -> Dict[str, Any]:
    antibody_cdr = summary.get("classes", {}).get("Antibody_CDR", {})
    cdr_freq = antibody_cdr.get("aa_frequency", {})

    nodes = {}
    for node_name, block in summary.get("cdr_nodes", {}).items():
        freq = block.get("aa_frequency", {})
        feature_summary = block.get("feature_summary", {})
        count = int(block.get("count") or 0)
        nodes[node_name] = {
            "priority_boost": 1.0 + min(0.45, count / 10000.0),
            "confidence": min(0.85, 0.25 + count / 6000.0),
            "aa_weights": freq,
            "favored_residues": _top_residues(freq, n=10),
            "favored_residue_classes": _classes_from_feature(feature_summary),
            "disfavored_residues": ["C"],
            "source": f"sabdab:{node_name}",
        }

    if "VH_CDR3" in nodes:
        nodes["VH_CDR3"]["priority_boost"] = max(1.35, float(nodes["VH_CDR3"]["priority_boost"]))
    if "VH_CDR2" in nodes:
        nodes["VH_CDR2"]["priority_boost"] = max(1.18, float(nodes["VH_CDR2"]["priority_boost"]))
    if "VL_CDR3" in nodes:
        nodes["VL_CDR3"]["priority_boost"] = max(1.15, float(nodes["VL_CDR3"]["priority_boost"]))

    return {
        "metadata": {
            "provider_schema": "astevolve_external_prior_v1",
            "source": "sabdab_antibody_kb_summary",
            "records_path": str(records_path),
            "description": "SAbDab-derived antibody-antigen prior cache for ASTevolve MCTS.",
            "do_not_copy_sequences": True,
        },
        "defaults": {
            "priority_boost": 1.0,
            "confidence": 0.40,
            "disfavored_residues": ["C"],
        },
        "by_kind": {
            "cdr": {
                "priority_boost": 1.12,
                "confidence": 0.55,
                "aa_weights": cdr_freq,
                "favored_residues": _top_residues(cdr_freq, n=10),
                "favored_residue_classes": ["aromatic", "polar_uncharged", "contextual_charge"],
                "disfavored_residues": ["C"],
                "source": "sabdab:Antibody_CDR",
            },
            "linker": {
                "priority_boost": 0.65,
                "confidence": 0.45,
                "favored_residue_classes": ["flexible_small"],
                "favored_residues": ["G", "S", "A", "T"],
                "disfavored_residues": ["C", "W", "F", "I", "L", "V"],
                "source": "scfv_linker_default",
            },
        },
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ASTevolve SAbDab antibody_kb records.")
    parser.add_argument("--records", default="D:/Downloads/ast/data/antibody_kb/sabdab_records.jsonl")
    parser.add_argument("--out", default="D:/Downloads/ast/data/antibody_kb/sabdab_summary.json")
    parser.add_argument("--prior-out", default="D:/Downloads/ast/data/antibody_kb/sabdab_external_prior_cache.json")
    args = parser.parse_args()

    summary = summarize(Path(args.records), Path(args.out), Path(args.prior_out))
    print(json.dumps({
        "records_path": summary["records_path"],
        "total_records": summary["total_records"],
        "records_with_sequence": summary["records_with_sequence"],
        "classes": {k: v["count"] for k, v in summary["classes"].items()},
        "prior_out": args.prior_out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
