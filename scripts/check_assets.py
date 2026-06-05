from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astevolve.runtime.paths import data_path, model_path, project_root


ANTIBODY_KB_ASSETS = [
    data_path("antibody_kb", "sabdab_external_prior_cache.json"),
    data_path("antibody_kb", "embedding_manifest_esm2_t6_8M_sabdab_cdr.json"),
    data_path("antibody_kb", "embeddings_esm2_t6_8M_sabdab_cdr.npy"),
]

ATF_KB_ASSETS = [
    data_path("atf_kb", "atf_interpro_external_prior_cache.json"),
    data_path("atf_kb", "embedding_manifest_esm2_t6_8M_atf_interpro.json"),
    data_path("atf_kb", "embeddings_esm2_t6_8M_atf_interpro.npy"),
]

CASE_ASSETS = {
    "cd25_scfv": ANTIBODY_KB_ASSETS,
    "cd25_scfv_selectivity": ANTIBODY_KB_ASSETS,
    "pdl1_scfv_selectivity": ANTIBODY_KB_ASSETS,
    "proteor1_cdr_mask": ANTIBODY_KB_ASSETS,
    "tetr_dopamine": ATF_KB_ASSETS,
    "calcium_efhand_switch": ATF_KB_ASSETS,
    "pdz_peptide_selectivity": [],
}


def _discover_case_assets() -> Dict[str, List[Path]]:
    cases_root = PROJECT_ROOT / "cases"
    if not cases_root.exists():
        return dict(CASE_ASSETS)
    out = dict(CASE_ASSETS)
    for manifest in cases_root.glob("*/case.json"):
        out.setdefault(manifest.parent.name, [])
    return out

PROGEN2_SMALL_FILES = [
    "config.json",
    "configuration_progen.py",
    "generation_config.json",
    "modeling_progen.py",
    "tokenizer.json",
    "model.safetensors",
]


def _status(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def main() -> int:
    case_assets = _discover_case_assets()
    parser = argparse.ArgumentParser(description="Check ASTevolve local/server assets without loading models.")
    parser.add_argument("--case", choices=sorted(case_assets), default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    cases = [args.case] if args.case else sorted(case_assets)
    progen_dir = Path(os.environ.get("ASTEVOLVE_PROGEN_MODEL_DIR", model_path("progen2-small")))
    protenix_dir = Path(os.environ.get("ASTEVOLVE_PROTENIX_ROOT", model_path("protenix")))
    report: Dict[str, Any] = {
        "project_root": str(project_root()),
        "cases": {},
        "models": {
            "progen": {
                "directory": _status(progen_dir),
                "required_files": [_status(progen_dir / name) for name in PROGEN2_SMALL_FILES],
            },
            "protenix": {
                "directory": _status(protenix_dir),
                "note": "Protenix checkpoint filenames are version-specific; see model_weights/WEIGHTS_MANIFEST.txt.",
            },
        },
    }
    missing: List[str] = []
    for case_id in cases:
        entries = [_status(path) for path in case_assets[case_id]]
        report["cases"][case_id] = entries
        missing.extend(entry["path"] for entry in entries if not entry["exists"])
    missing.extend(
        entry["path"]
        for entry in report["models"]["progen"]["required_files"]
        if not entry["exists"]
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"project_root: {report['project_root']}")
        for case_id, entries in report["cases"].items():
            print(f"\n[{case_id}]")
            for entry in entries:
                marker = "ok" if entry["exists"] else "missing"
                size = f" {entry['bytes']} bytes" if entry["bytes"] is not None else ""
                print(f"- {marker}: {entry['path']}{size}")
        print("\n[models]")
        progen = report["models"]["progen"]
        marker = "ok" if progen["directory"]["exists"] else "missing"
        print(f"- progen: {marker}: {progen['directory']['path']}")
        for entry in progen["required_files"]:
            marker = "ok" if entry["exists"] else "missing"
            print(f"  - {marker}: {Path(entry['path']).name}")
        protenix = report["models"]["protenix"]
        marker = "ok" if protenix["directory"]["exists"] else "missing"
        print(f"- protenix: {marker}: {protenix['directory']['path']}")
        print(f"  - note: {protenix['note']}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
