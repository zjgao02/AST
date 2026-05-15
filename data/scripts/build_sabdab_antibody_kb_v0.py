from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SAbDab-derived antibody_kb in the same style as the Magneton mini KB."
    )
    parser.add_argument("--summary", default=str(PROJECT_ROOT / "data/sabdab_raw/sabdab_summary.tsv"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data/antibody_kb"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    summary = Path(args.summary)
    out_dir = Path(args.out_dir)
    records = out_dir / "sabdab_records.jsonl"
    manifest = out_dir / "sabdab_records.manifest.json"
    cdr_html = Path(str(summary)).with_name("cdrs_chothia.html")
    cdr_records = out_dir / "sabdab_cdr_records.jsonl"
    cdr_manifest = out_dir / "sabdab_cdr_records.manifest.json"
    summary_json = out_dir / "sabdab_summary.json"
    prior_json = out_dir / "sabdab_external_prior_cache.json"

    if not args.skip_download and not summary.exists():
        run([
            sys.executable,
            str(HERE / "download_sabdab_v0.py"),
            "--out",
            str(summary),
        ])

    if not summary.exists():
        raise SystemExit(
            f"SAbDab summary TSV not found: {summary}\n"
            "Either run without --skip-download or manually download the summary TSV first."
        )

    build_cmd = [
        sys.executable,
        str(HERE / "build_sabdab_records_v0.py"),
        "--summary",
        str(summary),
        "--out",
        str(records),
        "--manifest",
        str(manifest),
    ]
    if args.max_rows is not None:
        build_cmd += ["--max-rows", str(args.max_rows)]
    run(build_cmd)

    if not cdr_html.exists():
        run([
            sys.executable,
            str(HERE / "download_sabdab_cdrs_v0.py"),
            "--cdr-def",
            "Chothia",
            "--out",
            str(cdr_html),
        ])

    run([
        sys.executable,
        str(HERE / "build_sabdab_cdr_records_v0.py"),
        "--html",
        str(cdr_html),
        "--out",
        str(cdr_records),
        "--manifest",
        str(cdr_manifest),
    ])

    run([
        sys.executable,
        str(HERE / "summarize_sabdab_records_v0.py"),
        "--records",
        str(cdr_records),
        "--out",
        str(summary_json),
        "--prior-out",
        str(prior_json),
    ])

    print("\n[done] antibody_kb written:")
    print(f"  pair records : {records}")
    print(f"  pair manifest: {manifest}")
    print(f"  cdr records  : {cdr_records}")
    print(f"  cdr manifest : {cdr_manifest}")
    print(f"  summary : {summary_json}")
    print(f"  prior   : {prior_json}")


if __name__ == "__main__":
    main()
