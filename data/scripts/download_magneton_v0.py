from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm


REPO_ID = "rcalef/magneton-data"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))

FILES = [
    "interpro_103.0/debug_subset/swissprot.with_ss.0.jsonl.gz",
    "interpro_103.0/debug_subset/index.tsv",
    "sequences/swissprot_subset.tsv",
    "sequences/uniprot_sprot.fasta.gz",
    "interpro_103.0/labels/selected_subset/Active_site.labels.tsv",
    "interpro_103.0/labels/selected_subset/Binding_site.labels.tsv",
    "interpro_103.0/labels/selected_subset/Conserved_site.labels.tsv",
    "interpro_103.0/labels/selected_subset/Domain.labels.tsv",
    "interpro_103.0/labels/selected_subset/Homologous_superfamily.labels.tsv",
]


def make_url(endpoint: str, repo_id: str, rel_path: str) -> str:
    safe_path = quote(rel_path, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{safe_path}"


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = out_path.stat().st_size if out_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        if response.status_code == 416:
            print(f"[skip] already complete: {out_path}")
            return

        response.raise_for_status()

        append = existing > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        if not append:
            existing = 0

        content_length = int(response.headers.get("content-length", 0))
        total = existing + content_length if content_length else None

        with tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=out_path.name,
        ) as progress:
            with out_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the minimal Magneton files for ASTevolve external KB v0."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "magneton_raw",
        help="Output directory for Magneton raw files.",
    )
    parser.add_argument(
        "--endpoint",
        default="https://huggingface.co",
        help="Use https://hf-mirror.com if HuggingFace is slow or inaccessible.",
    )
    args = parser.parse_args()

    print(f"[repo] {REPO_ID}")
    print(f"[endpoint] {args.endpoint}")
    print(f"[out] {args.out.resolve()}")

    for rel_path in FILES:
        print(f"\n[download] {rel_path}")
        download_file(make_url(args.endpoint, REPO_ID, rel_path), args.out / rel_path)

    print("\n[done] Magneton debug files are ready.")


if __name__ == "__main__":
    main()
