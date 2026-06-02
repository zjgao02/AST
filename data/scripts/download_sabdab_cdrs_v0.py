from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlopen


SABDAB_CDR_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/cdrsearch/?CDRdef_all={cdr_def}"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))


def download(url: str, out: Path, timeout: int = 120) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as resp, out.open("wb") as f:
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[sabdab-cdr] {done / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB", end="")
            else:
                print(f"\r[sabdab-cdr] {done / 1024 / 1024:.1f} MB", end="")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAbDab all-CDR HTML result page.")
    parser.add_argument("--cdr-def", default="Chothia", choices=["Chothia", "Contact", "IMGT", "Kabat", "North"])
    parser.add_argument("--out", default=str(DATA_ROOT / "sabdab_raw" / "cdrs_chothia.html"))
    parser.add_argument("--url", default=None)
    args = parser.parse_args()

    url = args.url or SABDAB_CDR_URL.format(cdr_def=args.cdr_def)
    print(f"[sabdab-cdr] downloading: {url}")
    download(url, Path(args.out))
    print(f"[sabdab-cdr] saved: {args.out}")


if __name__ == "__main__":
    main()
