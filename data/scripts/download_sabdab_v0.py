from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


SABDAB_SUMMARY_URLS = [
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/?all=true",
    "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))


def download(url: str, out: Path, timeout: int = 60) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
        from tqdm import tqdm
    except Exception as exc:
        print(f"[sabdab] requests/tqdm unavailable, using urllib fallback: {exc}")
        from urllib.request import urlopen

        with urlopen(url, timeout=timeout) as resp, out.open("wb") as f:
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = 100.0 * done / total
                    print(f"\r[sabdab] {done / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r[sabdab] {done / 1024 / 1024:.1f} MB", end="")
            print()
        return

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        with out.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=out.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def looks_like_summary(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return False
    fields = {x.strip().lower() for x in first.split("\t")}
    return "pdb" in fields and ("hchain" in fields or "h_chain" in fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the SAbDab summary TSV used to build ASTevolve antibody_kb."
    )
    parser.add_argument(
        "--out",
        default=str(DATA_ROOT / "sabdab_raw" / "sabdab_summary.tsv"),
        help="Output TSV path.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Optional explicit SAbDab summary URL. Defaults try current and legacy endpoints.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    urls = [args.url] if args.url else SABDAB_SUMMARY_URLS
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            print(f"[sabdab] downloading: {url}")
            download(url, out)
            if looks_like_summary(out):
                print(f"[sabdab] summary saved: {out}")
                return
            print(f"[sabdab] downloaded file did not look like a summary TSV: {out}")
        except Exception as exc:
            last_error = exc
            print(f"[sabdab] failed: {url}\n  {exc}")

    raise SystemExit(f"Could not download a valid SAbDab summary TSV. Last error: {last_error}")


if __name__ == "__main__":
    main()
