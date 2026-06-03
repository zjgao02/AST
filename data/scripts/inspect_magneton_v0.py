from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ASTEVOLVE_DATA_ROOT", PROJECT_ROOT / "data"))
ROOT = DATA_ROOT / "magneton_raw"
JSONL = ROOT / "interpro_103.0" / "debug_subset" / "swissprot.with_ss.0.jsonl.gz"
SEQ_TSV = ROOT / "sequences" / "swissprot_subset.tsv"


def main() -> None:
    if not JSONL.exists():
        raise FileNotFoundError(JSONL)
    if not SEQ_TSV.exists():
        raise FileNotFoundError(SEQ_TSV)

    with gzip.open(JSONL, "rt", encoding="utf-8") as handle:
        obj = json.loads(next(handle))

    print("[jsonl]", JSONL)
    print("\n[top-level keys]")
    for key, value in obj.items():
        print(f"- {key}: {type(value).__name__}")

    print("\n[first protein preview]")
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:8000])

    print("\n[sequence TSV]", SEQ_TSV)
    with SEQ_TSV.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for i, row in zip(range(5), reader):
            print(f"line {i + 1}: {row[:8]}")


if __name__ == "__main__":
    main()
