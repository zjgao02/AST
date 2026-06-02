from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astevolve.cases import resolve_case
from astevolve.evaluation.open_evolve import evaluate
from astevolve.runtime.paths import artifact_path


def _bool_env(value: bool) -> str:
    return "1" if value else "0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal ASTevolve case smoke evaluation.")
    parser.add_argument("--case", required=True, help="Case id, for example cd25_scfv or tetr_dopamine.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--protenix", action="store_true", help="Enable Protenix structure evaluation.")
    parser.add_argument("--external-kb", action="store_true", help="Enable external KB priors/retrieval.")
    parser.add_argument("--external-retrieval", action="store_true", help="Enable embedding retrieval from the external KB.")
    parser.add_argument("--no-external-retrieval", action="store_true", help="Disable retrieval even when external KB priors are enabled.")
    parser.add_argument("--progen", action="store_true", help="Enable ProGen sequence prior.")
    parser.add_argument("--progen-weight", type=float, default=None, help="Override ASTEVOLVE_PROGEN_WEIGHT for this smoke run.")
    parser.add_argument("--output-name", default="smoke")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.no_external_retrieval:
        external_retrieval = False
    elif args.external_retrieval:
        external_retrieval = True
    else:
        external_retrieval = args.external_kb

    progen_weight = args.progen_weight
    if progen_weight is None:
        progen_weight = 1.0 if args.progen else 0.0

    os.environ["ASTEVOLVE_CASE_ID"] = args.case
    os.environ["ASTEVOLVE_INNER_ITERATIONS"] = str(args.iterations)
    os.environ["ASTEVOLVE_ENABLE_PROTENIX"] = _bool_env(args.protenix)
    os.environ["ASTEVOLVE_ENABLE_EXTERNAL_KB"] = _bool_env(args.external_kb)
    os.environ["ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL"] = _bool_env(external_retrieval)
    os.environ["ASTEVOLVE_PROGEN_WEIGHT"] = str(max(0.0, float(progen_weight)))
    os.environ.setdefault("ASTEVOLVE_PROTENIX_NUM_WORKERS", "1")
    os.environ.setdefault("ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA", "0")
    os.environ.setdefault("ASTEVOLVE_PROTENIX_COMPLEX_CYCLE", "1")
    os.environ.setdefault("ASTEVOLVE_PROTENIX_COMPLEX_STEP", "1")
    os.environ.setdefault("ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE", "1")
    os.environ.setdefault("ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS", "0")

    case = resolve_case(args.case)
    run_root = artifact_path("smoke", args.case, args.output_name)
    os.environ.setdefault("ASTEVOLVE_MCTS_OUTPUT_DIR", str(run_root / "inner"))
    os.environ.setdefault("ASTEVOLVE_PROTENIX_TMP", str(run_root / "protenix_tmp"))

    program = case.root / "cases" / case.case_id / str(case.metadata.get("entry_program", "initial_program.py"))
    result = evaluate(str(program))
    payload = {
        "case": args.case,
        "program": str(program),
        "metrics": getattr(result, "metrics", {}),
        "artifacts": getattr(result, "artifacts", {}),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2, default=str))
        artifacts = payload["artifacts"] or {}
        search_artifacts = artifacts.get("search_artifacts") or {}
        if search_artifacts:
            print(json.dumps(search_artifacts, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["metrics"].get("combined_score", 0.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
