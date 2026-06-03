from __future__ import annotations

from astevolve.evaluation.open_evolve import evaluate


def _default_program(case_id: str | None = None) -> str:
    from astevolve.cases import resolve_case

    case = resolve_case(case_id)
    entry = str(case.metadata.get("entry_program", "initial_program.py"))
    return str(case.root / "cases" / case.case_id / entry)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an ASTevolve program through the OpenEvolve shim.")
    parser.add_argument("program", nargs="?", help="Path to a case initial_program.py.")
    parser.add_argument("--case", default=None, help="Case id to evaluate when no program path is supplied.")
    args = parser.parse_args()

    print(evaluate(args.program or _default_program(args.case)))


if __name__ == "__main__":
    main()
