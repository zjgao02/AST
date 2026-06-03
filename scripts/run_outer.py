from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astevolve.loops.outer import OpenEvolveRun
from astevolve.runtime.paths import artifact_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch an OpenEvolve outer-loop run for an ASTevolve case.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output", default=None, help="OpenEvolve output directory.")
    parser.add_argument("--conda-env", default=None, help="Optional conda env name to run through conda run -n.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output or str(artifact_path("openevolve", args.case))
    runner = OpenEvolveRun(project_root=PROJECT_ROOT, case_id=args.case, output_dir=output)
    cmd = runner.command(iterations=args.iterations)
    if args.conda_env:
        cmd = ["conda", "run", "-n", args.conda_env, *cmd]

    print(" ".join(cmd))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("ASTEVOLVE_CASE_ID", args.case)
    env.setdefault("ASTEVOLVE_PROJECT_ROOT", str(PROJECT_ROOT))
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
