from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from astevolve.cases import resolve_case


@dataclass
class OpenEvolveRun:
    """Thin description of an OpenEvolve-driven outer-loop run."""

    project_root: Path
    case_id: Optional[str] = None
    initial_program: Optional[str] = None
    evaluator: str = "evaluator.py"
    config: Optional[str] = None
    output_dir: Optional[str] = None

    def command(self, iterations: Optional[int] = None) -> List[str]:
        case = resolve_case(self.case_id)
        case_dir = Path("cases") / case.case_id
        initial_program = self.initial_program or str(case_dir / str(case.metadata.get("entry_program", "initial_program.py")))
        config = self.config or str(case_dir / str(case.metadata.get("config_path", "config.yaml")))

        cmd = [
            "python",
            "openevolve/openevolve-run.py",
            initial_program,
            self.evaluator,
            "--config",
            config,
        ]
        if iterations is not None:
            cmd.extend(["--iterations", str(iterations)])
        if self.output_dir:
            cmd.extend(["--output", self.output_dir])
        return cmd