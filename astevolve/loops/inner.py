from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from astevolve.cases import resolve_case


@dataclass
class InnerLoopRunner:
    """Task-oriented inner-loop runner for one resolved case."""

    case_id: Optional[str] = None

    def run(self, strategy: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        from engine.case_builder import run_design_search

        case = resolve_case(self.case_id)
        return run_design_search(
            strategy,
            seed=seed,
            design_state_path=str(case.design_state_path),
            memory_path=str(case.memory_path),
        )