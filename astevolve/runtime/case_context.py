from __future__ import annotations

from typing import Dict, Optional

from astevolve.cases.base import DesignCase
from astevolve.cases.registry import resolve_case


def get_current_case(case_id: Optional[str] = None) -> DesignCase:
    return resolve_case(case_id)


def current_case_kwargs(case_id: Optional[str] = None) -> Dict[str, str]:
    return get_current_case(case_id).as_kwargs()
