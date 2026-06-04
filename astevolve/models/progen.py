from __future__ import annotations

from typing import Any, Dict


class ProGenSequencePrior:
    name = "progen"

    def sequence_loglikelihood(self, seq: str, **kwargs: Any) -> Dict[str, float]:
        from astevolve.apis.progen import sequence_loglikelihood

        return sequence_loglikelihood(seq, **kwargs)

