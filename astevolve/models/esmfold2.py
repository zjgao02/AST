from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ESMFold2StructureModel:
    """ESMFold2 structure-model adapter.

    The underlying API is lazy: importing this class does not download weights
    or call the Biohub Platform. Local inference only downloads weights when
    ASTEVOLVE_ESMFOLD2_ALLOW_DOWNLOAD=1 is set.
    """

    name = "esmfold2"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from astevolve.apis.esmfold2 import run_esmfold2_confidence_multichain

        return run_esmfold2_confidence_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def scalar_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> float:
        from astevolve.apis.esmfold2 import run_esmfold2_plddt_multichain

        return run_esmfold2_plddt_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )
