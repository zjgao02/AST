from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ProtenixStructureModel:
    name = "protenix"

    def confidence_multichain(
        self,
        pred_name: Optional[str] = None,
        chains: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from astevolve.apis.protenix import run_protenix_confidence_multichain

        return run_protenix_confidence_multichain(
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
        from astevolve.apis.protenix import run_protenix_plddt_multichain

        return run_protenix_plddt_multichain(
            pred_name=pred_name,
            chains=chains,
            **kwargs,
        )

    def confidence_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from astevolve.apis.protenix import run_protenix_confidence_complex

        return run_protenix_confidence_complex(
            pred_name=pred_name,
            entities=entities,
            **kwargs,
        )

    def scalar_complex(
        self,
        pred_name: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> float:
        from astevolve.apis.protenix import run_protenix_plddt_complex

        return run_protenix_plddt_complex(
            pred_name=pred_name,
            entities=entities,
            **kwargs,
        )
