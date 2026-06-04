from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def _normalise_name(name: Optional[str], default: str) -> str:
    value = str(name or default).strip().lower()
    aliases = {
        "progen2": "progen",
        "progen2-small": "progen",
        "protenix_mini": "protenix",
        "protenix-mini": "protenix",
        "esmfold": "esmfold2",
        "esmfold_v2": "esmfold2",
    }
    return aliases.get(value, value)


def available_model_interfaces() -> Dict[str, List[str]]:
    return {
        "sequence_prior": ["progen"],
        "structure": ["protenix", "chai1", "esmfold2"],
    }


def _sequence_model(name: Optional[str] = None):
    provider = _normalise_name(
        name or os.environ.get("ASTEVOLVE_SEQUENCE_PRIOR_MODEL"),
        "progen",
    )
    if provider == "progen":
        from .progen import ProGenSequencePrior

        return ProGenSequencePrior()
    raise ValueError(f"Unknown sequence prior model: {provider}")


def _structure_model(name: Optional[str] = None):
    provider = _normalise_name(
        name or os.environ.get("ASTEVOLVE_STRUCTURE_MODEL"),
        "protenix",
    )
    if provider == "protenix":
        from .protenix import ProtenixStructureModel

        return ProtenixStructureModel()
    if provider == "chai1":
        from .chai1 import Chai1StructureModel

        return Chai1StructureModel()
    if provider == "esmfold2":
        from .esmfold2 import ESMFold2StructureModel

        return ESMFold2StructureModel()
    raise ValueError(f"Unknown structure model: {provider}")


def sequence_loglikelihood(
    seq: str,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    return _sequence_model(provider).sequence_loglikelihood(seq, **kwargs)


def run_structure_confidence_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return _structure_model(provider).confidence_multichain(
        pred_name=pred_name,
        chains=chains,
        **kwargs,
    )


def run_structure_plddt_multichain(
    pred_name: Optional[str] = None,
    chains: Optional[List[Tuple[str, str]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> float:
    return _structure_model(provider).scalar_multichain(
        pred_name=pred_name,
        chains=chains,
        **kwargs,
    )


def run_structure_confidence_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    model = _structure_model(provider)
    if not hasattr(model, "confidence_complex"):
        raise NotImplementedError(f"{model.name} does not support complex entities")
    return model.confidence_complex(
        pred_name=pred_name,
        entities=entities,
        **kwargs,
    )


def run_structure_plddt_complex(
    pred_name: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> float:
    model = _structure_model(provider)
    if not hasattr(model, "scalar_complex"):
        raise NotImplementedError(f"{model.name} does not support complex entities")
    return model.scalar_complex(
        pred_name=pred_name,
        entities=entities,
        **kwargs,
    )
