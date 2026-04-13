from __future__ import annotations
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import tempfile
import numpy as np

from chai_lab.chai1 import run_inference

_CACHE: Dict[Tuple[str, str], float] = {}


def run_chai1_plddt_multichain(
    chains: List[Tuple[str, str]],
    device: Optional[str] = None,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 50,
    seed: int = 0,
    use_esm_embeddings: bool = True,
) -> float:
    if not chains:
        return 0.0

    dev = device or ("cuda" if _has_cuda() else "cpu")
    key = "|".join(f"{cid}:{seq}" for cid, seq in chains)
    cache_key = (key, dev)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        fasta_path = td_path / "input.fasta"
        out_dir = td_path / "out"
        out_dir.mkdir(exist_ok=True)

        lines = []
        for cid, seq in chains:
            lines.append(f">protein|name={cid}")
            lines.append(seq)

        fasta_path.write_text("\n".join(lines) + "\n")

        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=out_dir,
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            seed=seed,
            device=dev,
            use_esm_embeddings=use_esm_embeddings,
        )

        plddt_scores = candidates.ranking_data[0].plddt_scores
        mean_plddt = float(plddt_scores.complex_plddt.numpy())

    _CACHE[cache_key] = mean_plddt
    return mean_plddt


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False