from __future__ import annotations
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import tempfile
import subprocess
import json

_CACHE: Dict[str, float] = {}


def run_protenix_plddt_multichain(
    pred_name,
    chains: List[Tuple[str, str]],
    metric: str = "plddt",                          # ← 新增
    device: Optional[str] = None,
    seed: int = 101,
    model_name: str = "protenix_mini_esm_v0.5.0",
    conda_env: str = "protenix_mini", 
) -> float:
    if not chains:
        return 0.0

    key = "|".join(f"{cid}:{seq}" for cid, seq in chains)
    cache_key = f"{key}|{seed}|{model_name}|{metric}"   # ← 加 metric
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    td_path = Path('/home/guchunbin/guchunbin/ast/openevolve/examples/protein_evolve_v3/tmp/')                        
    input_json_path = td_path / "input.json"
    out_dir = td_path / f"output_{pred_name}"
    out_dir.mkdir(exist_ok=True)

    sequences = []
    for cid, seq in chains:
        sequences.append({
            "proteinChain": {
                "sequence": seq,
                "count":1
            }
        })

    input_data = [{
        "sequences": sequences,
        "name": pred_name,
    }]
    
    input_json_path.write_text(json.dumps(input_data, indent=2))

    cmd = [
        "conda", "run", "-n", conda_env,
        "protenix", "predict",
        "--input", str(input_json_path),
        "--out_dir", str(out_dir),
        "--model_name", model_name,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("[protenix] prediction timed out (600s)")
        return 0.0
    except FileNotFoundError:
        print("[protenix] conda or protenix not found in PATH")
        return 0.0

    if result.returncode != 0:
        print(f"[protenix] failed (rc={result.returncode})")
        if result.stderr:
            print(f"[protenix] stderr: {result.stderr[:1000]}")
        return 0.0

    confidence_json = (
        out_dir
        / pred_name
        / f"seed_{seed}"
        / "predictions"
        / f"{pred_name}_seed_{seed}_summary_confidence_sample_0.json"
    )

    if not confidence_json.exists():
        possible = sorted(out_dir.rglob("*summary_confidence*.json"))
        if possible:
            confidence_json = possible[0]
            print(f"[protenix] fallback confidence file: {confidence_json}")
        else:
            print(f"[protenix] output not found, expected: {confidence_json}")
            all_files = list(out_dir.rglob("*"))
            print(f"[protenix] files under output: {all_files[:30]}")
            return 0.0

    try:
        with open(confidence_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[protenix] failed to parse {confidence_json}: {exc}")
        return 0.0

    result_value = float(data.get(metric, 0.0))             # ← 改：读 metric 字段

    _CACHE[cache_key] = result_value
    return result_value