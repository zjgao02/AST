# ESMFold2 Interface

The ESMFold2 adapter is wired but does not download weights by default.

Entry points:

```python
from astevolve.apis.esmfold2 import (
    describe_esmfold2_setup,
    run_esmfold2_confidence_multichain,
    run_esmfold2_plddt_multichain,
)
```

ASTevolve model registry:

```text
ASTEVOLVE_STRUCTURE_MODEL=esmfold2
```

Local Hugging Face mode:

```text
ASTEVOLVE_ESMFOLD2_MODE=local
ASTEVOLVE_ESMFOLD2_MODEL=biohub/ESMFold2
```

By default, local mode uses `local_files_only=True`. If the weights are not
already cached, it raises `ESMFold2WeightsRequired` and does not download.

Only set this when you explicitly want to download model weights:

```text
ASTEVOLVE_ESMFOLD2_ALLOW_DOWNLOAD=1
```

Biohub Platform mode:

```text
ASTEVOLVE_ESMFOLD2_MODE=biohub
ASTEVOLVE_ESMFOLD2_MODEL=esmfold2-fast-2026-05
ASTEVOLVE_ESMFOLD2_TOKEN=<Biohub token>
```

Current defaults for structure sampling:

```text
ASTEVOLVE_ESMFOLD2_NUM_LOOPS=3
ASTEVOLVE_ESMFOLD2_NUM_SAMPLING_STEPS=32
ASTEVOLVE_ESMFOLD2_NUM_DIFFUSION_SAMPLES=1
```

The adapter returns the same shape expected by the current Protenix integration:

```python
{
    "provider": "esmfold2",
    "metrics": {"plddt": ..., "ptm": ..., "iptm": ...},
    "chain_metrics": {"plddt": {"BB": ..., "T": ...}},
    "residue_plddt": {"BB": [...], "T": [...]},
    "cif_path": "...",
}
```

The local/API dependencies from the upstream README are not installed or run by
this interface file. Install them only when you are ready to use ESMFold2.

## Using A Separate Conda Environment

Biohub's ESMFold2 package currently requires Python 3.12 or newer. The main
ASTevolve process can still run in the existing `pytorch` environment, while
ESMFold2 runs as a structure backend in its own conda environment:

```text
ASTEVOLVE_STRUCTURE_MODEL=esmfold2
ASTEVOLVE_STRUCTURE_MODEL_NAME=/srv/astevolve/models/biohub/ESMFold2-Fast
ASTEVOLVE_ESMFOLD2_MODE=local
ASTEVOLVE_ESMFOLD2_CONDA_ENV=esmfold2
```

When `ASTEVOLVE_ESMFOLD2_CONDA_ENV` is set, `astevolve/apis/esmfold2.py` launches a small
conda worker and returns the same confidence dictionary as the in-process
adapter. This makes Protenix and ESMFold2 interchangeable from the inner loop's
point of view.
