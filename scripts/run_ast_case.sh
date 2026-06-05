#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_SH="${CONDA_SH:-/mnt/petrelfs/zhaoyiyang/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-ast}"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV"
else
  echo "Could not find conda. Set CONDA_SH or run with NO_CONDA=1 from an activated environment." >&2
  exit 2
fi

if [[ -f "$PROJECT_ROOT/configs/env.linux.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/configs/env.linux.local"
  set +a
fi

export ASTEVOLVE_PROJECT_ROOT="${ASTEVOLVE_PROJECT_ROOT:-$PROJECT_ROOT}"
export ASTEVOLVE_DATA_ROOT="${ASTEVOLVE_DATA_ROOT:-$PROJECT_ROOT/data}"
export ASTEVOLVE_MODEL_ROOT="${ASTEVOLVE_MODEL_ROOT:-$PROJECT_ROOT/model_weights}"
export ASTEVOLVE_ARTIFACT_ROOT="${ASTEVOLVE_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
export ASTEVOLVE_TMP_ROOT="${ASTEVOLVE_TMP_ROOT:-$ASTEVOLVE_ARTIFACT_ROOT/tmp}"
export HF_HOME="${HF_HOME:-$ASTEVOLVE_MODEL_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

CASE="${CASE:-tetr_dopamine}"
STAGE="${STAGE:-outer}"
PROFILE="${PROFILE:-formal}"
RUN_NAME="${RUN_NAME:-${CASE}_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
OUTER_ITERATIONS="${OUTER_ITERATIONS:-}"
INNER_ITERATIONS="${INNER_ITERATIONS:-}"
USE_PROTENIX="${USE_PROTENIX:-on}"
EXTERNAL_KB="${EXTERNAL_KB:-on}"
EXTERNAL_RETRIEVAL="${EXTERNAL_RETRIEVAL:-auto}"
PROGEN_WEIGHT="${PROGEN_WEIGHT:-}"

mkdir -p "$ASTEVOLVE_ARTIFACT_ROOT/runs/logs" "$ASTEVOLVE_TMP_ROOT"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

args=(
  scripts/submit_ast_run.sh
  --case "$CASE"
  --stage "$STAGE"
  --profile "$PROFILE"
  --use-protenix "$USE_PROTENIX"
  --external-kb "$EXTERNAL_KB"
  --external-retrieval "$EXTERNAL_RETRIEVAL"
  --run-name "$RUN_NAME"
  --conda-env "$CONDA_ENV"
  --no-conda
)

if [[ -n "$OUTER_ITERATIONS" ]]; then
  args+=(--outer-iterations "$OUTER_ITERATIONS")
fi
if [[ -n "$INNER_ITERATIONS" ]]; then
  args+=(--inner-iterations "$INNER_ITERATIONS")
fi
if [[ -n "$PROGEN_WEIGHT" ]]; then
  args+=(--progen-weight "$PROGEN_WEIGHT")
fi

bash "${args[@]}"
