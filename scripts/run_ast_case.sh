#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_SH="${CONDA_SH:-/mnt/petrelfs/zhaoyiyang/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-auto}"

resolve_conda_env() {
  local requested="${1:-auto}"
  if [[ -n "$requested" && "$requested" != "auto" ]]; then
    echo "$requested"
    return 0
  fi
  if [[ -n "${ASTEVOLVE_CONDA_ENV:-}" && "${ASTEVOLVE_CONDA_ENV}" != "auto" ]]; then
    echo "$ASTEVOLVE_CONDA_ENV"
    return 0
  fi
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local envs
    envs="$(conda env list 2>/dev/null | awk 'NF && $1 !~ /^#/ { if ($1 == "*") print $2; else print $1 }' || true)"
    if grep -qx "ast" <<<"$envs"; then echo "ast"; return 0; fi
    if grep -qx "pytorch" <<<"$envs"; then echo "pytorch"; return 0; fi
  fi
  echo "ast"
}

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  CONDA_ENV_RESOLVED="$(resolve_conda_env "$CONDA_ENV")"
  conda activate "$CONDA_ENV_RESOLVED"
elif command -v conda >/dev/null 2>&1; then
  CONDA_ENV_RESOLVED="$(resolve_conda_env "$CONDA_ENV")"
  conda activate "$CONDA_ENV_RESOLVED"
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
export ASTEVOLVE_CONDA_ENV="${ASTEVOLVE_CONDA_ENV:-${CONDA_ENV_RESOLVED:-$CONDA_ENV}}"
export ASTEVOLVE_PROTENIX_CONDA_ENV="${ASTEVOLVE_PROTENIX_CONDA_ENV:-$ASTEVOLVE_CONDA_ENV}"
export HF_HOME="${HF_HOME:-$ASTEVOLVE_MODEL_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

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

if [[ -n "${ASTEVOLVE_PROTENIX_MODEL_NAME:-}" ]]; then
  echo "protenix_model_name=$ASTEVOLVE_PROTENIX_MODEL_NAME"
else
  echo "protenix_model_name=case_default"
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

args=(
  scripts/submit_ast_run.sh
  --case "$CASE"
  --stage "$STAGE"
  --profile "$PROFILE"
  --use-protenix "$USE_PROTENIX"
  --external-kb "$EXTERNAL_KB"
  --external-retrieval "$EXTERNAL_RETRIEVAL"
  --run-name "$RUN_NAME"
  --conda-env "${CONDA_ENV_RESOLVED:-$CONDA_ENV}"
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
