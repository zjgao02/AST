#!/usr/bin/env bash
set -euo pipefail

CASE="tetr_dopamine"
STAGE="preview"
PROFILE="smoke"
OUTER_ITERATIONS=""
INNER_ITERATIONS=""
PROTENIX="auto"
EXTERNAL_KB="auto"
EXTERNAL_RETRIEVAL="auto"
PROGEN_WEIGHT=""
RUN_NAME=""
CONDA_ENV="pytorch"
NO_CONDA=0
DRY_RUN=0
SKIP_LLM_KEY_CHECK=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_ast_run.sh --case tetr_dopamine --profile formal --stage formal

Options:
  --case CASE_ID (any cases/CASE_ID/case.json)
  --stage assets|preview|inner-smoke|outer|formal|all
  --profile smoke|cheap|formal
  --outer-iterations N
  --inner-iterations N
  --use-protenix auto|on|off
  --external-kb auto|on|off
  --external-retrieval auto|on|off
  --progen-weight FLOAT
  --run-name NAME
  --conda-env NAME
  --no-conda
  --dry-run
  --skip-llm-key-check
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --outer-iterations) OUTER_ITERATIONS="$2"; shift 2 ;;
    --inner-iterations) INNER_ITERATIONS="$2"; shift 2 ;;
    --use-protenix|--protenix) PROTENIX="$2"; shift 2 ;;
    --external-kb) EXTERNAL_KB="$2"; shift 2 ;;
    --external-retrieval) EXTERNAL_RETRIEVAL="$2"; shift 2 ;;
    --progen-weight) PROGEN_WEIGHT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --no-conda) NO_CONDA=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-llm-key-check) SKIP_LLM_KEY_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$STAGE" in
  assets|preview|inner-smoke|outer|formal|all) ;;
  *) echo "Invalid --stage: $STAGE" >&2; exit 2 ;;
esac
case "$PROFILE" in
  smoke|cheap|formal) ;;
  *) echo "Invalid --profile: $PROFILE" >&2; exit 2 ;;
esac

resolve_toggle() {
  local value="$1"
  local default_value="$2"
  case "$value" in
    on|1|true|yes) echo 1 ;;
    off|0|false|no) echo 0 ;;
    auto) echo "$default_value" ;;
    *) echo "Invalid toggle value: $value" >&2; exit 2 ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "cases/$CASE/case.json" ]]; then
  echo "Invalid --case: $CASE (missing cases/$CASE/case.json)" >&2
  echo "Available cases:" >&2
  find cases -maxdepth 2 -name case.json -printf '  %h\n' | sed 's#cases/##' >&2 || true
  exit 2
fi

case "$CASE" in
  tetr_dopamine)
    CASE_INNER_DEFAULT=240
    CASE_PROGEN_DEFAULT="0.5"
    CASE_RETRIEVAL_DEFAULT=0
    ;;
  cd25_scfv|cd25_scfv_selectivity|pdl1_scfv_selectivity|proteor1_cdr_mask)
    CASE_INNER_DEFAULT=1200
    CASE_PROGEN_DEFAULT="1.0"
    CASE_RETRIEVAL_DEFAULT=1
    ;;
  pdz_peptide_selectivity|calcium_efhand_switch)
    CASE_INNER_DEFAULT=360
    CASE_PROGEN_DEFAULT="0.6"
    CASE_RETRIEVAL_DEFAULT=0
    ;;
  *)
    CASE_INNER_DEFAULT=360
    CASE_PROGEN_DEFAULT="0.6"
    CASE_RETRIEVAL_DEFAULT=0
    ;;
esac

if [[ "$PROFILE" == "smoke" ]]; then
  PROFILE_OUTER=1
  PROFILE_INNER=1
  PROFILE_PROTENIX=0
  PROFILE_EXTERNAL_KB=0
  PROFILE_EXTERNAL_RETRIEVAL=0
  PROFILE_PROGEN="0.0"
elif [[ "$PROFILE" == "cheap" ]]; then
  PROFILE_OUTER=10
  PROFILE_INNER=10
  PROFILE_PROTENIX=0
  PROFILE_EXTERNAL_KB=1
  PROFILE_EXTERNAL_RETRIEVAL="$CASE_RETRIEVAL_DEFAULT"
  PROFILE_PROGEN="$CASE_PROGEN_DEFAULT"
else
  PROFILE_OUTER=200
  PROFILE_INNER="$CASE_INNER_DEFAULT"
  PROFILE_PROTENIX=1
  PROFILE_EXTERNAL_KB=1
  PROFILE_EXTERNAL_RETRIEVAL="$CASE_RETRIEVAL_DEFAULT"
  PROFILE_PROGEN="$CASE_PROGEN_DEFAULT"
fi

OUTER_ITERATIONS="${OUTER_ITERATIONS:-$PROFILE_OUTER}"
INNER_ITERATIONS="${INNER_ITERATIONS:-$PROFILE_INNER}"
PROGEN_WEIGHT="${PROGEN_WEIGHT:-$PROFILE_PROGEN}"
USE_PROTENIX="$(resolve_toggle "$PROTENIX" "$PROFILE_PROTENIX")"
USE_EXTERNAL_KB="$(resolve_toggle "$EXTERNAL_KB" "$PROFILE_EXTERNAL_KB")"
USE_EXTERNAL_RETRIEVAL="$(resolve_toggle "$EXTERNAL_RETRIEVAL" "$PROFILE_EXTERNAL_RETRIEVAL")"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="${CASE}_${PROFILE}_$(date +%Y%m%d_%H%M%S)"
fi

export ASTEVOLVE_PROJECT_ROOT="${ASTEVOLVE_PROJECT_ROOT:-$PROJECT_ROOT}"
export ASTEVOLVE_DATA_ROOT="${ASTEVOLVE_DATA_ROOT:-$PROJECT_ROOT/data}"
export ASTEVOLVE_ARTIFACT_ROOT="${ASTEVOLVE_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
export ASTEVOLVE_TMP_ROOT="${ASTEVOLVE_TMP_ROOT:-$ASTEVOLVE_ARTIFACT_ROOT/tmp}"
RUN_ROOT="$ASTEVOLVE_ARTIFACT_ROOT/runs/$CASE/$RUN_NAME"

export ASTEVOLVE_RUN_ROOT="$RUN_ROOT"
export ASTEVOLVE_CASE_OUTPUT_ROOT="${ASTEVOLVE_CASE_OUTPUT_ROOT:-$RUN_ROOT/case}"
export ASTEVOLVE_TRANSIENT_ARTIFACT_DIR="${ASTEVOLVE_TRANSIENT_ARTIFACT_DIR:-transient}"
export ASTEVOLVE_CASE_ID="$CASE"
export ASTEVOLVE_INNER_ITERATIONS="$INNER_ITERATIONS"
export ASTEVOLVE_ENABLE_PROTENIX="$USE_PROTENIX"
export ASTEVOLVE_ENABLE_EXTERNAL_KB="$USE_EXTERNAL_KB"
export ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL="$USE_EXTERNAL_RETRIEVAL"
export ASTEVOLVE_PROGEN_WEIGHT="$PROGEN_WEIGHT"
export ASTEVOLVE_MCTS_OUTPUT_DIR="$RUN_ROOT/inner"
export ASTEVOLVE_PROTENIX_TMP="$RUN_ROOT/protenix_tmp"
export ASTEVOLVE_PROTENIX_NUM_WORKERS="${ASTEVOLVE_PROTENIX_NUM_WORKERS:-1}"
export ASTEVOLVE_PROTENIX_CONDA_ENV="${ASTEVOLVE_PROTENIX_CONDA_ENV:-${CONDA_DEFAULT_ENV:-$CONDA_ENV}}"
export ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA="${ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA:-0}"
export ASTEVOLVE_PROTENIX_COMPLEX_CYCLE="${ASTEVOLVE_PROTENIX_COMPLEX_CYCLE:-1}"
export ASTEVOLVE_PROTENIX_COMPLEX_STEP="${ASTEVOLVE_PROTENIX_COMPLEX_STEP:-1}"
export ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE="${ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE:-1}"
export ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS="${ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS:-0}"

echo "case=$CASE stage=$STAGE profile=$PROFILE run=$RUN_NAME"
echo "outer_iterations=$OUTER_ITERATIONS inner_iterations=$INNER_ITERATIONS protenix=$USE_PROTENIX external_kb=$USE_EXTERNAL_KB retrieval=$USE_EXTERNAL_RETRIEVAL progen_weight=$PROGEN_WEIGHT"
echo "run_root=$RUN_ROOT"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" == 1 ]]; then
    return 0
  fi
  "$@"
}

if [[ "$NO_CONDA" == 1 ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
fi

run_python() {
  run_cmd "${PYTHON_CMD[@]}" "$@"
}

if [[ "$STAGE" == "formal" || "$STAGE" == "all" ]]; then
  STEPS=(assets preview inner-smoke outer)
else
  STEPS=("$STAGE")
fi

for step in "${STEPS[@]}"; do
  case "$step" in
    assets)
      run_python scripts/check_assets.py --case "$CASE"
      ;;
    preview)
      run_python "cases/$CASE/initial_program.py"
      ;;
    inner-smoke)
      smoke_args=(scripts/smoke_case.py --case "$CASE" --iterations "$INNER_ITERATIONS" --output-name "$RUN_NAME" --progen-weight "$PROGEN_WEIGHT" --json)
      if [[ "$USE_PROTENIX" == 1 ]]; then
        smoke_args+=(--protenix)
      fi
      if [[ "$USE_EXTERNAL_KB" == 1 ]]; then
        smoke_args+=(--external-kb)
      fi
      if [[ "$USE_EXTERNAL_RETRIEVAL" == 1 ]]; then
        smoke_args+=(--external-retrieval)
      else
        smoke_args+=(--no-external-retrieval)
      fi
      run_python "${smoke_args[@]}"
      ;;
    outer)
      if [[ "$SKIP_LLM_KEY_CHECK" != 1 && -z "${ASTEVOLVE_LLM_API_KEY:-}" ]]; then
        echo "ASTEVOLVE_LLM_API_KEY is required for outer-loop OpenEvolve runs." >&2
        exit 2
      fi
      run_python openevolve/openevolve-run.py "cases/$CASE/initial_program.py" evaluator.py --config "cases/$CASE/config.yaml" --iterations "$OUTER_ITERATIONS" --output "$RUN_ROOT/openevolve"
      ;;
  esac
done
