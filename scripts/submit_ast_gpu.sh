#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CASE="${CASE:-tetr_dopamine}"
STAGE="${STAGE:-outer}"
PROFILE="${PROFILE:-formal}"
RUN_NAME="${RUN_NAME:-${CASE}_${STAGE}_$(date +%Y%m%d_%H%M%S)}"

PARTITION="${PARTITION:-sciverse_agent}"
QUOTATYPE="${QUOTATYPE:-reserved}"
NODES="${NODES:-1}"
NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-80G}"
TIME="${TIME:-24:00:00}"
CONDA_ENV="${CONDA_ENV:-ast}"
CONDA_SH="${CONDA_SH:-/mnt/petrelfs/zhaoyiyang/anaconda3/etc/profile.d/conda.sh}"
SUBMIT_MODE="${SUBMIT_MODE:-sbatch}"

export ASTEVOLVE_ARTIFACT_ROOT="${ASTEVOLVE_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}"
LOG_DIR="${LOG_DIR:-$ASTEVOLVE_ARTIFACT_ROOT/runs/logs}"
mkdir -p "$LOG_DIR"

if [[ ! -f "cases/$CASE/case.json" ]]; then
  echo "Invalid CASE=$CASE (missing cases/$CASE/case.json)" >&2
  find cases -maxdepth 2 -name case.json -printf '  %h\n' | sed 's#cases/##' >&2 || true
  exit 2
fi

if [[ "$SUBMIT_MODE" == "sbatch" && -n "$(command -v sbatch || true)" ]]; then
  BATCH_FILE="$LOG_DIR/${RUN_NAME}.sbatch.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=${PARTITION}"
    [[ -n "$QUOTATYPE" ]] && echo "#SBATCH --quotatype=${QUOTATYPE}"
    echo "#SBATCH --nodes=${NODES}"
    echo "#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}"
    echo "#SBATCH --gres=gpu:${GPUS}"
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    echo "#SBATCH --mem=${MEM}"
    echo "#SBATCH --time=${TIME}"
    echo "#SBATCH --job-name=${RUN_NAME}"
    echo "#SBATCH --output=${LOG_DIR}/${RUN_NAME}.log"
    echo "#SBATCH --error=${LOG_DIR}/${RUN_NAME}.err"
    echo "set -euo pipefail"
    echo "cd '$PROJECT_ROOT'"
    echo "export CASE='$CASE'"
    echo "export STAGE='$STAGE'"
    echo "export PROFILE='$PROFILE'"
    echo "export RUN_NAME='$RUN_NAME'"
    echo "export OUTER_ITERATIONS='${OUTER_ITERATIONS:-}'"
    echo "export INNER_ITERATIONS='${INNER_ITERATIONS:-}'"
    echo "export USE_PROTENIX='${USE_PROTENIX:-on}'"
    echo "export EXTERNAL_KB='${EXTERNAL_KB:-on}'"
    echo "export EXTERNAL_RETRIEVAL='${EXTERNAL_RETRIEVAL:-auto}'"
    echo "export PROGEN_WEIGHT='${PROGEN_WEIGHT:-}'"
    echo "export CONDA_ENV='$CONDA_ENV'"
    echo "export CONDA_SH='$CONDA_SH'"
    echo "export ASTEVOLVE_ARTIFACT_ROOT='$ASTEVOLVE_ARTIFACT_ROOT'"
    echo "bash scripts/run_ast_case.sh"
  } > "$BATCH_FILE"
  chmod +x "$BATCH_FILE"
  sbatch "$BATCH_FILE"
else
  echo "Submitting with srun --async. For terminal-independent jobs, prefer SUBMIT_MODE=sbatch when sbatch is available." >&2
  srun -p "$PARTITION" --quotatype="$QUOTATYPE" \
    --nodes="$NODES" \
    --ntasks-per-node="$NTASKS_PER_NODE" \
    --gres="gpu:${GPUS}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name "$RUN_NAME" \
    -o "$LOG_DIR/${RUN_NAME}.log" \
    --async \
    bash -lc "cd '$PROJECT_ROOT' && export CASE='$CASE' STAGE='$STAGE' PROFILE='$PROFILE' RUN_NAME='$RUN_NAME' OUTER_ITERATIONS='${OUTER_ITERATIONS:-}' INNER_ITERATIONS='${INNER_ITERATIONS:-}' USE_PROTENIX='${USE_PROTENIX:-on}' EXTERNAL_KB='${EXTERNAL_KB:-on}' EXTERNAL_RETRIEVAL='${EXTERNAL_RETRIEVAL:-auto}' PROGEN_WEIGHT='${PROGEN_WEIGHT:-}' CONDA_ENV='$CONDA_ENV' CONDA_SH='$CONDA_SH' ASTEVOLVE_ARTIFACT_ROOT='$ASTEVOLVE_ARTIFACT_ROOT' && bash scripts/run_ast_case.sh"
fi

echo "log: $LOG_DIR/${RUN_NAME}.log"
