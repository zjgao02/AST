#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${ASTEVOLVE_ARTIFACT_ROOT:-$ROOT/artifacts}/logs"
mkdir -p "$LOG_DIR"

MAX_PROTEINS_PER_FAMILY="${MAX_PROTEINS_PER_FAMILY:-200}"
PAGE_SIZE="${PAGE_SIZE:-100}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "[$(date -Is)] Building representative ATF KB"
python data/scripts/build_atf_interpro_kb_v0.py \
  --max-proteins-per-family "$MAX_PROTEINS_PER_FAMILY" \
  --page-size "$PAGE_SIZE" \
  --sleep "$SLEEP_SECONDS" \
  --include-full \
  2>&1 | tee "$LOG_DIR/atf_kb_representative_build.log"

echo "[$(date -Is)] Embedding representative ATF KB"
python data/scripts/embed_atf_interpro_records_v0.py \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$LOG_DIR/atf_kb_representative_embed.log"

python scripts/check_assets.py --case tetr_dopamine
