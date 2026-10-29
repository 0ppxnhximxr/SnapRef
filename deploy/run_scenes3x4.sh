#!/usr/bin/env bash
# scenes3x4 data run on the Lambda 2x H100 SXM box: tag catalogs -> character tags -> manifest -> 2 GPU workers.
#   PROJECT_ROOT=~/anima-ref-adapter-research bash deploy/run_scenes3x4.sh [prep|smoke|production]
# prep:       install deps, link the datagen nodes, build data/character_tags.jsonl and the manifest
# smoke:      prep + 2 batches per GPU
# production: prep (skipped when outputs exist) + full generation in the background
set -euo pipefail
MODE="${1:-prep}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export COMFYUI_ROOT="${COMFYUI_ROOT:-$HOME/ComfyUI}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/comfy-venv/bin/python}"
export DATA_ROOT="${DATA_ROOT:-$HOME/data}"
export PROJECT_ROOT
export DATAGEN_ARGS="--config $PROJECT_ROOT/configs/datagen_scenes.json ${DATAGEN_ARGS:-}"
RUN=scenes3x4
cd "$PROJECT_ROOT"
log() { echo "[scenes3x4 $(date +%H:%M:%S)] $*"; }

"$PYTHON_BIN" -m pip install -q polars 'huggingface_hub>=0.30' pillow
ln -sfn "$PROJECT_ROOT/comfyui_nodes/anima_ref_datagen_nodes" "$COMFYUI_ROOT/custom_nodes/anima_ref_datagen_nodes"
[[ -f data/danbooru_tag_taxonomy.parquet ]] || "$PYTHON_BIN" -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copy(hf_hub_download('Mexes/danbooru-tag-taxonomy', 'tags.parquet', repo_type='dataset'), 'data/danbooru_tag_taxonomy.parquet')"

POSTS_DIR="${POSTS_DIR:-$HOME/hf/danbooru2025-metadata}"
if [[ ! -s data/character_tags.jsonl ]]; then
    log "danbooru2025-metadata snapshot -> $POSTS_DIR (one download, resumable)"
    for attempt in 1 2 3 4 5 6; do
        "$(dirname "$PYTHON_BIN")/hf" download trojblue/danbooru2025-metadata --repo-type dataset             --include 'data/*.parquet' --local-dir "$POSTS_DIR" --max-workers 4 >/dev/null && break
        log "download attempt $attempt failed; retrying in 60 s"; sleep 60
    done
    log "character identity tags (local scan of 43 parquet files)"
    DANBOORU_POSTS_DIR="$POSTS_DIR/data" "$PYTHON_BIN" datagen/build_character_tags.py --min-post-count 1500
fi
if [[ ! -s "runs/$RUN/manifest.jsonl" ]]; then
    log "manifest"
    "$PYTHON_BIN" datagen/build_manifest_scenes.py --config configs/datagen_scenes.json
fi

case "$MODE" in
    prep) log "prep done: runs/$RUN/manifest_summary.json" ;;
    smoke) bash deploy/launch.sh "$RUN" smoke; wait ;;
    production) bash deploy/launch.sh "$RUN" production; log "workers started; deploy/status.sh $RUN" ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
