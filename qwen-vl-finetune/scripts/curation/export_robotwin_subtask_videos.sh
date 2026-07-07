#!/usr/bin/env bash
set -euo pipefail

source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

DA3_SITE=$(/media/damoxing/fileset/conda/envs/da3/bin/python -c 'import site; print(site.getsitepackages()[0])')

ROOT="${ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ISSUES_LOG="${ISSUES_LOG:-}"
SEED="${SEED:-0}"

args=(
  --root "${ROOT}"
  --mode review
  --selection random
  --seed "${SEED}"
)

if [[ -n "${ISSUES_LOG}" ]]; then
  OUTPUT="${OUTPUT:-${ROOT}/_subtask_review_videos/issue_samples_latest}"
  ISSUE_SAMPLE_COUNT="${ISSUE_SAMPLE_COUNT:-10}"
  args+=(
    --output "${OUTPUT}"
    --issues-log "${ISSUES_LOG}"
    --issue-sample-count "${ISSUE_SAMPLE_COUNT}"
  )
else
  OUTPUT="${OUTPUT:-${ROOT}/_subtask_review_videos/random_task_samples_3}"
  SAMPLE_COUNT="${SAMPLE_COUNT:-3}"
  args+=(
    --output "${OUTPUT}"
    --one-per-task
    --sample-count "${SAMPLE_COUNT}"
  )
fi

PYTHONPATH="$DA3_SITE" python /media/damoxing/fileset/Qwen3-VL/qwen-vl-finetune/scripts/curation/export_robotwin_subtask_videos.py "${args[@]}"
