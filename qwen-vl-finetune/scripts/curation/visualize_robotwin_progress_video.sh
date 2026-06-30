#!/usr/bin/env bash
set -euo pipefail

source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

ROOT="${ROOT:-/media/damoxing/datasets/VLN-CE/cogwam_data/20260629}"
REPO="${REPO:-real__tienkung_station_dualArm-gripper-3cameras_13__tienkung_station_dualArm-gripper-3cameras_13_D-K-02_01_20260313_1}"
EPISODE_INDEX="${EPISODE_INDEX:-76}"
OUTPUT="${OUTPUT:-${ROOT}/_progress_videos/${REPO}_episode_$(printf '%06d' "${EPISODE_INDEX}")_progress.mp4}"

cd /media/damoxing/fileset/Qwen3-VL/qwen-vl-finetune
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python tools/visualize/progress_video.py \
  --root "${ROOT}" \
  --repo "${REPO}" \
  --episode-index "${EPISODE_INDEX}" \
  --output "${OUTPUT}" \
  --stride 2 \
  --fps 10 \
  "$@"
