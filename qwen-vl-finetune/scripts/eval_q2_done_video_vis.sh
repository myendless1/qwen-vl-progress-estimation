#!/usr/bin/env bash
# Visualize Q2 on random test-split episodes (per test task repo), with model inference.
set -euo pipefail
source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

CHECKPOINT="${CHECKPOINT:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2/checkpoint-4000}"
BASE_MODEL="${BASE_MODEL:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2/eval_q2_ckpt2000_3cam_rgb/done_errors}"
PER_REPO_EPISODES="${PER_REPO_EPISODES:-2}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"

python qwen-vl-finetune/tools/visualize/q2_split_vis.py \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --data-root "${DATA_ROOT}" \
  --anno-root "${ANNO_ROOT}" \
  --views "${VIEWS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split test \
  --test-ratio 0.05 \
  --split-seed 0 \
  --num-tasks 0 \
  --per-repo-episodes "${PER_REPO_EPISODES}" \
  --seed "${SEED}" \
  --q2-frame-stride 1 \
  --q2-progress-bucket-size 0.01 \
  --boundary-extra-frames 2 \
  --batch-size "${BATCH_SIZE}" \
  --video-fps 2.0 \
  --device cuda
