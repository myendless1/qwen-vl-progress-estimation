#!/usr/bin/env bash
# Visualize Q2 on random train-split episodes (one episode per sampled task repo).
set -euo pipefail
source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

CHECKPOINT="${CHECKPOINT:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2/checkpoint-6000}"
BASE_MODEL="${BASE_MODEL:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2/q2_train_vis}"
NUM_TASKS="${NUM_TASKS:-15}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"

python qwen-vl-finetune/tools/visualize/q2_split_vis.py \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --data-root "${DATA_ROOT}" \
  --anno-root "${ANNO_ROOT}" \
  --views "${VIEWS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split train \
  --test-ratio 0.05 \
  --split-seed 0 \
  --num-tasks "${NUM_TASKS}" \
  --per-repo-episodes 1 \
  --seed "${SEED}" \
  --q2-frame-stride 1 \
  --q2-progress-bucket-size 0.01 \
  --boundary-extra-frames 2 \
  --batch-size "${BATCH_SIZE}" \
  --video-fps 2.0 \
  --device cuda
