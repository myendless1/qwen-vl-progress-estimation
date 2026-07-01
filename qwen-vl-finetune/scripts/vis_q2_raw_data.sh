#!/usr/bin/env bash
# Visualize raw RobotWin Q2 training labels (undone progress buckets + all done frames).
set -euo pipefail
source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b/q2_raw_data_vis_episode}"
NUM_EPISODES="${NUM_EPISODES:-10}"
SEED="${SEED:-0}"

python qwen-vl-finetune/tools/visualize/q2_raw_data.py \
  --data-root "${DATA_ROOT}" \
  --anno-root "${ANNO_ROOT}" \
  --views "${VIEWS}" \
  --split "${SPLIT}" \
  --test-ratio 0.05 \
  --split-seed 0 \
  --q2-frame-stride 1 \
  --boundary-extra-frames 2 \
  --num-episodes "${NUM_EPISODES}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}"
