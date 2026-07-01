#!/usr/bin/env bash
# Fine-tune Qwen3-VL-2B on RobotWin simulator data.
#
# Multi-node launch (platform convention):
#   export NNODES=$WORLD_SIZE
#   export NODE_RANK=$RANK
#   export MASTER_ADDR=<rank0 host>
#   export MASTER_PORT=29500
#   bash qwen-vl-finetune/scripts/sft_robotwin_qwen3vl_2b.sh
set -euo pipefail
cd /media/damoxing/fileset/Qwen3-VL

FINETUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FINETUNE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/media/damoxing/fileset/conda/envs/qwen3-vl-ft-min/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR MASTER_PORT
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
# RGB video HDF5: .../videos_240x320_240x320/chunk-*/observation.images.cam_*/episode_*.hdf5 (frames)
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
ROBOTWIN_VIEWS="${ROBOTWIN_VIEWS:-main,left_wrist,right_wrist}"
ROBOTWIN_EXCLUDE_EPISODES="${ROBOTWIN_EXCLUDE_EPISODES:-${FINETUNE_ROOT}/scripts/curation/tests/robotwin_anno_eval_exclude_episodes.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2}"
ROBOTWIN_INIT_CHECKPOINT="${ROBOTWIN_INIT_CHECKPOINT:-${OUTPUT_DIR}/pytorch_model.bin}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export TENSORBOARD_LOGGING_DIR="${TENSORBOARD_LOGGING_DIR:-${OUTPUT_DIR}/tb}"

EXTRA_ARGS=()
if [[ -f "${ROBOTWIN_EXCLUDE_EPISODES}" ]]; then
  EXTRA_ARGS+=(--robotwin_exclude_episodes "${ROBOTWIN_EXCLUDE_EPISODES}")
fi
if [[ -f "${ROBOTWIN_INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--robotwin_init_checkpoint "${ROBOTWIN_INIT_CHECKPOINT}")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${FINETUNE_ROOT}/qwenvl/train/train_qwen.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --robotwin_data_root "${DATA_ROOT}" \
  --robotwin_anno_root "${ANNO_ROOT}" \
  --robotwin_views "${ROBOTWIN_VIEWS}" \
  --robotwin_test_ratio 0.05 \
  --robotwin_q2_frame_stride 1 \
  --robotwin_q2_progress_bucket_size 0.01 \
  --robotwin_boundary_extra_frames 2 \
  --robotwin_done_sample_prob 0.4 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --model_max_length 4096 \
  --bf16 True \
  --gradient_checkpointing True \
  --tune_mm_llm True \
  --tune_mm_mlp False \
  --tune_mm_vision False \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 3 \
  --report_to tensorboard \
  --logging_steps 10 \
  "${EXTRA_ARGS[@]}"
