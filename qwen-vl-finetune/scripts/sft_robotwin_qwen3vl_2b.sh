#!/usr/bin/env bash
set -euo pipefail
cd /media/damoxing/fileset/Qwen3-VL

FINETUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FINETUNE_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/media/damoxing/fileset/conda/envs/qwen3-vl-ft-min/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b}"
ROBOTWIN_INIT_CHECKPOINT="${ROBOTWIN_INIT_CHECKPOINT:-${OUTPUT_DIR}/pytorch_model.bin}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export TENSORBOARD_LOGGING_DIR="${TENSORBOARD_LOGGING_DIR:-${OUTPUT_DIR}/tb}"

EXTRA_ARGS=()
if [[ -f "${ROBOTWIN_INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--robotwin_init_checkpoint "${ROBOTWIN_INIT_CHECKPOINT}")
fi

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=8 "${FINETUNE_ROOT}/qwenvl/train/train_qwen.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --robotwin_data_root "${DATA_ROOT}" \
  --robotwin_test_ratio 0.05 \
  --robotwin_q2_frame_stride 8 \
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
