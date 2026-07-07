#!/usr/bin/env bash
# Fine-tune RobotWin from a JSON config in qwen-vl-finetune/config.
set -euo pipefail
cd /media/damoxing/fileset/Qwen3-VL

FINETUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FINETUNE_ROOT}:${PYTHONPATH:-}"

CONFIG="${1:-${CONFIG:-${FINETUNE_ROOT}/config/robotwin_qwen3vl_2b.json}}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-/media/damoxing/fileset/conda/envs/qwen3-vl-ft-min/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
export MASTER_ADDR MASTER_PORT
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${FINETUNE_ROOT}/qwenvl/train/train_qwen.py" \
  --config "${CONFIG}" \
  "$@"
