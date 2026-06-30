#!/usr/bin/env bash
set -euo pipefail

source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

ROOT="${ROOT:-/media/damoxing/datasets/VLN-CE/cogwam_data/20260629}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${FINETUNE_ROOT}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/convert_real_lerobot_to_robotwin.py" \
  --root "${ROOT}" \
  "$@"
