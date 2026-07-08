#!/usr/bin/env bash
# Benchmark RobotWin Q2 latency from a JSON config in qwen-vl-finetune/config.
set -euo pipefail

source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

FINETUNE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FINETUNE_ROOT}:${PYTHONPATH:-}"

CONFIG="${1:-${CONFIG:-${FINETUNE_ROOT}/config/robotwin_qwen3vl_2b_voting.json}}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${FINETUNE_ROOT}/tools/eval/benchmark_q2_latency.py" \
  --config "${CONFIG}" \
  "$@"
