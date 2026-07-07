source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

CHECKPOINT="${CHECKPOINT:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f4-voting-done/checkpoint-8000}"
BASE_MODEL="${BASE_MODEL:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
# RGB video HDF5 under robotwin_gt_depth (frames in videos_240x320_240x320/...)
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f4-voting-done/eval_q1_ckpt8000_3cam_rgb_train_50tasks}"
MAX_TASKS="${MAX_TASKS:-50}"
VOTING_DONE="${VOTING_DONE:-1}"
DONE_VOTE_COUNT="${DONE_VOTE_COUNT:-4}"

EXTRA_ARGS=()
if [[ "${VOTING_DONE}" == "1" || "${VOTING_DONE}" == "true" || "${VOTING_DONE}" == "True" ]]; then
  EXTRA_ARGS+=(--voting-done --done-vote-count "${DONE_VOTE_COUNT}")
fi

mkdir -p "${OUTPUT_DIR}"

python qwen-vl-finetune/tools/eval/q1.py \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --data-root "${DATA_ROOT}" \
  --anno-root "${ANNO_ROOT}" \
  --views "${VIEWS}" \
  --output-json "${OUTPUT_DIR}/q1_predictions.json" \
  --output-xlsx "${OUTPUT_DIR}/q1_predictions.xlsx" \
  --split train \
  --test-ratio 0.05 \
  --split-seed 0 \
  --q2-frame-stride 8 \
  --boundary-extra-frames 2 \
  --one-example-per-task \
  --max-tasks "${MAX_TASKS}" \
  --shuffle-samples \
  "${EXTRA_ARGS[@]}" \
  --device cuda
