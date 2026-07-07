source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

CHECKPOINT="${CHECKPOINT:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f4-voting-done/checkpoint-8000}"
BASE_MODEL="${BASE_MODEL:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
# RGB video HDF5 under robotwin_gt_depth (frames in videos_240x320_240x320/...)
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f4-voting-done/eval_q2_ckpt8000_3cam_rgb_train_voting_done}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-/media/damoxing/ckp/qwen_ft/robotwin_q2_eval_manifests/train_seed0_5000_q2_stride8_bucket001_voting_ckpt8000.json}"
VOTING_DONE="${VOTING_DONE:-1}"
DONE_VOTE_COUNT="${DONE_VOTE_COUNT:-4}"
DONE_VOTE_THRESHOLD="${DONE_VOTE_THRESHOLD:-3}"

EXTRA_ARGS=()
if [[ "${VOTING_DONE}" == "1" || "${VOTING_DONE}" == "true" || "${VOTING_DONE}" == "True" ]]; then
  EXTRA_ARGS+=(--voting-done --done-vote-count "${DONE_VOTE_COUNT}" --done-vote-threshold "${DONE_VOTE_THRESHOLD}")
fi

python qwen-vl-finetune/tools/eval/q2.py \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --data-root "${DATA_ROOT}" \
  --anno-root "${ANNO_ROOT}" \
  --views "${VIEWS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split train \
  --test-ratio 0.05 \
  --split-seed 0 \
  --q2-frame-stride 8 \
  --boundary-extra-frames 2 \
  --batch-size 32 \
  --max-samples "${MAX_SAMPLES}" \
  --shuffle-samples \
  --sample-manifest "${SAMPLE_MANIFEST}" \
  "${EXTRA_ARGS[@]}" \
  --device cuda
