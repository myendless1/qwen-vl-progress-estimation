source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

CHECKPOINT="${CHECKPOINT:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2/checkpoint-6000}"
# RGB video HDF5 under robotwin_gt_depth (frames in videos_240x320_240x320/...)
DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth}"
ANNO_ROOT="${ANNO_ROOT:-${DATA_ROOT}}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b-f2/eval_q2_ckpt6000_3cam_rgb_train}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

python qwen-vl-finetune/tools/eval/q2.py \
  --checkpoint "${CHECKPOINT}" \
  --base-model /media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct \
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
  --device cuda
