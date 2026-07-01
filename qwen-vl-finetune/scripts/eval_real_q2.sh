source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min
cd /media/damoxing/fileset/Qwen3-VL

DATA_ROOT="${DATA_ROOT:-/media/damoxing/datasets/VLN-CE/cogwam_data/20260629}"
VIEWS="${VIEWS:-main,left_wrist,right_wrist}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b_real/eval_q2_ckpt12000_3cam_rgb}"

python qwen-vl-finetune/tools/eval/q2.py \
  --checkpoint /media/damoxing/ckp/qwen_ft/robotwin_qwen3vl_2b_real/checkpoint-12000 \
  --base-model /media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct \
  --data-root "${DATA_ROOT}" \
  --views "${VIEWS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split test \
  --test-ratio 0.05 \
  --split-seed 0 \
  --q2-frame-stride 8 \
  --boundary-extra-frames 2 \
  --batch-size 32 \
  --max-samples 5000 \
  --shuffle-samples \
  --device cuda
