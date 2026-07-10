source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

GRANULARITY="${GRANULARITY:-coarse}"

python qwen-vl-finetune/scripts/curation/annotate_robotwin_vlm.py \
  --root /media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth \
  --raw-root /media/damoxing/datasets/robotwin-depth-f1 \
  --granularity "${GRANULARITY}" \
  --overwrite \
  "$@"
