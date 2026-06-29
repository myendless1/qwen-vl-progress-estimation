source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

python qwen-vl-finetune/scripts/curation/annotate_robotwin_vlm.py \
  --root /media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin \
  --raw-root /media/damoxing/datasets/RoboTwin2_0/dataset \
  --overwrite \
#   --only place_bread_skillet-aloha-agilex_clean_50 \
