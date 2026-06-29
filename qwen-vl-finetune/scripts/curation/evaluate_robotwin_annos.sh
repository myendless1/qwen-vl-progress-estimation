source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

python qwen-vl-finetune/scripts/curation/evaluate_robotwin_annos.py \
  --root /media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin_gt_depth \
  --min-frames 7 \
  --log-name robotwin_anno_eval_latest
