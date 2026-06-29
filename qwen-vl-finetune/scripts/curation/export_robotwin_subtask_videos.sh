source /media/damoxing/fileset/conda/bin/activate qwen3-vl-ft-min

DA3_SITE=$(/media/damoxing/fileset/conda/envs/da3/bin/python -c 'import site; print(site.getsitepackages()[0])')

PYTHONPATH="$DA3_SITE" python /media/damoxing/fileset/Qwen3-VL/qwen-vl-finetune/scripts/curation/export_robotwin_subtask_videos.py \
  --root /media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin \
  --output /media/damoxing/datasets/vae4d/lerobot-vae4d-org/robotwin/_subtask_review_videos/issue_samples_latest \
  --mode review \
  --issues-log /media/damoxing/fileset/Qwen3-VL/qwen-vl-finetune/scripts/curation/tests/robotwin_anno_eval_latest.jsonl \
  --issue-sample-count 10 \
  --selection random \
  --seed 0
