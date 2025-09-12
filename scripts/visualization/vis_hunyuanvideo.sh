GPU_NUM=8 

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
 scripts/visualization/vis_hunyuanvideo.py \
  --pretrained_model_name_or_path data/HunyuanVideo \
  --data_json_path data/rl_embeddings/videos2caption.json \
  --vae_model_path data/HunyuanVideo \
  --output_dir ./assets/hunyuanvideo_visualization/rlhf \
  --num_visualization_samples 8 \
  --h 640 \
  --w 640 \
  --t 53 \
  --fps 18 \
  --sampling_steps 30 \
  --eta 0.0 \
  --shift 7 \
  --seed 312 
