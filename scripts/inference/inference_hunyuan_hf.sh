#!/bin/bash

find_available_port() {
  local port=$1  
  local original_port=$port  

  while true; do
    if ! lsof -i:$port > /dev/null; then
      echo $port
      return
    fi
    port=$((port+1))
  done
}

  master_port=${master_port:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
  master_port=$(find_available_port $master_port)

num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port $master_port \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path ./data/HunyuanVideo/ \
    --prompt_path "assets/prompt.txt" \
    --num_frames 16 \
    --height 480 \
    --width 480 \
    --num_inference_steps 20 \
    --output_path outputs_video/hunyuan_hf/ \
    --seed 1024 \
    --prompt_embed_path "./data/rl_small_embeddings/prompt_embed/44.pt" \
    --encoder_attention_mask_path "./data/rl_small_embeddings/prompt_attention_mask/44.pt"


