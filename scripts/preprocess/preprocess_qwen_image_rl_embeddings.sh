

pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0


GPU_NUM=8 # 2,4,8
MODEL_PATH="data/qwenimage"
OUTPUT_DIR="data/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_qwenimage_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/prompts.txt"

