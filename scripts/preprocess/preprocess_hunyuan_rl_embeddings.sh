GPU_NUM=8 # 2,4,8
MODEL_PATH="data/HunyuanVideo"
OUTPUT_DIR="data/rl_embeddings"


cp -rf data/HunyuanVideo/tokenizer/* data/HunyuanVideo/text_encoder
cp -rf data/HunyuanVideo/tokenizer_2/* data/HunyuanVideo/text_encoder_2

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/video_prompts.txt" \
    --model_type hunyuan_hf