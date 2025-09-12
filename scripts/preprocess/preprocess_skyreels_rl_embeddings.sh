GPU_NUM=8 # 2,4,8
MODEL_PATH="data/SkyReels-I2V"
OUTPUT_DIR="data/rl_embeddings"

cp -rf data/SkyReels-I2V/tokenizer/* data/SkyReels-I2V/text_encoder
cp -rf data/SkyReels-I2V/tokenizer_2/* data/SkyReels-I2V/text_encoder_2

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/consist-id.txt" \
    --model_type hunyuan_hf


GPU_NUM=8 # 2,4,8
MODEL_PATH="data/SkyReels-I2V"
OUTPUT_DIR="data/empty"

torchrun --nproc_per_node=$GPU_NUM --master_port 19003 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/empty.txt" \
    --model_type hunyuan_hf