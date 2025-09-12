
pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0


GPU_NUM=8 # 2,4,8
MODEL_PATH="data/qwenimage_edit"
OUTPUT_DIR="data/rl_embeddings"


torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_qwenimage_edit_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/edit_data.jsonl" \
    --height 512 \
    --width 512 