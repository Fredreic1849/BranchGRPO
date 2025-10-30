#!/bin/bash
# BranchGRPO Training Script - 8 GPUs
# Key Features:
#   1. Support for branch sampling and tree rollout
#   2. Use custom split points --tree_split_points
#   3. Support depth and width pruning strategies
#   4. Mixed ODE/SDE sliding window optimization
#   5. Automatic experiment naming and directory organization

conda activate dancegrpo
# export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"

# Create root directories (specific experiment directories will be created automatically by code)
mkdir -p images_branchgrpo log checkpoints tmp

# sudo apt-get update
# yes | sudo apt-get install python3-tk

git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e . 
cd ..

pip3 install trl

# ===== Core Training Parameters =====
TREE_SPLIT_POINTS="0,3,6,9"
TREE_SPLIT_NOISE_SCALE="4.0"

# Tree mixed ODE/SDE: SDE within window, ODE outside window; split steps always SDE
MIX_ODE_SDE_TREE="false"
MIX_SDE_WINDOW_SIZE="4"

TREE_PROB_WEIGHTED="false"   # true/false controls whether to enable child-edge log_prob softmax weighting

# ===== Pruning Strategy Parameters =====
DEPTH_PRUNING=""                            # Depth pruning window, e.g., "15,16,17,18"
DEPTH_PRUNING_SLIDE=true                   # Whether to enable sliding window
DEPTH_PRUNING_SLIDE_INTERVAL=30             # Sliding window interval
DEPTH_PRUNING_STOP_DEPTH=9                  # Sliding stop depth

WIDTH_PRUNING_MODE="0"                      # Width pruning mode: 0=off, 1=best per branch, 2=global best/worst

# Convert boolean switches to command line flags
if [[ "$TREE_PROB_WEIGHTED" == "true" ]]; then
  TREE_PROB_WEIGHTED_FLAG="--tree_prob_weighted"
else
  TREE_PROB_WEIGHTED_FLAG=""
fi

if [[ "$MIX_ODE_SDE_TREE" == "true" ]]; then
  MIX_ODE_SDE_TREE_FLAG="--mix_ode_sde_tree --mix_sde_window_size ${MIX_SDE_WINDOW_SIZE}"
else
  MIX_ODE_SDE_TREE_FLAG=""
fi

torchrun --nproc_per_node=8 --master_port 19003 \
    fastvideo/train_branchgrpo_flux.py \
    --seed 42 \
    --pretrained_model_name_or_path ./data/flux \
    --vae_model_path ./data/flux \
    --cache_dir ./data/.cache \
    --data_json_path ./data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 12 \
    --max_train_steps 300 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_tree \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 16 \
    --shift 3 \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-3 \
    --adv_clip_max 5.0 \
    --tree_split_points "${TREE_SPLIT_POINTS}" \
    --tree_split_noise_scale ${TREE_SPLIT_NOISE_SCALE} \
    --depth_pruning "${DEPTH_PRUNING}" \
    $( [[ "$DEPTH_PRUNING_SLIDE" == "true" ]] && echo --depth_pruning_slide ) \
    --depth_pruning_slide_interval ${DEPTH_PRUNING_SLIDE_INTERVAL} \
    $( [[ -n "${DEPTH_PRUNING_STOP_DEPTH}" ]] && echo --depth_pruning_stop_depth ${DEPTH_PRUNING_STOP_DEPTH} ) \
    --width_pruning_mode ${WIDTH_PRUNING_MODE} \
    --pruning_step_ratio ${PRUNING_STEP_RATIO} \
    ${TREE_PROB_WEIGHTED_FLAG} \
    ${MIX_ODE_SDE_TREE_FLAG}
