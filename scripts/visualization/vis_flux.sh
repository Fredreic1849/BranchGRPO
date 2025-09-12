GPU_NUM=8 

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
 scripts/visualization/vis_flux.py