# BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models (WIP)

**BranchGRPO** is a novel approach that restructures the rollout process into a branching tree, where shared prefixes amortize computation and pruning removes low-value paths and redundant depths.

📄 **Paper**: [arXiv:2509.06040](https://arxiv.org/abs/2509.06040)  
🌐 **Project Page**: [https://fredreic1849.github.io/BranchGRPO-Webpage/](https://fredreic1849.github.io/BranchGRPO-Webpage/)  
💻 **Code**: [GitHub Repository](https://github.com/your-username/BranchGRPO)

## Abstract

Recent progress in aligning image and video generative models with Group Relative Policy Optimization (GRPO) has improved human preference alignment, yet existing approaches still suffer from high computational cost due to sequential rollouts and large numbers of SDE sampling steps, as well as training instability caused by sparse rewards. In this paper, we present BranchGRPO, a method that restructures the rollout process into a branching tree, where shared prefixes amortize computation and pruning removes low-value paths and redundant depths.

## Key Features

BranchGRPO introduces three main contributions:

1. **Branch Sampling Scheme**: Reduces rollout cost by reusing common segments
2. **Tree-based Advantage Estimator**: Converts sparse terminal rewards into dense, step-level signals  
3. **Pruning Strategies**: Accelerate convergence while preserving exploration

## Performance

- **16% improvement** in alignment scores over strong baselines on HPDv2.1 image alignment
- **55% reduction** in per-iteration training time
- Higher Video-Align scores with sharper and temporally consistent frames on WanX-1.3B video generation

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 8+ GPUs (H800/A100 recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/BranchGRPO.git
cd BranchGRPO

# Set up environment
./env_setup.sh branchgrpo

# Install dependencies
pip install -r requirements.txt
```

### Download Checkpoints

1. **FLUX checkpoints**: Download from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev) to `./data/flux`
2. **HPS-v2.1 checkpoint**: Download from [here](https://huggingface.co/xswu/HPSv2/tree/main) to `./hps_ckpt`
3. **CLIP H-14 checkpoint**: Download from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) to `./hps_ckpt`

### Quick Start


#### Multi-GPU Training
```bash
# Preprocess embeddings (8 GPUs)
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh

# Train with BranchGRPO (8 GPUs)
bash scripts/finetune/finetune_flux_branchgrpo_8gpus.sh
```

Note: For multi-node training, please configure the launcher (e.g., Slurm, torchrun, MPI) according to your own cluster environment.

### Configuration

Key parameters for BranchGRPO:

- `--tree_split_points`: Comma-separated split points (e.g., "0,3,6,9")
- `--tree_split_noise_scale`: Noise scale for tree splits (default: 4.0)
- `--depth_pruning`: Depths to prune from training (e.g., "15,16,17")
- `--width_pruning_mode`: Width pruning strategy (0=none, 1=best per branch, 2=global best/worst)
- `--mix_ode_sde_tree`: Enable mixed ODE/SDE rollout

## Method Overview

BranchGRPO restructures sequential GRPO rollouts into a branching tree:

1. **Branching Rollouts**: At selected denoising steps, trajectories split into multiple children that share early prefixes
2. **Reward Fusion**: Leaf rewards are fused upward using path-probability weighting
3. **Depth-wise Normalization**: Normalized per depth to obtain dense, step-wise advantages
4. **Pruning**: Lightweight width and depth pruning limit backpropagation to selected nodes

## Results

### Efficiency-Quality Comparison

| Method              | NFE π_θ_old | NFE π_θ | Iteration Time (s)↓ | HPS-v2.1↑ | Pick Score↑ | Image Reward↑ |
| ------------------- | ----------- | ------- | ------------------- | --------- | ----------- | ------------- |
| FLUX                | -           | -       | -                   | 0.313     | 0.227       | 1.112         |
| DanceGRPO (tf=1.0)  | 20          | 20      | 698                 | 0.360     | 0.229       | 1.189         |
| DanceGRPO (tf=0.6)  | 20          | 12      | 469                 | 0.353     | 0.228       | 1.219         |
| MixGRPO (20,5)      | 20          | 5       | 289                 | 0.359     | 0.228       | 1.211         |
| BranchGRPO          | 13.68       | 13.68   | 493                 | 0.363     | 0.229       | 1.233         |
| BranchGRPO-WidPru   | 13.68       | 8.625   | 314                 | 0.364     | 0.230       | 1.300         |
| BranchGRPO-DepPru   | 13.68       | 8.625   | 314                 | **0.369** | **0.231**   | **1.319**     |
| BranchGRPO-Mix      | 13.68       | 4.25    | 148                 | 0.363     | 0.230       | 1.290         |


## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Acknowledgments

This work builds upon:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [diffusers](https://github.com/huggingface/diffusers)

## Citation

If you use BranchGRPO in your research, please cite our paper:

```bibtex
@article{li2025branchgrpo,
  title={BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models},
  author={Li, Yuming and Wang, Yikai and Zhu, Yuying and Zhao, Zhongyu and Lu, Ming and She, Qi and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2509.06040},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.