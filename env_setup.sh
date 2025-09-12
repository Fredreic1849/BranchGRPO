#!/bin/bash
# BranchGRPO 环境设置脚本

ENV_NAME=${1:-branchgrpo}

echo "Setting up BranchGRPO environment: $ENV_NAME"

# 创建 conda 环境
conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME

# 安装 PyTorch (根据你的 CUDA 版本调整)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate $ENV_NAME"