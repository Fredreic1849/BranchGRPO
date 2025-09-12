# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.
#
# Modified for Tree-based GRPO with split rollout strategy
# 
# Latest modifications (support multi-batch training):
# - Added use_group logic support, consistent with original train_grpo_flux.py
# - Fixed hardcoded batch_size=1 issue, now supports train_batch_size > 1
# - Support parameters: --train_batch_size 2 --train_sp_batch_size 2 --use_group
# - When using use_group, each prompt generates num_generations samples for comparison

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor
from datetime import datetime
import yaml
import subprocess
import json

def generate_experiment_name(args):
    """Generate unique experiment name based on time and key parameters"""
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = f"BranchGRPO_{now}"
    
    # Add key tree parameters to distinguish experiments
    if hasattr(args, 'tree_split_points') and args.tree_split_points:
        # If using custom split points, show number of points
        points = args.tree_split_points.split(',')
        name += f"_sp{len(points)}pts"
    elif hasattr(args, 'tree_split_rounds'):
        name += f"_r{args.tree_split_rounds}"
    
    if hasattr(args, 'tree_split_noise_scale'):
        name += f"_ns{args.tree_split_noise_scale}"
    if hasattr(args, 'learning_rate'):
        name += f"_lr{args.learning_rate}"
    if hasattr(args, 'clip_range'):
        name += f"_clip{args.clip_range}"
    if hasattr(args, 'tree_split_points'):
        name += f"_sp{args.tree_split_points}"
    
    # Add depth pruning identifier
    if hasattr(args, 'depth_pruning') and args.depth_pruning:
        depths = args.depth_pruning.split(',')
        name += f"_dp{args.depth_pruning}"  # e.g., _dp5d means pruning 5 depths
    
        if hasattr(args, 'depth_pruning_slide') and args.depth_pruning_slide:
            name += f"_dp_slide"
      
    # Add width pruning identifier
    if hasattr(args, 'width_pruning_mode') and args.width_pruning_mode is not None and args.width_pruning_mode > 0:
        ratio = getattr(args, 'width_pruning_ratio', 0.5)
        name += f"_wp{args.width_pruning_mode}"  # e.g., _wp1 means mode 1
        

    if hasattr(args, 'tree_prob_weighted') and args.tree_prob_weighted:
        name += f"_tpw"

    # Mixed ODE/SDE sliding window marker
    if hasattr(args, 'mix_ode_sde_tree') and args.mix_ode_sde_tree:
        win = getattr(args, 'mix_sde_window_size', 4)
        name += f"_mixwin{win}"

  
    return name

def save_experiment_config(args, exp_name, rank):
    """Save experiment configuration and Git information"""
    if rank > 0:
        return  # Only save in main process
    
    # Create experiment directory
    exp_log_dir = f"log/{exp_name}"
    os.makedirs(exp_log_dir, exist_ok=True)
    
    # Save configuration as YAML
    config_path = f"{exp_log_dir}/config.yaml"
    config_dict = vars(args)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
    
    # Save Git information
    git_info_path = f"{exp_log_dir}/git_info.txt"
    try:
        # Get Git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                            stderr=subprocess.DEVNULL).decode().strip()
        # Get branch name
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                       stderr=subprocess.DEVNULL).decode().strip()
        # Get working directory status
        status = subprocess.check_output(['git', 'status', '--porcelain'], 
                                       stderr=subprocess.DEVNULL).decode().strip()
        
        git_info = f"Commit: {commit_hash}\nBranch: {branch}\n"
        if status:
            git_info += f"Working directory status:\n{status}\n"
        else:
            git_info += "Working directory clean\n"
            
        with open(git_info_path, 'w') as f:
            f.write(git_info)
    except (subprocess.CalledProcessError, FileNotFoundError):
        with open(git_info_path, 'w') as f:
            f.write("Git information not available\n")
    
    print(f"Experiment config saved to {exp_log_dir}")

def parse_split_points(args, total_steps):
    """Parse split points parameters"""
    if hasattr(args, 'tree_split_points') and args.tree_split_points:
        # Use custom split points
        points = [int(p.strip()) for p in args.tree_split_points.split(',')]
        # Ensure split points are within valid range
        points = [min(max(p, 0), total_steps - 1) for p in points]
        return sorted(points)
    else:
        # Use default uniform splitting
        if hasattr(args, 'tree_split_rounds') and args.tree_split_rounds > 0:
            if total_steps % args.tree_split_rounds != 0:
                print(f"Warning: total_steps ({total_steps}) is not divisible by tree_split_rounds ({args.tree_split_rounds})")
            split_interval = total_steps // args.tree_split_rounds
            points = [i * split_interval for i in range(args.tree_split_rounds)]
            return [min(p, total_steps - 1) for p in points]
        else:
            return []

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List, Dict, Tuple
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL
import random


class TreeNode:
    """Tree node for tracking branching rollout tree structure"""
    def __init__(self, node_id: str, latent: torch.Tensor, parent=None, step: int = 0, batch_idx: int = 0):
        self.node_id = node_id
        self.latent = latent
        self.parent = parent
        self.children = []
        self.step = step
        self.batch_idx = batch_idx  # Add batch_idx field to record which batch sample this belongs to
        self.log_prob = None  # log_prob from parent to this node
        self.reward = None    # Node reward value (leaf nodes have actual reward, internal nodes have aggregated reward)
        self.advantage = None # Node advantage value
        self.depth = 0 if parent is None else parent.depth + 1  # Node depth
        self.is_sde_edge = None  # Whether the edge from parent to this node is SDE generated (for training filtering)
        
    def add_child(self, child):
        self.children.append(child)
        child.depth = self.depth + 1  # Update child node depth
        
    def is_leaf(self):
        return len(self.children) == 0
        
    def get_path_from_root(self):
        """Get path from root node to current node"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_all_leaf_descendants(self):
        """Get all leaf descendants of current node"""
        if self.is_leaf():
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaf_descendants())
        return leaves


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step_with_split(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
    num_splits: int = 1,
    split_noise_scale: float = 1.0,
):
    """
    Modified flux_step that supports splitting operations
    When num_splits > 1, generates multiple different samples
    """
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)
    
    # Numerical stability protection: prevent std_dev_t from being too small, use device-consistent tensor
    std_dev_t = torch.clamp(torch.as_tensor(std_dev_t, device=latents.device, dtype=torch.float32), min=1e-8)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo:
        if prev_sample is None:
            # Splitting strategy: use independent randomness to create diversity without changing global RNG
            if num_splits > 1:
                split_samples = []
                log_probs = []
                
                for i in range(num_splits):
                    # Use split_noise_scale to control noise intensity without changing global random seed
                    noise = torch.randn_like(prev_sample_mean) * split_noise_scale
                    sample = prev_sample_mean + noise * std_dev_t
                    split_samples.append(sample)
                    
                    # log_prob calculation
                    two_pi = torch.as_tensor(2 * math.pi, device=prev_sample_mean.device, dtype=torch.float32)
                    log_prob = (
                        -((sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
                    ) - torch.log(std_dev_t) - 0.5 * torch.log(two_pi)
                    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
                    log_probs.append(log_prob)
                
                # Collect results from all branches
                prev_sample = torch.cat(split_samples, dim=0)
                log_prob = torch.cat(log_probs, dim=0)
                # pred_original_sample is the same for all branches, expand with correct dimensions
                if pred_original_sample.dim() == 3:  # [B, num_patches, channels]
                    pred_original_sample = pred_original_sample.repeat(num_splits, 1, 1)
                else:  # [B, C, H, W]
                    pred_original_sample = pred_original_sample.repeat(num_splits, 1, 1, 1)
                
            else:
                # Original single sample logic
                prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
                two_pi = torch.as_tensor(2 * math.pi, device=prev_sample_mean.device, dtype=torch.float32)
                log_prob = (
                    -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
                ) - torch.log(std_dev_t) - 0.5 * torch.log(two_pi)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        else:
            # When prev_sample is not None, calculate log_prob for given prev_sample
            two_pi = torch.as_tensor(2 * math.pi, device=prev_sample_mean.device, dtype=torch.float32)
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
            ) - torch.log(std_dev_t) - 0.5 * torch.log(two_pi)
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if grpo:
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def run_tree_sample_step(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    image_ids, 
    grpo_sample,
    num_final_branches=16,
    num_split_rounds=4,
    batch_offset=0,  # Add batch_offset parameter to correctly set batch_idx
):
    """
    Branching rollout sampling step - supports multi-batch
    """
    if not grpo_sample:
        raise NotImplementedError("Tree sampling only supports grpo_sample=True")
    
    total_steps = len(sigma_schedule) - 1
    
    # Use new split points parsing function
    split_points = parse_split_points(args, total_steps)
    
    # Fix: Create independent root nodes for each batch sample, but maintain unified node list processing
    batch_size = z.shape[0]
    current_nodes = []
    
    for batch_idx in range(batch_size):
        # Create independent root node for each sample
        sample_z = z[batch_idx:batch_idx+1]  # Maintain dimensions: [1, seq_len, channels]
        # Fix: Use batch_offset to ensure correct batch_idx
        actual_batch_idx = batch_offset + batch_idx
        root = TreeNode(f"root_b{actual_batch_idx}", sample_z, parent=None, step=0, batch_idx=actual_batch_idx)
        current_nodes.append(root)
    
    all_log_probs_tree = {}
    
    main_print(f"Tree sampling: {total_steps} steps, split at {split_points}")
    main_print(f"Expected final branches per sample: {2**num_split_rounds}")
    main_print(f"Batch size: {batch_size}, Initial nodes: {len(current_nodes)}")

    # This variable will be overwritten at each step, and after the last step loop, it will save the final clean image prediction
    final_pred_originals = None

    for i in progress_bar:
        new_nodes = []  # Store all new nodes for current step
        step_pred_originals = []
        should_split = i in split_points
        num_splits = 2 if should_split else 1

        if should_split:
            main_print(f"Split at step {i}: nodes={len(current_nodes)} → {len(current_nodes)*2}")
        # Sliding window: SDE within window, ODE outside window (split steps always SDE)
        use_mix = getattr(args, 'mix_ode_sde_tree', False)
        window_size = int(getattr(args, 'mix_sde_window_size', 4))
        slide_interval = int(getattr(args, 'depth_pruning_slide_interval', 1))
        if use_mix:
            stride = max(1, slide_interval)
            window_start = (i // stride) * stride  # Starting position from 0, segmented by stride
            window_end = min(len(sigma_schedule) - 2, window_start + max(1, window_size) - 1)
            in_window = (i >= window_start and i <= window_end)
            if (i == 0 or should_split or (i % stride == 0)) and (dist.get_rank() % 8 == 0):
                main_print(f"[MIX] step {i}: window=({window_start},{window_end}), in_window={in_window}, split={should_split}")
        else:
            in_window = True  # When mixing not enabled, default to full SDE

        transformer.eval()
        with torch.autocast("cuda", torch.bfloat16):
            # Key fix: Process each node independently, but don't group by batch
            for node in current_nodes:
                sigma = sigma_schedule[i]
                timestep_value = int(sigma * 1000)
                timestep = torch.full([1], timestep_value, device=z.device, dtype=torch.long)
                
                # Get corresponding input parameters based on node's batch_idx
                node_batch_idx = node.batch_idx - batch_offset  # Convert to index within current batch
                sample_encoder_hidden_states = encoder_hidden_states[node_batch_idx:node_batch_idx+1]
                sample_pooled_prompt_embeds = pooled_prompt_embeds[node_batch_idx:node_batch_idx+1] 
                sample_text_ids = text_ids[node_batch_idx:node_batch_idx+1]
                sample_image_ids = image_ids  # image_ids are usually shared
                img_ids_for_node = sample_image_ids.squeeze(0) if sample_image_ids.dim() == 3 else sample_image_ids
                
                # 1. Make prediction for single node
                pred = transformer(
                    hidden_states=node.latent,  # Now ensure it's [1, seq_len, channels]
                    encoder_hidden_states=sample_encoder_hidden_states,
                    timestep=timestep/1000,
                    guidance=torch.tensor([3.5], device=z.device, dtype=torch.bfloat16),
                    txt_ids=sample_text_ids.repeat(sample_encoder_hidden_states.shape[1], 1),  # Restore original logic
                    pooled_projections=sample_pooled_prompt_embeds,
                    img_ids=img_ids_for_node,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # 2. Step forward with prediction results for single node:
                #    - Split step: always SDE
                #    - When sliding window enabled: SDE within window, ODE outside window
                #    - When mixing not enabled: keep SDE
                if (use_mix and (not should_split) and (not in_window)):
                    sigma = sigma_schedule[i]
                    dsigma = sigma_schedule[i + 1] - sigma
                    # Only do deterministic mean update, don't inject noise; log_prob set to 0 (will be filtered during training)
                    prev_sample_mean = node.latent.to(torch.float32) + dsigma * pred
                    # Align with existing interface: return shape consistent with SDE
                    next_latents_for_node = prev_sample_mean[0:1]
                    pred_original_for_node = (node.latent.to(torch.float32) - sigma * pred)[0:1]
                    zero_lp = torch.zeros((1,), device=node.latent.device, dtype=torch.float32)
                    log_probs_for_node = zero_lp
                    edge_is_sde = False
                else:
                    next_latents_for_node, pred_original_for_node, log_probs_for_node = flux_step_with_split(
                        pred, node.latent.to(torch.float32), args.eta, 
                        sigmas=sigma_schedule, index=i, prev_sample=None, 
                        grpo=True, sde_solver=True, num_splits=num_splits, 
                        split_noise_scale=args.tree_split_noise_scale
                    )
                    edge_is_sde = True

                # 3. Create child nodes based on step results
                if should_split:
                    for split_idx in range(num_splits):
                        child_id = f"{node.node_id}_s{i}_{split_idx}"
                        child_latent = next_latents_for_node[split_idx:split_idx+1].to(torch.bfloat16)
                        child_log_prob = log_probs_for_node[split_idx:split_idx+1]
                        
                        child = TreeNode(child_id, child_latent, parent=node, step=i+1, batch_idx=node.batch_idx)
                        child.log_prob = child_log_prob
                        child.is_sde_edge = True  # Split steps are always SDE
                        node.add_child(child)
                        new_nodes.append(child)
                        
                        step_pred_originals.append(pred_original_for_node[split_idx:split_idx+1])

                        if child_id not in all_log_probs_tree:
                           all_log_probs_tree[child_id] = []
                        all_log_probs_tree[child_id].append(child_log_prob)
                else: # No split
                    child_id = f"{node.node_id}_t{i}"
                    child_latent = next_latents_for_node[0:1].to(torch.bfloat16)
                    child_log_prob = log_probs_for_node[0:1]

                    child = TreeNode(child_id, child_latent, parent=node, step=i+1, batch_idx=node.batch_idx)
                    child.log_prob = child_log_prob
                    child.is_sde_edge = edge_is_sde
                    node.add_child(child)
                    new_nodes.append(child)

                    step_pred_originals.append(pred_original_for_node)

                    if child_id not in all_log_probs_tree:
                       all_log_probs_tree[child_id] = []
                    all_log_probs_tree[child_id].append(child_log_prob)

        # Key fix: Directly replace current_nodes, just like bs1 version
        current_nodes = new_nodes
        
        # Collect all clean image predictions generated in current step
        final_pred_originals = torch.cat(step_pred_originals, dim=0)

        if i == 0 or should_split:
            # Count nodes for each batch
            batch_node_counts = {}
            for node in current_nodes:
                batch_idx = node.batch_idx
                batch_node_counts[batch_idx] = batch_node_counts.get(batch_idx, 0) + 1
            main_print(f"Step {i}: {len(current_nodes)} total nodes, per batch: {batch_node_counts}")
            
    # After loop ends, current_nodes are leaf nodes
    leaf_nodes = current_nodes
    final_latents = torch.cat([node.latent for node in leaf_nodes], dim=0)
    
    # Build path log_probs for all leaf nodes
    path_log_probs = []
    for leaf in leaf_nodes:
        path = leaf.get_path_from_root()
        total_log_prob = torch.zeros_like(leaf.log_prob if leaf.log_prob is not None else torch.tensor(0.0))
        for node in path[1:]:  # Skip root node
            if node.log_prob is not None:
                total_log_prob += node.log_prob
        path_log_probs.append(total_log_prob)
    
    all_path_log_probs = torch.cat(path_log_probs, dim=0) if path_log_probs else torch.tensor([])
    
    # Return final noisy latent, clean image predictions, log_probs and tree structure
    return final_latents, final_pred_originals, all_path_log_probs, leaf_nodes, all_log_probs_tree


def compute_node_rewards_from_leaves(
    root_node: TreeNode,
    leaf_rewards: torch.Tensor,
    leaf_nodes: List[TreeNode],
    use_prob_weighted: bool = False,
):
    """
    Compute reward values for all nodes from bottom up
    Leaf nodes: use actual reward
    Internal nodes: use average reward of all leaf descendants
    """
    # 1. Assign actual rewards to leaf nodes
    for i, leaf_node in enumerate(leaf_nodes):
        leaf_node.reward = leaf_rewards[i]

    # 1.1 Only print path probability statistics when probability weighting is enabled (sum of log_prob from root to leaf then exp)
    if use_prob_weighted:
        try:
            path_probs = []
            for idx, leaf_node in enumerate(leaf_nodes):
                path = leaf_node.get_path_from_root()
                total_log_prob = None
                for node in path[1:]:  # Skip root node
                    if node.log_prob is not None:
                        lp = node.log_prob.squeeze().to(torch.float32)
                        total_log_prob = lp if total_log_prob is None else (total_log_prob + lp)
                if total_log_prob is not None:
                    prob = torch.exp(total_log_prob).item()
                    path_probs.append((prob, idx))
            if len(path_probs) > 0:
                probs_only = [p for p, _ in path_probs]
                path_probs_sorted = sorted(path_probs, key=lambda x: x[0], reverse=True)
                topk = path_probs_sorted[: min(5, len(path_probs_sorted))]
                bottomk = sorted(path_probs, key=lambda x: x[0])[: min(5, len(path_probs))]
                mean_prob = float(np.mean(probs_only)) if len(probs_only) > 0 else 0.0
                max_prob = float(topk[0][0]) if len(topk) > 0 else 0.0
                min_prob = float(bottomk[0][0]) if len(bottomk) > 0 else 0.0
                main_print(f"Path probability stats: mean={mean_prob:.4e}, max={max_prob:.4e}, min={min_prob:.4e}")
                if len(topk) > 0:
                    examples = []
                    for prob, i in topk:
                        r = leaf_nodes[i].reward
                        r_item = r.item() if isinstance(r, torch.Tensor) else float(r)
                        examples.append(f"prob={prob:.2e}, reward={r_item:.3f}")
                    main_print("Example high-probability paths: " + " | ".join(examples))
                if len(bottomk) > 0:
                    examples = []
                    for prob, i in bottomk:
                        r = leaf_nodes[i].reward
                        r_item = r.item() if isinstance(r, torch.Tensor) else float(r)
                        examples.append(f"prob={prob:.2e}, reward={r_item:.3f}")
                    main_print("Example low-probability paths: " + " | ".join(examples))
        except Exception as e:
            main_print(f"Failed to compute path probability statistics: {e}")

    # 2. Collect all nodes, group by depth
    all_nodes = []
    def collect_nodes(node):
        all_nodes.append(node)
        for child in node.children:
            collect_nodes(child)
    collect_nodes(root_node)
    
    # Group by depth
    max_depth = max(node.depth for node in all_nodes)
    nodes_by_depth = {depth: [] for depth in range(max_depth + 1)}
    for node in all_nodes:
        nodes_by_depth[node.depth].append(node)
    
    # 3. Compute intermediate node rewards from bottom up
    if use_prob_weighted:
        # 逐层子边log_prob softmax加权聚合
        alpha = 1.0  # 温度系数，最小改动：先固定为1.0
        for depth in reversed(range(max_depth)):  # 从最大深度-1往上
            for node in nodes_by_depth[depth]:
                if not node.is_leaf():
                    children = node.children
                    if len(children) == 1:
                        # 单子分支，权重为1，直接传递
                        node.reward = children[0].reward
                    else:
                        # 使用子边的log_prob做softmax权重
                        child_log_probs = []
                        child_rewards = []
                        for child in children:
                            # 保护：若缺少log_prob，退化为0
                            lp = child.log_prob
                            lp_val = lp.squeeze().to(torch.float32) if lp is not None else torch.tensor(0.0, dtype=torch.float32)
                            child_log_probs.append(lp_val)
                            child_rewards.append(child.reward)
                        child_log_probs = torch.stack(child_log_probs)
                        # 数值稳定的softmax
                        weights = torch.softmax(alpha * child_log_probs, dim=0)
                        # 对齐设备与dtype
                        device = weights.device
                        rewards_tensor = torch.stack([r.to(device=device, dtype=torch.float32) for r in child_rewards])
                        node.reward = torch.sum(weights * rewards_tensor)
    else:
        # 原始：对所有叶子后代的reward做简单平均
        for depth in reversed(range(max_depth)):  # 从最大深度-1往上
            for node in nodes_by_depth[depth]:
                if not node.is_leaf():
                    leaf_descendants = node.get_all_leaf_descendants()
                    descendant_rewards = [leaf.reward for leaf in leaf_descendants]
                    node.reward = torch.stack(descendant_rewards).mean()
    
    main_print(f"Node reward computation finished: total {len(all_nodes)} nodes")
    return all_nodes, nodes_by_depth


def compute_hierarchical_advantages_by_depth(nodes_by_depth: Dict[int, List[TreeNode]]) -> Dict[str, torch.Tensor]:
    """
    按深度分层计算advantage
    同一深度的节点之间进行相对比较
    """
    node_advantages = {}
    
    for depth, nodes in nodes_by_depth.items():
        if len(nodes) <= 1:
            # 单个节点的advantage设为0
            for node in nodes:
                node.advantage = torch.tensor(0.0)  # 修复：使用标量tensor，后续统一处理形状
                node_advantages[node.node_id] = node.advantage
        else:
            # 同层多个节点，计算相对advantage
            rewards = torch.stack([node.reward for node in nodes])
            mean_reward = rewards.mean()
            std_reward = rewards.std()
            
            # 修复：处理所有reward相同的情况
            if std_reward < 1e-6:  # 如果标准差太小，设置所有advantage为0
                advantages = torch.zeros_like(rewards)
                main_print(f"Warning: 深度{depth}的所有reward相同({mean_reward:.6f})，设置advantage为0")
            else:
                # 标准化得到advantage
                advantages = (rewards - mean_reward) / (std_reward + 1e-8)
            
            for i, node in enumerate(nodes):
                node.advantage = advantages[i]  # 修复：直接使用标量，保持一致的形状
                node_advantages[node.node_id] = node.advantage
    
    main_print(f"分层advantage计算完成，共{len(node_advantages)}个节点")
    return node_advantages




def validate_tree_training_logic(leaf_nodes, advantages):
    """
    验证树形训练逻辑的正确性
    """
    total_transitions = 0
    path_lengths = []
    
    for i, leaf_node in enumerate(leaf_nodes):
        path = leaf_node.get_path_from_root()
        path_length = len(path) - 1  # 转移数量
        path_lengths.append(path_length)
        total_transitions += path_length
        
        main_print(f"Leaf {i}: path_length={path_length}, advantage={advantages[i].item():.4f}")
    
    main_print(f"Total transitions to train: {total_transitions}")
    main_print(f"Average path length: {np.mean(path_lengths):.2f}")
    main_print(f"Path length range: [{min(path_lengths)}, {max(path_lengths)}]")
    
    return total_transitions


def grpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            transformer,
            timesteps,
            i,
            sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps/1000,
            guidance=torch.tensor(
                [3.5],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),  # 恢复原始逻辑
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids.squeeze(0) if image_ids.dim() == 3 else image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    # 仅在分裂边上训练时，log_prob计算仍需与采样分布一致。
    # 为保持简单，这里仍复用SDE式 log_prob；当调用方只传入分裂边时不会引入不一致。
    z, pred_original, log_prob = flux_step_with_split(
        pred, latents.to(torch.float32), args.eta, sigma_schedule, i, 
        prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True,
        split_noise_scale=args.tree_split_noise_scale
    )
    return log_prob


def sample_reference_model_tree(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
):
    """
    使用分裂式树形结构进行参考模型采样
    """
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps    # 20
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    # 🔧 修复：使用实际的batch_size而不是硬编码的1
    # 新逻辑：输入是原始的prompt数量，每个prompt生成一个树
    if args.use_group:
        # use_group模式：每个原始prompt生成一个树，树有num_generations个叶子节点
        batch_size = 1  # 每次处理一个原始prompt
    else:
        # 非use_group模式：每个输入样本生成一个树
        batch_size = min(B, args.train_batch_size)
    
    batch_indices = torch.chunk(torch.arange(B), max(1, B // batch_size))

    all_rewards = []  
    all_leaf_nodes = []
    
    # 对于树形采样，我们期望最终得到16个分支
    target_final_branches = args.num_generations if hasattr(args, 'num_generations') else 16
    num_split_rounds = int(math.log2(target_final_branches))  # 4轮分裂得到16个分支
    
    if args.init_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        
        if not args.init_same_noise:
            input_latents = torch.randn(
                    (len(batch_idx), IN_CHANNELS, latent_h, latent_w),
                    device=device,
                    dtype=torch.bfloat16,
                )
        else:
            # 🔧 修复：如果使用相同噪声但batch_size > 1，需要重复latents
            if len(batch_idx) > 1:
                input_latents = input_latents.repeat(len(batch_idx), 1, 1, 1)
        
        input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        
        progress_bar = tqdm(range(0, sample_steps), desc="Tree Sampling Progress")
        
        with torch.no_grad():
            final_latents, pred_original, path_log_probs, leaf_nodes, all_log_probs_tree = run_tree_sample_step(
                args,
                input_latents_new,
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                image_ids,
                grpo_sample=True,
                num_final_branches=target_final_branches,
                num_split_rounds=num_split_rounds,
                batch_offset=batch_idx[0] if len(batch_idx) > 0 else 0, # 🔧 修正：使用batch_idx的第一个元素
            )
        
        all_leaf_nodes.extend(leaf_nodes)
        vae.enable_tiling()
        
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        # 处理每个叶子节点生成的图像
        batch_rewards = []
        for leaf_idx, leaf_node in enumerate(leaf_nodes):
            latent = pred_original[leaf_idx:leaf_idx+1]
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    unpacked_latent = unpack_latents(latent, h, w, 8)
                    unpacked_latent = (unpacked_latent / 0.3611) + 0.1159
                    image = vae.decode(unpacked_latent, return_dict=False)[0]
                    decoded_image = image_processor.postprocess(image)
            
            # 保存图像到实验特定目录
            exp_name = getattr(args, '_exp_name', 'default')
            image_dir = f"images_branchgrpo/{exp_name}/rank_{rank}"
            os.makedirs(image_dir, exist_ok=True)
            image_path = f"{image_dir}/flux_branchgrpo_{index}_{leaf_idx}.png"
            decoded_image[0].save(image_path)

            # 🔧 关键修复：根据leaf_node的batch_idx获取正确的caption
            leaf_batch_idx = leaf_node.batch_idx
            # 从原始caption列表中获取对应的caption
            correct_caption = caption[leaf_batch_idx] if leaf_batch_idx < len(caption) else batch_caption[0]
            
            # 🔧 添加调试信息（只打印前几个样本）
            if leaf_idx < 5 or leaf_idx % 16 == 0:  # 每组的第一个样本
                main_print(f"Leaf {leaf_idx}: batch_idx={leaf_batch_idx}, caption='{correct_caption[:50]}...'")

            # 计算奖励
            if args.use_hpsv2:
                with torch.no_grad():
                    image_pil = decoded_image[0]
                    image_tensor = preprocess_val(image_pil).unsqueeze(0).to(device=device, non_blocking=True)
                    # 🔧 使用正确的caption而不是总是batch_caption[0]
                    text = tokenizer([correct_caption]).to(device=device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        outputs = reward_model(image_tensor, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image)
                    batch_rewards.append(hps_score)
            
            if args.use_pickscore:
                def calc_probs(processor, model, prompt, images, device):
                    image_inputs = processor(
                        images=images,
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    ).to(device)
                    text_inputs = processor(
                        text=prompt,
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        image_embs = model.get_image_features(**image_inputs)
                        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                    
                        text_embs = model.get_text_features(**text_inputs)
                        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                    
                        scores = (text_embs @ image_embs.T)[0]
                    
                    return scores
                
                pil_images = [Image.open(image_path)]
                # 🔧 使用正确的caption
                score = calc_probs(tokenizer, reward_model, correct_caption, pil_images, device)
                batch_rewards.append(score)

        all_rewards.extend(batch_rewards)

    all_rewards = torch.stack(all_rewards) if all_rewards else torch.tensor([])
    
    return all_rewards, path_log_probs, all_leaf_nodes, sigma_schedule, all_log_probs_tree


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step_tree(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    current_step=None,
    max_steps=None,
):
    """
    使用分裂式树形结构的训练步骤
    """
    total_loss = 0.0
    
    # 🕒 时间记录：开始计时
    step_start_time = time.time()
    
    (
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        caption,
    ) = next(loader)

    # 🔧 关键修复：不要在采样前进行repeat，保持原始的prompt数量
    # 原始的encoder_hidden_states: [2, 512, 4096] (2个不同的prompt)
    original_batch_size = encoder_hidden_states.shape[0]
    
    # 🕒 时间记录：采样开始
    sample_start_time = time.time()
    
    # 使用树形采样生成样本（每个原始prompt生成一个树）
    rewards, path_log_probs, leaf_nodes, sigma_schedule, all_log_probs_tree = sample_reference_model_tree(
        args,
        device, 
        transformer,
        vae,
        encoder_hidden_states,  # 保持原始维度 [2, 512, 4096]
        pooled_prompt_embeds,   # 保持原始维度 [2, ...]
        text_ids,               # 保持原始维度 [2, 3]
        reward_model,
        tokenizer,
        caption,
        preprocess_val,
    )
    
    # 🕒 时间记录：采样结束，奖励计算开始
    sample_end_time = time.time()
    reward_start_time = sample_end_time
    
    # 🔧 树采样完成后，应用use_group逻辑
    if args.use_group:
        # 验证采样结果的数量
        expected_samples = original_batch_size * args.num_generations
        actual_samples = len(rewards)
        main_print(f"Use_group mode: expected {expected_samples} samples, got {actual_samples}")
        
        if actual_samples != expected_samples:
            main_print(f"Warning: Sample count mismatch! Expected {expected_samples}, got {actual_samples}")
        
        # 🔧 验证reward和leaf_nodes的batch_idx对应关系
        main_print("Verifying reward-batch_idx correspondence:")
        for i in range(min(10, len(leaf_nodes))):  # 只打印前10个样本
            leaf = leaf_nodes[i]
            reward = rewards[i].item() if hasattr(rewards[i], 'item') else rewards[i]
            main_print(f"  Sample {i}: batch_idx={leaf.batch_idx}, reward={reward:.4f}")
        
        # 为caption应用use_group逻辑（用于reward计算时的匹配）
        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")
    
    # 收集分布式奖励
    gathered_reward = gather_tensor(rewards)
    if dist.get_rank() == 0:
        print("gathered_reward (tree)", gathered_reward)
        exp_name = getattr(args, '_exp_name', 'default')
        log_dir = f"log/{exp_name}"
        os.makedirs(log_dir, exist_ok=True)
        with open(f'{log_dir}/reward_tree.txt', 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")
    
    # 🕒 时间记录：奖励计算结束
    reward_end_time = time.time()

    # 🌟 添加 use_group 的 advantage 计算逻辑
    # 1. 从叶子节点收集所有根节点
    root_nodes = []
    processed_roots = set()
    for leaf in leaf_nodes:
        current = leaf
        while current.parent is not None:
            current = current.parent
        if current.node_id not in processed_roots:
            root_nodes.append(current)
            processed_roots.add(current.node_id)
    
    main_print(f"Found {len(root_nodes)} root nodes (for {len(leaf_nodes)} leaf nodes)")
    
    # 🔧 添加调试信息验证batch_idx分配
    if len(leaf_nodes) > 0:
        batch_idx_counts = {}
        for leaf in leaf_nodes:
            batch_idx = leaf.batch_idx
            batch_idx_counts[batch_idx] = batch_idx_counts.get(batch_idx, 0) + 1
        main_print(f"Leaf nodes by batch_idx: {batch_idx_counts}")
        
        if args.use_group:
            expected_per_batch = args.num_generations
            for batch_idx, count in batch_idx_counts.items():
                if count != expected_per_batch:
                    main_print(f"Warning: batch_idx {batch_idx} has {count} leaf nodes, expected {expected_per_batch}")
    
    if args.use_group:
        # 🔧 修复：按batch_idx对leaf_nodes排序，确保正确的分组
        # 首先按batch_idx排序叶子节点
        leaf_nodes_sorted = sorted(leaf_nodes, key=lambda node: node.batch_idx)
        rewards_sorted = []
        
        # 重新排序rewards以匹配排序后的叶子节点
        batch_leaf_mapping = {}  # batch_idx -> list of (original_idx, leaf_node)
        for i, leaf in enumerate(leaf_nodes):
            batch_idx = leaf.batch_idx
            if batch_idx not in batch_leaf_mapping:
                batch_leaf_mapping[batch_idx] = []
            batch_leaf_mapping[batch_idx].append((i, leaf))
        
        # 按batch_idx顺序重构rewards
        for batch_idx in sorted(batch_leaf_mapping.keys()):
            batch_leaves = batch_leaf_mapping[batch_idx]
            for original_idx, leaf in batch_leaves:
                rewards_sorted.append(rewards[original_idx])
        rewards_sorted = torch.stack(rewards_sorted)
        
        # 验证排序正确性
        main_print("Verifying reward-batch_idx correspondence after sorting:")
        for i in range(min(10, len(leaf_nodes_sorted))):  # 只打印前10个样本
            leaf = leaf_nodes_sorted[i]
            reward = rewards_sorted[i].item() if hasattr(rewards_sorted[i], 'item') else rewards_sorted[i]
            main_print(f"  Sample {i}: batch_idx={leaf.batch_idx}, reward={reward:.4f}")
        
        # 按组计算advantage，类似原始GRPO
        n = len(rewards_sorted) // args.num_generations
        leaf_advantages = torch.zeros_like(rewards_sorted)
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = rewards_sorted[start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            leaf_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        # 为叶子节点分配group计算的advantage
        for i, leaf_node in enumerate(leaf_nodes_sorted):
            leaf_node.advantage = leaf_advantages[i]
            
        # use_group模式下，构建节点层次（处理所有根节点）
        all_nodes = []
        def collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        
        for root_node in root_nodes:
            collect_nodes(root_node)
        
        # 构建node_advantages字典
        node_advantages = {}
        for node in all_nodes:
            if hasattr(node, 'advantage'):
                node_advantages[node.node_id] = node.advantage
            else:
                node_advantages[node.node_id] = torch.tensor(0.0)
                
        main_print(f"使用use_group模式计算advantage: {n}组，每组{args.num_generations}个生成")
    else:
        # 原始的分层advantage计算（每棵树单独处理，避免跨样本信息泄露）
        all_nodes = []
        node_advantages = {}

        # 🔧 修复：按batch_idx组织leaf_nodes和rewards的对应关系
        batch_leaf_mapping = {}  # batch_idx -> list of (leaf_node, reward)
        for i, leaf in enumerate(leaf_nodes):
            batch_idx = leaf.batch_idx
            if batch_idx not in batch_leaf_mapping:
                batch_leaf_mapping[batch_idx] = []
            batch_leaf_mapping[batch_idx].append((leaf, rewards[i]))

        max_depth_across_trees = 0

        # 为每个根节点正确分配对应的叶子节点rewards，并分别计算advantage
        for root_node in root_nodes:
            # 获取当前根节点的所有叶子后代
            root_leaf_descendants = root_node.get_all_leaf_descendants()
            root_batch_idx = root_node.batch_idx

            # 🔧 从batch_leaf_mapping获取对应的rewards
            if root_batch_idx in batch_leaf_mapping:
                batch_leaf_rewards = batch_leaf_mapping[root_batch_idx]
                # 确保叶子节点顺序匹配
                root_rewards = []
                for leaf_desc in root_leaf_descendants:
                    for leaf_node, reward in batch_leaf_rewards:
                        if leaf_desc.node_id == leaf_node.node_id:
                            root_rewards.append(reward)
                            break
                root_rewards = torch.stack(root_rewards) if len(root_rewards) > 0 else torch.zeros(len(root_leaf_descendants), device=rewards.device)
            else:
                main_print(f"Warning: No rewards found for batch_idx {root_batch_idx}")
                root_rewards = torch.zeros(len(root_leaf_descendants), device=rewards.device)

            # 2. 计算当前树的所有节点的reward（自底向上聚合）
            tree_nodes, tree_nodes_by_depth = compute_node_rewards_from_leaves(
                root_node,
                root_rewards,
                root_leaf_descendants,
                use_prob_weighted=getattr(args, "tree_prob_weighted", False),
            )
            all_nodes.extend(tree_nodes)

            # 3. 按深度分层计算advantage（仅在该树内部归一化）
            tree_node_advantages = compute_hierarchical_advantages_by_depth(tree_nodes_by_depth)
            node_advantages.update(tree_node_advantages)

            # 记录该树的最大深度
            if len(tree_nodes_by_depth) > 0:
                max_depth_across_trees = max(max_depth_across_trees, max(tree_nodes_by_depth.keys()))

        main_print(f"Tree depth (max across trees): {max_depth_across_trees}")
    
    main_print(f"Tree training: {len(leaf_nodes)} leaf nodes, {len(all_nodes)} total nodes")
    main_print(f"Leaf rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    

    
    # 🌟 统一的树形训练逻辑 - 修改为与原始GRPO一致的梯度累积
    # 现在所有转移都是树中的父子关系，包括分裂和非分裂步骤
    
    training_samples = []
    
    # 收集所有树转移（现在包括分裂转移和连续转移）
    def collect_all_transitions(node):
        for child in node.children:
            # 判断转移类型：多个子节点=分裂，单个子节点=连续
            transition_type = "split" if len(node.children) > 1 else "sequential"
            
            child_advantage = node_advantages[child.node_id]
            sample = {
                "latent": node.latent,                     # 父节点状态
                "next_latent": child.latent,               # 子节点状态  
                "log_prob": child.log_prob,                # 转移的log_prob
                "advantage": child_advantage,              # 子节点的advantage
                "step": node.step,                         # 父节点的时间步
                "batch_idx": node.batch_idx,               # 🔧 添加batch_idx信息
                "parent_id": node.node_id,                 # 用于调试
                "child_id": child.node_id,                 # 用于调试
                "child_depth": child.depth,                # 用于调试
                "transition_type": transition_type,        # 转移类型
                "is_sde_edge": getattr(child, "is_sde_edge", transition_type=="split"),
            }
            training_samples.append(sample)
            
            # 递归处理子节点
            collect_all_transitions(child)
    
    # 🔧 修复：对所有根节点调用collect_all_transitions
    for root_node in root_nodes:
        collect_all_transitions(root_node)
    
    # 验证树结构的完整性
    total_nodes = len(all_nodes)
    leaf_count = len([node for node in all_nodes if node.is_leaf()])
    main_print(f"树结构验证: 总节点数={total_nodes}, 叶子节点数={leaf_count}")

    # 🌗 混合模式：仅训练SDE边（分裂边始终SDE；若启用滑动窗口，窗口内非分裂边也作为SDE）
    if getattr(args, 'mix_ode_sde_tree', False):
        before = len(training_samples)
        # 基于窗口与分裂点，按 step 再次确定应保留的 SDE 步
        total_steps = int(args.sampling_steps)
        stride = max(1, int(getattr(args, 'depth_pruning_slide_interval', 1)))
        window_size = max(1, int(getattr(args, 'mix_sde_window_size', 4)))
        sde_steps = set()
        # 窗口块：从 0, stride, 2*stride, ... 开始，各自覆盖 window_size 步
        start_step = 0
        while start_step < total_steps:
            end_step = min(total_steps - 1, start_step + window_size - 1)
            for s in range(start_step, end_step + 1):
                sde_steps.add(s)
            start_step += stride
        # 分裂点加入 SDE
        split_points = parse_split_points(args, total_steps)
        for sp in split_points:
            sde_steps.add(sp)
        # 过滤：仅保留父步在 sde_steps 的边
        training_samples = [s for s in training_samples if (s.get("step", -1) in sde_steps)]
        # 统计分布
        split_cnt = sum(1 for s in training_samples if s["transition_type"] == "split")
        seq_cnt = len(training_samples) - split_cnt
        main_print(f"混合模式启用：仅使用SDE边训练 {before} -> {len(training_samples)} (split={split_cnt}, sequential={seq_cnt}), sde_steps={sorted(list(sde_steps))}")
    
    # 🕐 步骤裁剪控制：判断当前步骤是否应该进行裁剪
    should_prune = True
    if current_step is not None and max_steps is not None and hasattr(args, 'pruning_step_ratio'):
        pruning_step_ratio = args.pruning_step_ratio
        pruning_cutoff_step = int(max_steps * pruning_step_ratio)
        should_prune = current_step <= pruning_cutoff_step
        main_print(f"🕐 裁剪步骤控制: 当前步骤 {current_step}/{max_steps}, 裁剪截止步骤 {pruning_cutoff_step}, 是否裁剪: {should_prune}")
    else:
        main_print(f"🕐 裁剪步骤控制: 未设置步骤信息，默认进行裁剪")
    
    # 🌿 深度裁剪逻辑：支持滑动窗口与固定窗口
    original_sample_count = len(training_samples)
    if should_prune and hasattr(args, 'depth_pruning') and args.depth_pruning:
        try:
            base_depths = [int(d.strip()) for d in args.depth_pruning.split(',') if d.strip()]
            base_depths = sorted(base_depths)

            active_pruning_depths = base_depths

            # 可选：滑动窗口
            if getattr(args, 'depth_pruning_slide', False):
                # 自动推断“停止滑动深度”：最后一次分裂父节点深度 = max(split_points)
                sampling_steps = args.sampling_steps
                split_points = parse_split_points(args, sampling_steps)
                auto_stop_depth = max(split_points) if len(split_points) > 0 else 0
                stop_depth = auto_stop_depth
                if hasattr(args, 'depth_pruning_stop_depth') and args.depth_pruning_stop_depth is not None:
                    stop_depth = args.depth_pruning_stop_depth

                interval = max(1, int(getattr(args, 'depth_pruning_slide_interval', 1)))
                # 在 t 个训练 step 后滑动一次：shift = current_step // interval
                shift_now = max(0, int(current_step // interval))
                # 限制最大滑动次数：窗口最浅层不能高于 stop_depth
                max_shift = max(0, base_depths[0] - stop_depth)
                shift_now = min(shift_now, max_shift)

                active_pruning_depths = [d - shift_now for d in base_depths]
                main_print(
                    f"🌿 深度裁剪(滑动窗口): step={current_step}, interval={interval}, shift={shift_now}, "
                    f"window {base_depths} -> {active_pruning_depths}, stop_at_depth={stop_depth}"
                )

            pruning_depths = set(active_pruning_depths)

            if pruning_depths:
                # 过滤掉指定深度的训练样本
                filtered_samples = []
                pruned_count = 0
                for sample in training_samples:
                    child_depth = sample["child_depth"]
                    if child_depth not in pruning_depths:
                        filtered_samples.append(sample)
                    else:
                        pruned_count += 1

                training_samples = filtered_samples
                main_print(f"🌿 深度裁剪: 裁剪深度 {sorted(pruning_depths)}")
                main_print(f"   裁剪前样本数: {original_sample_count}")
                main_print(f"   裁剪后样本数: {len(training_samples)}")
                main_print(f"   裁剪样本数: {pruned_count}")
                main_print(f"   裁剪比例: {pruned_count/original_sample_count*100:.1f}%")
        except ValueError as e:
            main_print(f"Warning: 深度裁剪参数解析失败: {e}")
    else:
        main_print(f"训练样本总数: {original_sample_count} (无深度裁剪)")
    
    # 🌳 宽度裁剪逻辑：在深度裁剪后进行，保留指定比例的训练样本
    samples_after_depth_pruning = len(training_samples)
    if should_prune and hasattr(args, 'width_pruning_mode') and args.width_pruning_mode is not None and args.width_pruning_mode > 0:
        try:
            width_pruning_ratio = getattr(args, 'width_pruning_ratio', 0.5)  # 默认保留50%
            mode = args.width_pruning_mode
            
            # 🔍 首先找到最后一次分裂的步骤
            # 重新计算total_steps（与采样时一致）
            sampling_steps = args.sampling_steps
            split_points = parse_split_points(args, sampling_steps)
            if not split_points:
                main_print("Warning: 没有分裂点，跳过宽度裁剪")
            else:
                last_split_step = max(split_points)
                main_print(f"🔍 最后一次分裂步骤: {last_split_step}")
                
                # 🎯 识别最后一层父节点的后续转移（step > last_split_step的转移）
                last_layer_samples = []
                other_samples = []
                
                for sample in training_samples:
                    if sample["step"] > last_split_step:
                        last_layer_samples.append(sample)
                    else:
                        other_samples.append(sample)
                
                main_print(f"🎯 最后一层转移样本数: {len(last_layer_samples)}")
                main_print(f"🎯 其他层转移样本数: {len(other_samples)}")
                
                if len(last_layer_samples) == 0:
                    main_print("Warning: 没有最后一层转移样本，跳过宽度裁剪")
                    total_pruned = 0
                else:
                    if mode == 1:
                        # 方式1：按最后分裂产生的分支分组，保留每个分支最好的后续转移
                        main_print(f"🌳 宽度裁剪模式1: 保留每个最后分裂分支最好的{width_pruning_ratio*100:.0f}%后续转移")
                        
                        # 🔍 按最后分裂产生的分支分组
                        # sample["step"]是父节点的步骤，sample["parent_id"]是父节点的ID
                        
                        def find_last_split_branch(sample, all_nodes_dict):
                            """追溯样本到最后分裂步骤的分支"""
                            current_node_id = sample["parent_id"]
                            target_step = last_split_step + 1  # 最后分裂产生的子节点的步骤
                            
                            # 如果父节点就是最后分裂产生的节点，直接返回
                            if sample["step"] == target_step:
                                return current_node_id
                            
                            # 否则向上追溯
                            while current_node_id in all_nodes_dict:
                                current_node = all_nodes_dict[current_node_id]
                                if current_node.step == target_step:
                                    return current_node_id
                                if current_node.parent is not None:
                                    current_node_id = current_node.parent.node_id
                                else:
                                    break
                            return f"unknown_{current_node_id}"  # 如果追溯失败，返回标记
                        
                        # 构建所有节点的字典以便快速查找
                        all_nodes_dict = {}
                        for node in all_nodes:
                            all_nodes_dict[node.node_id] = node
                        
                        # 🔍 调试：统计最后层样本的step分布
                        step_distribution = {}
                        for sample in last_layer_samples:
                            step = sample["step"]
                            step_distribution[step] = step_distribution.get(step, 0) + 1
                        main_print(f"🔍 最后层样本step分布: {step_distribution}")
                        
                        # 按最后分裂分支分组
                        branch_groups = {}
                        unknown_count = 0
                        for sample in last_layer_samples:
                            branch_id = find_last_split_branch(sample, all_nodes_dict)
                            if branch_id.startswith("unknown_"):
                                unknown_count += 1
                            if branch_id not in branch_groups:
                                branch_groups[branch_id] = []
                            branch_groups[branch_id].append(sample)
                        
                        if unknown_count > 0:
                            main_print(f"⚠️  无法追溯的样本数: {unknown_count}")
                        
                        # 🔍 调试：显示前几个分支组的信息
                        for i, (branch_id, group) in enumerate(branch_groups.items()):
                            if i < 5:  # 只显示前5个
                                main_print(f"🔍 分支 {branch_id}: {len(group)}个样本")
                        
                        main_print(f"🔍 最后分裂分支数: {len(branch_groups)}")
                        
                        # 对每个分支组内按advantage排序，保留前width_pruning_ratio
                        filtered_last_layer = []
                        total_pruned = 0
                        for branch_id, group in branch_groups.items():
                            group_size = len(group)
                            keep_count = max(1, int(group_size * width_pruning_ratio))  # 至少保留1个
                            
                            # 按advantage降序排序（最好的在前面）
                            sorted_group = sorted(group, key=lambda x: x["advantage"].item() if isinstance(x["advantage"], torch.Tensor) else float(x["advantage"]), reverse=True)
                            
                            kept_samples = sorted_group[:keep_count]
                            filtered_last_layer.extend(kept_samples)
                            total_pruned += (group_size - keep_count)
                        
                        # 组合其他层的样本和筛选后的最后一层样本
                        training_samples = other_samples + filtered_last_layer
                        main_print(f"   按最后分裂分支组裁剪: {len(branch_groups)}个分支组")
                        main_print(f"   平均每分支样本数: {len(last_layer_samples)/len(branch_groups):.1f}")
                        
                    elif mode == 2:
                        main_print(f"🌳 宽度裁剪模式2: 在最后层转移中保留最好和最坏各{width_pruning_ratio/2*100:.0f}%样本")
                        
                        sorted_last_layer = sorted(last_layer_samples, key=lambda x: x["advantage"].item() if isinstance(x["advantage"], torch.Tensor) else float(x["advantage"]), reverse=True)
                        
                        total_last_layer = len(sorted_last_layer)
                        keep_count = max(2, int(total_last_layer * width_pruning_ratio))  
                        
                        # 计算最好和最坏各保留多少
                        best_count = keep_count // 2
                        worst_count = keep_count - best_count
                        
                        # 保留最好的和最坏的
                        best_samples = sorted_last_layer[:best_count]
                        worst_samples = sorted_last_layer[-worst_count:] if worst_count > 0 else []
                        
                        filtered_last_layer = best_samples + worst_samples
                        total_pruned = total_last_layer - len(filtered_last_layer)
                        
                        # 组合其他层的样本和筛选后的最后一层样本
                        training_samples = other_samples + filtered_last_layer
                        main_print(f"   保留最后层最好样本: {best_count}个, 最坏样本: {worst_count}个")
                        
                    else:
                        main_print(f"Warning: 未知的宽度裁剪模式: {mode}, 跳过宽度裁剪")
                        total_pruned = 0
                
                if mode in [1, 2]:
                    main_print(f"   深度裁剪后样本数: {samples_after_depth_pruning}")
                    main_print(f"   宽度裁剪后样本数: {len(training_samples)}")
                    main_print(f"   宽度裁剪样本数: {total_pruned}")
                    main_print(f"   宽度裁剪比例: {total_pruned/len(last_layer_samples)*100:.1f}% (仅针对最后层转移)")
                    main_print(f"   总裁剪比例: {(original_sample_count-len(training_samples))/original_sample_count*100:.1f}%")
            
        except Exception as e:
            main_print(f"Warning: 宽度裁剪执行失败: {e}")
    else:
        if should_prune:
            main_print(f"无宽度裁剪，训练样本数: {len(training_samples)}")
        else:
            main_print(f"🕐 当前步骤超出裁剪范围，跳过所有裁剪，训练样本数: {len(training_samples)}")
    
    # 随机打乱训练样本（类似原始 GRPO）
    import random
    random.shuffle(training_samples)
    
    # 🌟 关键修改：使用与原始GRPO相同的梯度累积逻辑
    total_log_loss = 0.0
    grad_norm = None
    
    # 🕒 时间记录：损失计算和反向传播开始
    backward_start_time = time.time()
    
    # 准备图像ID（对所有样本使用相同的配置）
    latent_h, latent_w = args.h // 8, args.w // 8  # VAE下采样系数是8
    image_ids = prepare_latent_image_ids(
        1,  # 每次处理一个样本
        latent_h // 2,  # pack后的高度
        latent_w // 2,  # pack后的宽度
        device, 
        torch.bfloat16
    )
    
    # 计算每棵树的平均转移数用于loss归一化，避免batch变大导致有效学习率下降
    num_trees = max(1, len(root_nodes))
    avg_samples_per_tree = max(1, len(training_samples) // num_trees)
    num_samples_per_step = avg_samples_per_tree
    for i, sample in enumerate(training_samples):
        # 4. 为当前样本计算新的对数概率
        single_latent = sample["latent"]
        single_next_latent = sample["next_latent"]
        single_step = sample["step"]
        sample_batch_idx = sample["batch_idx"]  # 🔧 获取批次索引
        single_timestep = torch.tensor([int(sigma_schedule[single_step] * 1000)], 
                                     device=device, dtype=torch.long)
        
        # 🔧 根据batch_idx提取对应的参数，确保维度匹配
        sample_encoder_hidden_states = encoder_hidden_states[sample_batch_idx:sample_batch_idx+1]
        sample_pooled_prompt_embeds = pooled_prompt_embeds[sample_batch_idx:sample_batch_idx+1]
        sample_text_ids = text_ids[sample_batch_idx:sample_batch_idx+1]
        
        
        # 5. 计算新的对数概率
        new_log_prob = grpo_one_step(
            args,
            single_latent,
            single_next_latent,
            sample_encoder_hidden_states,  # 现在是[1, text_seq_len, text_channels]
            sample_pooled_prompt_embeds,   # 现在是[1, ...]
            sample_text_ids,               # 现在是[1, ...]
            image_ids,                     # image_ids通常是共享的
            transformer,
            single_timestep,
            single_step,
            sigma_schedule,
        )
        
        # 6. 计算重要性采样比率和 PPO clipped loss
        clip_range = args.clip_range
        adv_clip_max = args.adv_clip_max
        
        # 获取advantage并确保正确的形状
        adv = sample["advantage"]
        if adv.dim() > 0:
            advantage = adv.item()  # 转为标量
        else:
            advantage = adv.item()
        advantage = torch.tensor(advantage, device=device)
        
        # 对优势进行剪切
        clipped_advantage = torch.clamp(advantage, -adv_clip_max, adv_clip_max)
        
        # 计算比率
        old_log_prob = sample["log_prob"]
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        # PPO clipped loss
        unclipped_loss = -clipped_advantage * ratio
        clipped_loss = -clipped_advantage * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        
        # 数值稳定性检查
        if torch.isnan(ratio).any() or torch.isinf(ratio).any():
            print(f"Warning: ratio contains NaN or Inf at sample {i}")
            print(f"  new_log_prob: {new_log_prob}")
            print(f"  old_log_prob: {old_log_prob}")
            continue  # 跳过这个样本
            
        if torch.isnan(clipped_advantage).any() or torch.isinf(clipped_advantage).any():
            print(f"Warning: advantage contains NaN or Inf at sample {i}")
            continue  # 跳过这个样本
        
        # 🌟 关键修复：使用与原始GRPO完全相同的loss计算
        # 原始GRPO使用: loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)
        # 这里我们用总样本数替代train_timesteps
        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * num_samples_per_step)
        
        # 检查 loss 是否为 NaN 或 Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: loss is NaN or Inf at sample {i}, skipping")
            continue
        
        # 反向传播
        loss.backward()
        
        # 累积 loss 用于日志记录
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_log_loss += avg_loss.item()
        
        # 🌟 关键修改：只有当累积到gradient_accumulation_steps时才执行optimizer.step()
        # 这与原始GRPO的逻辑完全一致: if (i+1)%args.gradient_accumulation_steps==0:
        if (i+1) % args.gradient_accumulation_steps == 0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        # 调试输出（每隔一定步数）
        if dist.get_rank() % 8 == 0 and i % 10 == 0:
            print(f"TreeGRPO training - sample {i}/{len(training_samples)}, ratio: {ratio.mean().item():.4f}, adv: {advantage.item():.4f}, loss: {loss.item():.4f}")
        
    # 如果最后还有未完成的梯度累积，执行最后一次更新
    if len(training_samples) % args.gradient_accumulation_steps != 0:
        if grad_norm is None:  # 如果还没有执行过clip_grad_norm_
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    # 如果grad_norm还是None，说明没有有效的训练样本
    if grad_norm is None:
        grad_norm = torch.tensor(0.0)
    
    # 🕒 时间记录：损失计算和反向传播结束
    backward_end_time = time.time()
    
    # 🕒 计算各个环节的时间
    sample_time = sample_end_time - sample_start_time
    reward_time = reward_end_time - reward_start_time
    backward_time = backward_end_time - backward_start_time
    total_step_time = backward_end_time - step_start_time
    
    # 🕒 打印时间统计信息
    if dist.get_rank() == 0:
        main_print(f"⏱️  训练时间统计:")
        main_print(f"   采样时间: {sample_time:.2f}s ({sample_time/total_step_time*100:.1f}%)")
        main_print(f"   奖励计算: {reward_time:.2f}s ({reward_time/total_step_time*100:.1f}%)")
        main_print(f"   损失反向传播: {backward_time:.2f}s ({backward_time/total_step_time*100:.1f}%)")
        main_print(f"   总时间: {total_step_time:.2f}s")
    
    # 🌟 修复loss日志记录：返回平均loss而非累积loss，更准确反映训练状态
    # 计算实际处理的样本数（排除跳过的NaN样本）
    effective_samples = len(training_samples)  # 简化：假设大部分样本都是有效的
    if effective_samples > 0:
        total_loss = total_log_loss / effective_samples  # 平均loss
    else:
        total_loss = 0.0
    
    # 准备返回的奖励统计信息
    reward_stats = {
        "mean": gathered_reward.mean().item() if gathered_reward.numel() > 0 else 0.0,
        "std": gathered_reward.std().item() if gathered_reward.numel() > 0 else 0.0,
        "min": gathered_reward.min().item() if gathered_reward.numel() > 0 else 0.0,
        "max": gathered_reward.max().item() if gathered_reward.numel() > 0 else 0.0,
    }
    
    return total_loss, grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, reward_stats


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        set_seed(args.seed + rank)

    # 生成实验名并保存到args中
    exp_name = generate_experiment_name(args)
    args._exp_name = exp_name  # 保存实验名到args中，供其他函数使用
    
    # 保存实验配置
    save_experiment_config(args, exp_name, rank)
    
    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 为树形 rollout 创建实验特定的目录
    if rank <= 0:
        os.makedirs(f"images_branchgrpo/{exp_name}", exist_ok=True)
        os.makedirs(f"checkpoints/{exp_name}", exist_ok=True)
        os.makedirs(f"tmp/{exp_name}", exist_ok=True)

    # 初始化奖励模型
    preprocess_val = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()

    if args.use_pickscore:
        from transformers import AutoProcessor, AutoModel
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        reward_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    
    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )
    
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs,)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    main_print(f"--> model loaded")

    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if rank <= 0:
        project = "flux"
        wandb.init(project=project, name=exp_name, id=exp_name, config=args, resume="allow")

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running Tree-based GRPO training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                checkpoint_dir = f"checkpoints/{exp_name}"
                save_checkpoint(transformer, rank, checkpoint_dir, step, epoch)
                dist.barrier()
            
            loss, grad_norm, reward_stats = train_one_step_tree(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
                current_step=step,
                max_steps=args.max_train_steps,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "reward_mean": f"{reward_stats['mean']:.3f}",
                    "reward_std": f"{reward_stats['std']:.3f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                        "reward_mean": reward_stats["mean"],
                        "reward_std": reward_stats["std"],
                        "reward_min": reward_stats["min"],
                        "reward_max": reward_stats["max"],
                    },
                    step=step,
                )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training parameters
    parser.add_argument("--h", type=int, default=None, help="video height")
    parser.add_argument("--w", type=int, default=None, help="video width")
    parser.add_argument("--t", type=int, default=None, help="video length")
    parser.add_argument("--sampling_steps", type=int, default=None, help="sampling steps")
    parser.add_argument("--eta", type=float, default=None, help="noise eta")
    parser.add_argument("--sampler_seed", type=int, default=None, help="seed of sampler")
    parser.add_argument("--loss_coef", type=float, default=1.0, help="the global loss should be divided by")
    parser.add_argument("--use_group", action="store_true", default=False, help="whether compute advantages for each prompt")
    parser.add_argument("--num_generations", type=int, default=16, help="num_generations per prompt")
    parser.add_argument("--use_hpsv2", action="store_true", default=False, help="whether use hpsv2 as reward model")
    parser.add_argument("--use_pickscore", action="store_true", default=False, help="whether use pickscore as reward model")
    parser.add_argument("--ignore_last", action="store_true", default=False, help="whether ignore last step of mdp")
    parser.add_argument("--init_same_noise", action="store_true", default=False, help="whether use the same noise within each prompt")
    parser.add_argument("--shift", type=float, default=1.0, help="shift for timestep scheduler")
    parser.add_argument("--timestep_fraction", type=float, default=1.0, help="timestep downsample ratio")
    parser.add_argument("--clip_range", type=float, default=1e-4, help="clip range for grpo")
    parser.add_argument("--adv_clip_max", type=float, default=5.0, help="clipping advantage")
    
    # Tree-specific parameters
    parser.add_argument("--tree_split_rounds", type=int, default=4, help="Number of split rounds for tree rollout")
    parser.add_argument("--tree_split_points", type=str, default=None, help="Comma-separated list of split points (e.g., '10,14,17,19'). If provided, overrides tree_split_rounds")
    parser.add_argument("--tree_split_noise_scale", type=float, default=0.3, help="Noise scale for tree splits")
    parser.add_argument("--tree_prob_weighted", action="store_true", default=False, help="Use child-edge log_prob softmax weighting for internal node rewards")
    parser.add_argument("--depth_pruning", type=str, default=None, help="Comma-separated list of depths to prune from training (e.g., '15,16,17,18'). Sampling and reward calculation remain unchanged.")
    parser.add_argument("--width_pruning_mode", type=int, default=0, choices=[0, 1, 2], help="Width pruning mode: 0=no width pruning, 1=keep best from each parent, 2=keep best and worst globally")
    parser.add_argument("--width_pruning_ratio", type=float, default=0.5, help="Ratio of samples to keep after width pruning (default: 0.5)")
    parser.add_argument("--pruning_step_ratio", type=float, default=1.0, help="Ratio of training steps where pruning is applied (0.5 = pruning in first 50%% of steps, default: 1.0 = always prune)")

    # Depth pruning sliding window
    parser.add_argument("--depth_pruning_slide", action="store_true", default=False, help="Enable sliding window for depth pruning")
    parser.add_argument("--depth_pruning_slide_interval", type=int, default=1, help="Slide the depth pruning window every N training steps")
    parser.add_argument("--depth_pruning_stop_depth", type=int, default=None, help="Optional: stop sliding when the shallowest depth reaches this value; if None, use last split parent depth")

    # 树形混合 ODE/SDE：窗口内使用SDE，窗口外使用ODE；分裂步始终SDE
    parser.add_argument("--mix_ode_sde_tree", action="store_true", default=False, help="Enable mixed ODE/SDE on tree rollout: SDE inside a sliding window; ODE outside; split steps are always SDE")
    parser.add_argument("--mix_sde_window_size", type=int, default=4, help="Sliding window size (in steps) for SDE in tree rollout")

    args = parser.parse_args()
    main(args) 