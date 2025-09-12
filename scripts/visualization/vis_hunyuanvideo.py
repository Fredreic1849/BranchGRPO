import argparse
import math
import os
import re
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video, check_min_version
from tqdm.auto import tqdm
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from accelerate.utils import set_seed

# --- Custom Module Imports (assuming they exist in the user's environment) ---
# These imports are based on the original script and must be in the user's Python environment.
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, destroy_sequence_parallel_group
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs
from fastvideo.utils.load import load_transformer, load_vae
from fastvideo.dataset.latent_rl_datasets import LatentDataset, latent_collate_function
from fastvideo.utils.logging_ import main_print

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.31.0")

# --- Core Diffusion Sampling Logic (Unchanged from original script) ---

def sd3_time_shift(shift, t):
    """Applies a time shift to the sigma schedule."""
    return (shift * t) / (1 + (shift - 1) * t)

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    sde_solver: bool,
):
    """Performs a single step of the FLUX/Rectified Flow sampling process."""
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    return prev_sample_mean, pred_original_sample

def assert_eq(x, y, msg=None):
    """A simple assertion helper for checking equality."""
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
    """
    Runs the main denoising loop for the sampling process.
    This function iterates through the timesteps and applies the model to denoise the latents.
    """
    for i in progress_bar:
        B = encoder_hidden_states.shape[0]
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        
        transformer.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            model_pred = transformer(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                guidance=torch.tensor([6018.0], device=z.device, dtype=torch.bfloat16),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        z, pred_original = flux_step(
            model_pred, z.to(torch.float32), args.eta, 
            sigmas=sigma_schedule, index=i, sde_solver=True
        )
        z = z.to(torch.bfloat16)

    # The final predicted original sample is scaled and returned.
    # The scaling factor is likely specific to the VAE used.
    latents = pred_original.to(torch.float32) / 0.476986
    return latents

def sanitize_filename(text, max_length=50):
    """
    Cleans a string to be used as a valid filename.
    It removes illegal characters and truncates it to a specified length.
    """
    # Truncate to max_length
    sanitized = text[:max_length]
    # Remove characters that are invalid in filenames on most OSes
    sanitized = re.sub(r'[\\/*?:"<>|]', "", sanitized)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove any trailing/leading whitespace or underscores
    sanitized = sanitized.strip('_')
    return sanitized

def generate_and_save_video(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    encoder_attention_mask,
    captions,
):
    """
    This function is adapted from the original `sample_reference_model`.
    It generates a video based on text embeddings and saves it locally.
    The training-specific logic like reward calculation has been removed.
    """
    # --- Setup Sampling Parameters ---
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    assert_eq(len(sigma_schedule), sample_steps + 1, "Sigma schedule length mismatch")

    # --- Prepare Latent Tensor and Batching ---
    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    # Process one video at a time to manage memory, even if the dataloader batch size is larger.
    batch_indices = torch.chunk(torch.arange(B), B)

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
        # Get the caption for the current item in the batch
        caption = captions[index] 

        # --- Generate Initial Latent Noise ---
        generator = torch.Generator(device=device)
        # Use a fixed seed for the first item, then increment for variety within a batch
        generator.manual_seed(args.seed + index)
        input_latents = torch.randn(
            (len(batch_idx), IN_CHANNELS, latent_t, latent_h, latent_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        # --- Run Denoising Process ---
        # The progress bar is only shown on the main process (rank 0)
        progress_bar = tqdm(range(0, sample_steps), desc=f"Sampling for '{caption[:30]}...'", disable=(dist.get_rank() != 0))
        with torch.no_grad():
            final_latents = run_sample_step(
                args,
                input_latents.clone(),
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_encoder_attention_mask,
            )
        
        # --- Decode, Save, and Visualize (only on rank 0) ---
        # This prevents multiple processes from trying to write the same file.
        main_print(f"Decoding latents for caption: {caption}")
        vae.enable_tiling()
        video_processor = VideoProcessor(vae_scale_factor=SPATIAL_DOWNSAMPLE)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video_tensor = vae.decode(final_latents, return_dict=False)[0]
            video_frames = video_processor.postprocess_video(video_tensor)

        # --- Create Filename from Caption and Save Video ---
        safe_filename = sanitize_filename(caption)
        output_path = os.path.join(args.output_dir, f"{safe_filename}.mp4")
        
        main_print(f"Exporting video to: {output_path}")
        export_to_video(video_frames[0], output_path, fps=args.fps)

def main(args):
    """
    Main entry point for the visualization script.
    It sets up the distributed environment, loads models, and iterates through
    the dataloader to generate and save videos.
    """
    # --- Distributed Setup ---
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    if args.sp_size > 1:
        initialize_sequence_parallel_state(args.sp_size)

    # --- Seed and Output Directory ---
    if args.seed is not None:
        set_seed(args.seed)
    if rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        main_print(f"Output directory created at: {args.output_dir}")

    # --- Model Loading ---
    main_print(f"--> Loading transformer model from {args.pretrained_model_name_or_path}")
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.bfloat16, # Use bfloat16 for inference
    )

    main_print("--> Loading VAE model...")
    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)
    
    # --- FSDP Initialization ---
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    fsdp_kwargs, _ = get_dit_fsdp_kwargs(
        transformer, args.fsdp_sharding_startegy, False, args.use_cpu_offload, "bf16"
    )
    transformer = FSDP(transformer, **fsdp_kwargs)
    main_print(f"--> FSDP model loaded on rank {rank}")

    # Set models to evaluation mode
    transformer.eval()
    vae.eval()

    # --- Data Loading ---
    main_print("--> Setting up dataset and dataloader...")
    train_dataset = LatentDataset(args.data_json_path, args.t, args.cfg)
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False, seed=args.seed
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Wrapper for sequence parallelism if used
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader, device, args.train_batch_size, args.sp_size, args.train_sp_batch_size
    )

    # --- Main Visualization Loop ---
    main_print("***** Starting Video Generation *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Batch size per device = {args.train_batch_size}")
    
    # Set a limit on the number of samples to visualize if provided
    num_samples_to_process = args.num_visualization_samples or len(loader)
    progress_bar = tqdm(
        range(num_samples_to_process),
        desc="Processing Batches",
        disable=(rank != 0),
    )

    for i, (encoder_hidden_states, encoder_attention_mask, captions) in enumerate(loader):
        if i >= num_samples_to_process:
            break
            
        main_print(f"\nProcessing batch {i+1}/{num_samples_to_process}...")
        
        generate_and_save_video(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states,
            encoder_attention_mask,
            captions,
        )
        
        # Barrier to ensure all processes finish the batch before starting the next one.
        dist.barrier()
        progress_bar.update(1)

    main_print("***** Visualization Complete *****")

    # --- Cleanup ---
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from a dataset for visualization.")
    
    # --- Essential Arguments for Visualization ---
    parser.add_argument("--data_json_path", type=str, required=True, help="Path to the dataset JSON file.")
    parser.add_argument("--output_dir", type=str, default="./visualization_output", help="Directory to save the generated videos.")
    parser.add_argument("--num_visualization_samples", type=int, default=None, help="Limit the number of videos to generate. Default is all.")

    # --- Model Paths ---
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="hunyuan_hf", help="The type of model to use.")

    # --- Video & Sampling Parameters ---
    parser.add_argument("--h", type=int, default=256, help="Video height.")
    parser.add_argument("--w", type=int, default=256, help="Video width.")
    parser.add_argument("--t", type=int, default=16, help="Video length in frames.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the saved video.")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Number of diffusion sampling steps.")
    parser.add_argument("--eta", type=float, default=0.3, help="Eta for the SDE solver.")
    parser.add_argument("--shift", type=float, default=1.0, help="Time shift value for sigma schedule.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible generation.")
    parser.add_argument("--cfg", type=float, default=0.0, help="Classifier-Free Guidance scale (used in dataset).")

    # --- Dataloader and Batching ---
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device. Recommend 1 for visualization to process one-by-one.")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    # --- Distributed & System Configuration ---
    parser.add_argument("--sp_size", type=int, default=1, help="Sequence Parallelism size.")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel processing.")
    parser.add_argument("--fsdp_sharding_startegy", default="full", help="FSDP sharding strategy.")
    parser.add_argument("--use_cpu_offload", action="store_true", help="Use CPU offload with FSDP.")

    # The rest of the arguments from the original script are omitted for clarity,
    # as they are related to training, logging, and checkpointing.
    
    args = parser.parse_args()
    main(args)