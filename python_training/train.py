import os
os.environ["USE_LIBUV"] = "0"
import sys
import argparse
import time
import torch
import torch._C._distributed_c10d as C_dist

# Monkey-patch C++ level TCPStore immediately to force use_libuv=False
orig_tcp_store = C_dist.TCPStore
def patched_tcp_store(*args, **kwargs):
    args_list = list(args)
    if len(args_list) >= 6:
        args_list[5] = False
    else:
        kwargs["use_libuv"] = False
    return orig_tcp_store(*args_list, **kwargs)
C_dist.TCPStore = patched_tcp_store

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Dynamic monkey-patch across all imported modules in sys.modules
for ns in list(sys.modules.keys()):
    if ns.startswith("torch.distributed") or "rendezvous" in ns:
        try:
            mod = sys.modules[ns]
            if hasattr(mod, "TCPStore"):
                setattr(mod, "TCPStore", patched_tcp_store)
        except Exception:
            pass
dist.TCPStore = patched_tcp_store

from model import MayaModel, MayaConfig
from dataset import ShardedMmapDataset
from db_logger import AsyncMetricLogger
from lua_config_parser import parse_lua_file
from muon import Muon

def parse_args():
    parser = argparse.ArgumentParser(description="Maya-1 1B Training Core")
    parser.add_argument("--data_path", type=str, required=True, help="Path to pre-tokenized binary shard file")
    parser.add_argument("--checkpoint_dir", type=str, default="shared/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to specific checkpoint to resume training from")
    parser.add_argument("--lua_config", type=str, default=None, help="Optional path to a Lua configuration file")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden size (2048 for 1B)")
    parser.add_argument("--num_hidden_layers", type=int, default=22, help="Number of layers (22 for 1B)")
    parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of heads (32 for 1B)")
    parser.add_argument("--num_key_value_heads", type=int, default=8, help="Number of KV heads (8 for GQA)")
    parser.add_argument("--intermediate_size", type=int, default=5632, help="SwiGLU intermediate size")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for graph acceleration")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Steps between checkpoints")
    parser.add_argument("--node_id", type=str, default="node-0", help="Identifier for the training node")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="Learning rate for Muon optimizer")
    parser.add_argument("--use_mtp", type=bool, default=True, help="Enable Multi-Token Prediction")
    parser.add_argument("--lambda_mtp", type=float, default=0.15, help="Auxiliary loss weight for MTP")
    parser.add_argument("--use_mup", action="store_true", help="Enable Maximal Update Parametrization (μP)")
    parser.add_argument("--mup_base_hidden", type=int, default=256, help="Hidden dimension of the proxy base model for μP")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Override arguments if Lua config file is provided
    if args.lua_config:
        if os.path.exists(args.lua_config):
            lua_cfg = parse_lua_file(args.lua_config)
            
            # Map model table
            if "model" in lua_cfg:
                m = lua_cfg["model"]
                if "--vocab_size" not in sys.argv:
                    args.vocab_size = m.get("vocab_size", args.vocab_size)
                if "--hidden_size" not in sys.argv:
                    args.hidden_size = m.get("hidden_size", args.hidden_size)
                if "--num_hidden_layers" not in sys.argv:
                    args.num_hidden_layers = m.get("num_hidden_layers", args.num_hidden_layers)
                if "--num_attention_heads" not in sys.argv:
                    args.num_attention_heads = m.get("num_attention_heads", args.num_attention_heads)
                if "--num_key_value_heads" not in sys.argv:
                    args.num_key_value_heads = m.get("num_key_value_heads", args.num_key_value_heads)
                if "--intermediate_size" not in sys.argv:
                    args.intermediate_size = m.get("intermediate_size", args.intermediate_size)
                if "--seq_len" not in sys.argv:
                    args.seq_len = m.get("seq_len", args.seq_len)
            
            # Map training table
            if "training" in lua_cfg:
                t = lua_cfg["training"]
                if "--batch_size" not in sys.argv:
                    args.batch_size = t.get("batch_size", args.batch_size)
                if "--lr" not in sys.argv:
                    args.lr = t.get("lr", args.lr)
                if "--max_steps" not in sys.argv:
                    args.max_steps = t.get("max_steps", args.max_steps)
                if "--compile" not in sys.argv:
                    args.compile = t.get("compile", args.compile)
                if "--checkpoint_interval" not in sys.argv:
                    args.checkpoint_interval = t.get("checkpoint_interval", args.checkpoint_interval)
        else:
            print(f"Warning: Lua config file not found at {args.lua_config}. Using CLI defaults.")
            
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Check if we are running in distributed mode via torchrun
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        # Initialize process group using NCCL for GPU, Gloo for CPU
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        if rank == 0:
            print(f"Distributed mode active. Process count: {world_size}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Single process run on device: {device.upper()}")
    
    # Setup mixed precision settings
    device_type = "cuda" if "cuda" in device else "cpu"
    # Choose bfloat16 if supported, else float16
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # If on CPU, fall back to float32 for fast CPU computation
    if device == "cpu":
        dtype = torch.float32
    
    if rank == 0:
        print(f"Using mixed precision data type: {dtype}")

    # Model Config
    config = MayaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.seq_len,
        use_mtp=args.use_mtp,
        lambda_mtp=args.lambda_mtp
    )
    
    # Instantiate Model
    if rank == 0:
        print("Initializing Maya-1 model...")
    model = MayaModel(config)
    
    if args.use_mup:
        import mup
        if rank == 0:
            print("Applying Maximal Update Parametrization (muP)...")
        # Instantiate a small base shape proxy model
        # Scale intermediate_size proportionally with hidden_size for mup scaling consistency
        base_intermediate_size = max(1, int(args.intermediate_size * args.mup_base_hidden / args.hidden_size))
        base_config = MayaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.mup_base_hidden,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            intermediate_size=base_intermediate_size,
            max_seq_len=args.seq_len,
            use_mtp=args.use_mtp,
            lambda_mtp=args.lambda_mtp
        )
        base_model = MayaModel(base_config)
        mup.set_base_shapes(model, base_model)
        
    if rank == 0:
        print(f"Total model parameters: {model.estimate_parameter_count():,}")
    
    model = model.to(device)

    # Wrap Model in DDP if running distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Compile Model if requested (only compile the internal module if using DDP)
    if args.compile:
        if rank == 0:
            print("Compiling model using torch.compile (mode='max-autotune')...")
        try:
            raw_model = model.module if is_distributed else model
            compiled_raw = torch.compile(raw_model, mode="max-autotune")
            if is_distributed:
                model = DDP(compiled_raw, device_ids=[local_rank] if torch.cuda.is_available() else None)
            else:
                model = compiled_raw
            if rank == 0:
                print("Model compilation completed successfully.")
        except Exception as e:
            if rank == 0:
                print(f"Warning: torch.compile failed with error: {e}. Falling back to uncompiled execution.")

    # Setup Optimizers (Muon for 2D hidden layers, AdamW for rest)
    raw_model = model.module if is_distributed else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    
    muon_params = []
    adamw_params = []
    
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok_embeddings" not in name and "output" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
            
    if rank == 0:
        print(f"Muon parameters: {len(muon_params)} tensors")
        print(f"AdamW parameters: {len(adamw_params)} tensors")
        
    use_fused = (device_type == "cuda")
    optimizers = []
    if len(muon_params) > 0:
        optimizers.append(Muon(muon_params, lr=args.muon_lr))
    if len(adamw_params) > 0:
        if args.use_mup:
            import mup
            if rank == 0:
                print("Using MuAdamW for AdamW parameters under muP.")
            optimizers.append(mup.MuAdamW(adamw_params, lr=args.lr, fused=use_fused))
        else:
            optimizers.append(torch.optim.AdamW(adamw_params, lr=args.lr, fused=use_fused))

    # Initialize Logger (Only Rank 0 writes to DB)
    logger = AsyncMetricLogger(node_id=args.node_id) if rank == 0 else None
    
    start_step = 0

    # Load from checkpoint if requested
    if args.resume_from:
        if os.path.exists(args.resume_from):
            if rank == 0:
                print(f"Resuming training from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            
            # Unwrap DDP prefix if needed
            first_key = list(state_dict.keys())[0]
            if first_key.startswith("_orig_mod.") and not args.compile:
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            elif not first_key.startswith("_orig_mod.") and args.compile:
                state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
                
            raw_model = model.module if is_distributed else model
            raw_model.load_state_dict(state_dict)
            
            # Load optimizer state if available
            if "optimizer_state_dicts" in checkpoint:
                for opt, state in zip(optimizers, checkpoint["optimizer_state_dicts"]):
                    opt.load_state_dict(state)
            elif "optimizer_state_dict" in checkpoint and len(optimizers) == 1:
                optimizers[0].load_state_dict(checkpoint["optimizer_state_dict"])
            
            start_step = checkpoint["step"]
            if rank == 0:
                print(f"Loaded checkpoint at step {start_step}")
        else:
            if rank == 0:
                print(f"Warning: Checkpoint not found at {args.resume_from}. Starting from scratch.")

    # Dataset & Loader (DistributedSampler partitions data among ranks)
    if rank == 0:
        print(f"Loading memory-mapped dataset from: {args.data_path}")
    dataset = ShardedMmapDataset(args.data_path, seq_len=args.seq_len)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=(sampler is None), 
        num_workers=0 if device_type == "cpu" else 2,
        pin_memory=(device_type == "cuda")
    )
    
    # Infinite iteration helper over the dataloader
    def get_batches():
        epoch = 0
        while True:
            if is_distributed and sampler is not None:
                sampler.set_epoch(epoch)
            for x, y in dataloader:
                yield x, y
            epoch += 1

    batch_iter = get_batches()
    model.train()
    
    step = start_step
    if rank == 0:
        print(f"Starting training loop from step {step} to {args.max_steps}...")
    
    # Enable TF32 for Matmul acceleration on compatible GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    
    while step < args.max_steps:
        step_start_time = time.time()
        
        # Load next batch of inputs (X) and targets (Y)
        x, y = next(batch_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        # Mixed Precision context
        with torch.autocast(device_type=device_type, dtype=dtype):
            logits, loss = model(x, y)
            
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        for opt in optimizers:
            opt.step()
        
        # Calculate wall-clock metrics
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        tokens_processed = x.numel()
        tokens_per_sec = tokens_processed / step_time
        
        loss_val = loss.item()
        
        # Log to SQLite and console (Rank 0 only)
        if rank == 0:
            logger.log(step, "loss", loss_val)
            logger.log(step, "tokens_per_sec", tokens_per_sec * world_size)  # Total throughput
            logger.log(step, "step_time_ms", step_time * 1000.0)
            logger.log(step, "lr", args.lr)
            
            # Log detailed NTP and MTP losses if available
            unwrapped_model = model.module if is_distributed else model
            if hasattr(unwrapped_model, "_orig_mod"):
                unwrapped_model = unwrapped_model._orig_mod
            if hasattr(unwrapped_model, "last_loss_dict") and unwrapped_model.last_loss_dict:
                for k, v in unwrapped_model.last_loss_dict.items():
                    logger.log(step, k, v)
            
            # Periodic printing
            if step % 10 == 0 or step == args.max_steps - 1:
                print(
                    f"Step {step} | Loss: {loss_val:.4f} | "
                    f"Speed: {tokens_per_sec * world_size:.1f} tok/s | "
                    f"Time: {step_time*1000:.1f}ms | Device: {device.upper()}"
                )
            
            # Periodic Checkpointing on Rank 0
            if (step > 0 and step % args.checkpoint_interval == 0) or step == args.max_steps - 1:
                ckpt_path = os.path.join(args.checkpoint_dir, "ckpt_latest.pt")
                print(f"Saving checkpoint to {ckpt_path} (optimized space)...")
                raw_model = model.module if is_distributed else model
                
                # Convert weights to bfloat16 to save space, and omit optimizer state (saves 10GB+ VRAM/disk)
                state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in raw_model.state_dict().items()}
                
                torch.save({
                    "step": step,
                    "model_state_dict": state_dict_bf16,
                    "loss": loss_val,
                }, ckpt_path)
            
        step += 1

    if rank == 0:
        total_time = time.time() - t0
        print(f"Training completed successfully in {total_time:.2f} seconds!")
        logger.close()
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
