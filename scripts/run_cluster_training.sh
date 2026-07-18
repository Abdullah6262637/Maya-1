#!/bin/bash
# Maya-1 AI System — 4x GPU Distributed DDP Production Training Launcher
# Designed to run directly on Linux (Ubuntu) Cloud GPU Instances (AWS, GCP, RunPod, etc.)

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Maya-1 Distributed Training Initializer ==="

# 1. Setup local environment variables for high-performance training
export OMP_NUM_THREADS=4
export PYTHONIOENCODING=utf-8
export TORCH_DISTRIBUTED_DEBUG=INFO

# Disable PyTorch Windows libuv socket engine (not needed on Linux but safe)
export USE_LIBUV=0

# 2. Check if virtual environment exists and activate
if [ -d "python_training/.venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source python_training/.venv/bin/activate
else
    echo "[WARNING] Virtual environment python_training/.venv not found. Using global python..."
fi

# 3. Compile custom PyTorch/CUDA extensions if GPU is available
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "[INFO] CUDA GPU detected. Compiling RMSNorm CUDA extension..."
    python3 mojo_kernels/rms_norm_extension.py
else
    echo "[INFO] No CUDA GPU found on this process node. Fallback CPU execution active."
fi

# 4. Start Distributed pre-training on 4 GPUs
# Runs with static rendezvous on localhost port 29500
echo "[LAUNCH] Starting DDP pre-training across 4 GPUs..."
torchrun \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    python_training/train.py \
    --data_path shared/mock_data.bin \
    --max_steps 50000 \
    --checkpoint_interval 5000 \
    --lua_config python_training/config.lua

echo "[SUCCESS] Training loop completed."
