#!/usr/bin/env bash
# Antigravity Cluster Provisioning Script for GPU Training Düğümü
# Targets: Ubuntu 22.04 LTS / 24.04 LTS clean installation.
# Sets up CUDA Drivers, Docker, Rust compiler, Python virtual environments, and verifies setup.

set -euo pipefail

echo "=========================================================="
echo "      STARTING SYSTEM PROVISIONING FOR TRAINING NODE"
echo "=========================================================="

# 1. Update and install standard Unix tools
echo -e "\n[STEP 1] Updating apt repositories and installing utilities..."
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    software-properties-common \
    sqlite3 \
    libsqlite3-dev

# 2. Install NVIDIA CUDA Drivers and Toolkit (if NVIDIA GPU detected)
echo -e "\n[STEP 2] Checking for NVIDIA GPU hardware..."
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA drivers..."
    # Add proprietary GPU driver repositories
    sudo add-apt-repository ppa:graphics-drivers/ppa -y
    sudo apt-get update -y
    
    # Install headless CUDA driver and standard toolkit
    sudo apt-get install -y nvidia-driver-535-server nvidia-utils-535-server cuda-toolkit-12-2
    echo "NVIDIA GPU drivers and CUDA toolkit 12.2 installed."
else
    echo "No NVIDIA GPU detected. Skipping CUDA graphics drivers installation."
fi

# 3. Install Docker and Docker Compose
echo -e "\n[STEP 3] Installing Docker Runtime & Compose..."
if ! command -v docker &> /dev/null; then
    sudo curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker "$USER"
    rm get-docker.sh
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# 4. Install Rust Compiler toolchain
echo -e "\n[STEP 4] Installing Rust Toolchain (rustup)..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "Rust compiler toolchain installed: $(rustc --version)"
else
    echo "Rust is already installed: $(rustc --version)"
fi

# 5. Install Python Venv and Pip dependencies
echo -e "\n[STEP 5] Installing Python 3 virtualenv..."
sudo apt-get install -y python3-pip python3-venv

# 6. Verify Workspace setup
echo -e "\n[STEP 6] Workspace environment verification..."
echo "System specs:"
uname -a
lscpu | grep "Model name"
free -h

echo "=========================================================="
echo "  PROVISIONING COMPLETED SUCCESSFULLY! REBOOT RECOMMENDED"
echo "=========================================================="
