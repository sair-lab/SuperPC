#!/bin/bash

# Comprehensive setup script for SuperPC environment
# This script creates the conda environment and builds all required CUDA extensions

set -e  # Exit on any error

echo "========================================="
echo "SuperPC Environment Setup"
echo "========================================="

# Resolve repository root from this script location so builds work from any checkout path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh

# Step 1: Create conda environment
echo ""
echo "[1/5] Creating conda environment 'superpc' with Python 3.9..."
conda create -n superpc python=3.9 -y

# Activate the environment
conda activate superpc

# Step 2: Install base packages
echo "[2/5] Installing base packages..."
pip install numpy==1.25.2 \
    open3d==0.17.0 \
    einops==0.3.2 \
    scikit-learn==1.3.1 \
    tqdm==4.62.3 \
    h5py==3.6.0 \
    plyfile==1.1.3 \
    pillow \
    transforms3d \
    tensorboard \
    wandb

# Optional but recommended: depth estimation stack used by test_depth_anything_overlap.py
echo "[2.5/5] Installing Depth Anything v2 dependencies..."
pip install transformers==4.49.0 safetensors

# Step 3: Install PyTorch with CUDA 12.1
# Note: torch==1.13 is incompatible with CUDA 12.8
# We use torch==2.4.0 which is forward-compatible
echo "[3/5] Installing PyTorch 2.4.0 with CUDA 12.1 support..."
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install compilation tools (ninja)
echo "[4/5] Installing compilation tools..."
conda install ninja -y

# Step 5: Build CUDA extensions
echo "[5/5] Building CUDA extensions..."

echo "  - Building pointops..."
cd "$SCRIPT_DIR/models/pointops"
python setup.py install > /dev/null 2>&1
echo "    ✓ pointops installed"

echo "  - Building Chamfer3D..."
cd "$SCRIPT_DIR/Chamfer3D"
python setup.py install > /dev/null 2>&1
echo "    ✓ Chamfer3D installed"

echo "  - Building emd_assignment..."
cd "$SCRIPT_DIR/emd_assignment"
python setup.py install > /dev/null 2>&1
echo "    ✓ emd_assignment installed"

echo ""
echo "========================================="
echo "✓ Environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate superpc"

