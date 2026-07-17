#!/bin/bash
# ============================================================
# ONE-TIME SETUP — run this first before anything else
# ============================================================
set -e

echo "====================================="
echo " Burn Scar Segmentation — Setup"
echo "====================================="

# 1. Create conda env
conda create -n burn_seg python=3.10 -y
conda activate burn_seg

# 2. PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Core dependencies
pip install \
    opencv-python-headless \
    pillow numpy scipy matplotlib tqdm \
    albumentations \
    segmentation-models-pytorch \
    transformers accelerate \
    scikit-learn scikit-image

# 4. Convert YOLO labels → binary masks (run once)
python utils/yolo_to_masks.py

echo ""
echo "====================================="
echo " Setup complete!"
echo " Masks saved to dataset/train/masks  and  dataset/test/masks"
echo "====================================="
echo ""
echo "Next steps:"
echo "  Method 3 (easiest, run first): python method3_unetpp/train.py"
echo "  Method 4:                      python method4_segformer/train.py"
echo "  Method 2 (needs download):     see README.md → MedSAM section"
echo "  Method 1 (needs clone):        see README.md → SAM 2 section"
