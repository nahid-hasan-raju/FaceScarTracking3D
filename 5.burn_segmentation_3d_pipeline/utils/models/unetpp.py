"""
utils/models/unetpp.py
======================
UNet++ inference utility.
Loads a fine-tuned EfficientNet-B4 UNet++ checkpoint and returns a binary burn mask.

REQUIREMENTS:
    pip install segmentation-models-pytorch albumentations
"""

import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

IMG_SIZE = 512
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_model(ckpt_path: str):
    """Load UNet++ from .pth checkpoint. Returns model on DEVICE."""
    model = smp.UnetPlusPlus(
        encoder_name    = "efficientnet-b4",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"[UNet++] Loaded checkpoint: {ckpt_path}")
    return model


def predict(model, img_rgb: np.ndarray, threshold: float = 0.5):
    """
    Run inference on a single RGB image (H x W x 3, uint8).
    Returns (mask, prob):
      mask — binary mask (H x W, uint8), 255 = burn, 0 = normal
      prob — float32 probability map (H x W), in [0, 1]
    """
    h, w = img_rgb.shape[:2]
    inp  = _transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(inp)
        prob  = torch.sigmoid(logit[0, 0]).cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > threshold).astype(np.uint8) * 255
    return mask, prob.astype(np.float32)
