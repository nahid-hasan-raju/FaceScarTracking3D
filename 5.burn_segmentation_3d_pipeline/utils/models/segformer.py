"""
utils/models/segformer.py
=========================
SegFormer-B5 inference utility.
Loads a fine-tuned HuggingFace SegFormer checkpoint and returns a binary burn mask.

REQUIREMENTS:
    pip install transformers==4.40.0
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_dir: str):
    """
    Load SegFormer from a HuggingFace-format checkpoint directory.
    Returns (model, processor).
    """
    model     = SegformerForSemanticSegmentation.from_pretrained(ckpt_dir).to(DEVICE)
    processor = SegformerImageProcessor.from_pretrained(ckpt_dir)
    model.eval()
    print(f"[SegFormer] Loaded checkpoint: {ckpt_dir}")
    return model, processor


# def predict(model, processor, img_rgb: np.ndarray, threshold: float = 0.5) -> np.ndarray:
#     """
#     Run inference on a single RGB image (H x W x 3, uint8).
#     Returns binary mask (H x W, uint8) — 255 = burn, 0 = normal.
#     """
#     h, w    = img_rgb.shape[:2]
#     pil_img = Image.fromarray(img_rgb)
#     inputs  = processor(images=pil_img, return_tensors="pt")
#     pv      = inputs["pixel_values"].to(DEVICE)
#     with torch.no_grad():
#         out = model(pixel_values=pv)
#     logits = F.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
#     pred   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
#     mask   = (pred > 0).astype(np.uint8) * 255
#     return mask



def predict(model, processor, img_rgb: np.ndarray, threshold: float = 0.5):
    h, w    = img_rgb.shape[:2]
    pil_img = Image.fromarray(img_rgb)
    inputs  = processor(images=pil_img, return_tensors="pt")
    pv      = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        out = model(pixel_values=pv)
    logits = F.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False)
    probs  = torch.softmax(logits, dim=1)          # convert to probabilities
    burn_prob = probs[0, 1].cpu().numpy()          # probability of class 1 (burn)
    mask = (burn_prob > threshold).astype(np.uint8) * 255
    return mask, burn_prob.astype(np.float32)