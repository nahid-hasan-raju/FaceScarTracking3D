"""
utils/models/medsam.py
======================
MedSAM (SAM ViT-B fine-tuned on medical images) inference utility.
Returns a binary burn mask using a full-image prompt (no GT bbox needed).

REQUIREMENTS:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Place MedSAM base weights at: checkpoints/medsam_vit_b.pth
    # Place fine-tuned weights at:  checkpoints/medsam_best.pth  (your trained .pth)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import sys
import os

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
FULL_BOX = np.array([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=np.float32)


def _ensure_sam(sam2_repo_path: str = None):
    """Add segment-anything to path if needed."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
        return sam_model_registry, SamPredictor
    except ImportError:
        if sam2_repo_path:
            sys.path.insert(0, sam2_repo_path)
        try:
            from segment_anything import sam_model_registry, SamPredictor
            return sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not found.\n"
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )


def load_model(base_ckpt: str, finetuned_ckpt: str, sam_repo_path: str = None):
    """
    Load MedSAM.
    base_ckpt      : path to medsam_vit_b.pth  (base SAM ViT-B weights)
    finetuned_ckpt : path to your best.pth      (fine-tuned decoder weights)
    Returns (model, predictor).
    """
    sam_model_registry, SamPredictor = _ensure_sam(sam_repo_path)
    model = sam_model_registry["vit_b"](checkpoint=base_ckpt)
    # Load fine-tuned weights (decoder + prompt encoder)
    ft_state = torch.load(finetuned_ckpt, map_location=DEVICE)
    model.load_state_dict(ft_state, strict=False)
    model.to(DEVICE).eval()
    predictor = SamPredictor(model)
    print(f"[MedSAM] Base: {base_ckpt}")
    print(f"[MedSAM] Fine-tuned: {finetuned_ckpt}")
    return model, predictor


# def predict(model, predictor, img_rgb: np.ndarray) -> np.ndarray:
#     """
#     Run inference on a single RGB image (H x W x 3, uint8).
#     Uses full-image box prompt — no GT used.
#     Returns binary mask (H x W, uint8) — 255 = burn, 0 = normal.
#     """
#     h, w = img_rgb.shape[:2]
#     img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

#     predictor.set_image(img_resized)
#     masks, scores, _ = predictor.predict(
#         box            = FULL_BOX,
#         multimask_output = False,
#     )
#     mask = masks[0].astype(np.uint8)                           # (1024, 1024)
#     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#     return (mask * 255).astype(np.uint8)

def predict(model, predictor, img_rgb: np.ndarray, threshold: float = 0.5):
    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    predictor.set_image(img_resized)
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            box              = FULL_BOX,
            multimask_output = False,
        )
    # logits is raw — apply sigmoid to get probability
    prob = torch.sigmoid(torch.tensor(logits[0])).numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > threshold).astype(np.uint8) * 255
    return mask, prob.astype(np.float32)