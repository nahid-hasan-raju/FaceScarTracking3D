"""
utils/models/sam2.py
====================
SAM 2 (Hiera-Large) inference utility.
Returns a binary burn mask using a full-image prompt (no GT bbox needed).

REQUIREMENTS:
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2 && pip install -e . && cd ..
    # Place SAM2 base weights at:   checkpoints/sam2_hiera_large.pt
    # Place fine-tuned weights at:  checkpoints/sam2_best.pth  (your trained .pth)
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


def load_model(base_ckpt: str, finetuned_ckpt: str, sam2_repo_path: str):
    """
    Load SAM2 Hiera-Large.
    base_ckpt      : path to sam2_hiera_large.pt
    finetuned_ckpt : path to your best.pth  (fine-tuned decoder weights)
    sam2_repo_path : path to cloned segment-anything-2 repo
    Returns (model, predictor).
    """
    sys.path.insert(0, sam2_repo_path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg_path = os.path.join(sam2_repo_path, "sam2", "configs", "sam2", "sam2_hiera_l.yaml")
    model    = build_sam2(cfg_path, base_ckpt, device=DEVICE)

    ft_state = torch.load(finetuned_ckpt, map_location=DEVICE)
    model.load_state_dict(ft_state, strict=False)
    model.eval()

    predictor = SAM2ImagePredictor(model)
    print(f"[SAM2] Base: {base_ckpt}")
    print(f"[SAM2] Fine-tuned: {finetuned_ckpt}")
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
#     with torch.no_grad():
#         masks, scores, _ = predictor.predict(
#             box              = FULL_BOX,
#             multimask_output = False,
#         )
#     mask = masks[0].astype(np.uint8)
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
    return mask, prob.astype(np.float32)   # ← now returns both