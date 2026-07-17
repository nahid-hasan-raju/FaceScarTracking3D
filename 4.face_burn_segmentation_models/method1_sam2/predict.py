"""
SAM2 Predict — HONEST version.
Full image prompt — no GT bbox. Consistent with training.
RUN: python method1_sam2/predict.py
"""
import os, sys
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_DIR = os.path.join(BASE, 'segment-anything-2')
sys.path.insert(0, BASE)
sys.path.insert(0, SAM2_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from utils.metrics import compute_metrics, print_metrics

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SAM2_CFG = os.path.join(SAM2_DIR, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
CKPT_IN  = os.path.join(BASE, 'checkpoints', 'sam2', 'sam2_hiera_large.pt')
CKPT_FT  = os.path.join(BASE, 'checkpoints', 'sam2', 'best.pth')
TEST_IMG = os.path.join(BASE, 'dataset', 'test', 'images')
TEST_MSK = os.path.join(BASE, 'dataset', 'test', 'masks')
OUT_DIR  = os.path.join(BASE, 'outputs', 'sam2')
IMG_SIZE = 1024
FULL_BOX = np.array([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=np.float32)
THRESHOLD = 0.65

def load_model():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(SAM2_CFG, CKPT_IN, device=DEVICE)
    model.load_state_dict(torch.load(CKPT_FT, map_location=DEVICE))
    model.eval()
    return model, SAM2ImagePredictor(model)


def predict_one(model, predictor, img_np):
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    box_t = torch.from_numpy(FULL_BOX).to(DEVICE)
    with torch.no_grad():
        predictor.set_image(img_r)
        feats    = predictor._features
        sparse, dense = model.sam_prompt_encoder(
            points=None, boxes=box_t, masks=None)
        out = model.sam_mask_decoder(
            image_embeddings=feats['image_embed'],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=feats['high_res_feats'],
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = F.interpolate(logits, size=(h,w),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits[0,0]) > THRESHOLD).cpu().numpy().astype(np.uint8) * 255


def overlay(img_bgr, pred, gt, alpha=0.4):
    color = np.zeros_like(img_bgr)
    color[pred > 128] = (0, 0, 255)
    vis = cv2.addWeighted(img_bgr, 1-alpha, color, alpha, 0)
    for c, col in [
        (cv2.findContours((pred>128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], (0,255,255)),
        (cv2.findContours((gt>128).astype(np.uint8),   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], (0,255,0)),
    ]:
        cv2.drawContours(vis, c, -1, col, 2)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, predictor = load_model()
    results = []

    print("Running SAM2 predictions (full image, no GT hints)...")
    for fname in sorted(os.listdir(TEST_IMG)):
        if not fname.endswith(('.jpg','.png')): continue
        stem     = os.path.splitext(fname)[0]
        img_np   = np.array(Image.open(os.path.join(TEST_IMG, fname)).convert('RGB'))
        msk_path = os.path.join(TEST_MSK, stem+'.png')
        gt_np    = np.array(Image.open(msk_path).convert('L')) \
                   if os.path.exists(msk_path) \
                   else np.zeros(img_np.shape[:2], dtype=np.uint8)

        pred = predict_one(model, predictor, img_np)
        cv2.imwrite(os.path.join(OUT_DIR, stem+'_mask.png'), pred)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gt_r    = cv2.resize(gt_np, (img_np.shape[1], img_np.shape[0]))
        cv2.imwrite(os.path.join(OUT_DIR, stem+'_overlay.png'),
                    overlay(img_bgr, pred, gt_r))

        if os.path.exists(msk_path):
            m = compute_metrics(pred/255., gt_np/255.)
            results.append(m)
            print(f"  {fname}  Dice:{m['dice']:.4f} IoU:{m['iou']:.4f} "
                  f"Prec:{m['precision']:.4f} Rec:{m['recall']:.4f}")

    if results:
        print_metrics(results, 'SAM2 (honest - no GT hints)')
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
