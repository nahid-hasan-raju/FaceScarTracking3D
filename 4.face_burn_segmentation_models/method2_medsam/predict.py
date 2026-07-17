"""
MedSAM Predict — HONEST version.
Full image prompt — no GT bbox. Consistent with training.
RUN: python method2_medsam/predict.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from utils.metrics import compute_metrics, print_metrics

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_IN  = os.path.join(BASE, 'checkpoints', 'medsam', 'medsam_vit_b.pth')
CKPT_FT  = os.path.join(BASE, 'checkpoints', 'medsam', 'best.pth')
TEST_IMG = os.path.join(BASE, 'dataset', 'test', 'images')
TEST_MSK = os.path.join(BASE, 'dataset', 'test', 'masks')
OUT_DIR  = os.path.join(BASE, 'outputs', 'medsam')
IMG_SIZE = 1024
FULL_BOX = np.array([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=np.float32)
THRESHOLD = 0.55


def load_model():
    from segment_anything import sam_model_registry
    model = sam_model_registry['vit_b'](checkpoint=CKPT_IN).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_FT, map_location=DEVICE))
    model.eval()
    return model


def predict_one(model, img_np):
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img_r).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.
    box_t = torch.from_numpy(FULL_BOX).to(DEVICE)
    with torch.no_grad():
        img_emb       = model.image_encoder(img_t)
        sparse, dense = model.prompt_encoder(
            points=None, boxes=box_t.unsqueeze(1), masks=None)
        logits, _     = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
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
    model   = load_model()
    results = []

    print("Running MedSAM predictions (full image, no GT hints)...")
    for fname in sorted(os.listdir(TEST_IMG)):
        if not fname.endswith(('.jpg','.png')): continue
        stem     = os.path.splitext(fname)[0]
        img_np   = np.array(Image.open(os.path.join(TEST_IMG, fname)).convert('RGB'))
        msk_path = os.path.join(TEST_MSK, stem+'.png')
        gt_np    = np.array(Image.open(msk_path).convert('L')) \
                   if os.path.exists(msk_path) \
                   else np.zeros(img_np.shape[:2], dtype=np.uint8)

        pred = predict_one(model, img_np)
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
        print_metrics(results, 'MedSAM (honest - no GT hints)')
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
