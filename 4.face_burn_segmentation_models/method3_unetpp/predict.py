"""
UNet++ Inference — no prompt needed, searches entire image.
RUN: python method3_unetpp/predict.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from utils.metrics import compute_metrics, print_metrics

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_OUT = os.path.join(BASE, 'checkpoints', 'unetpp', 'best.pth')
TEST_IMG = os.path.join(BASE, 'dataset', 'test', 'images')
TEST_MSK = os.path.join(BASE, 'dataset', 'test', 'masks')
OUT_DIR  = os.path.join(BASE, 'outputs', 'unetpp')
IMG_SIZE = 512
THRESHOLD = 0.6

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


def load_model():
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4', encoder_weights=None,
        in_channels=3, classes=1, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_OUT, map_location=DEVICE))
    model.eval()
    return model


def predict_one(model, img_np):
    h, w = img_np.shape[:2]
    inp  = transform(image=img_np)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)[0,0]).cpu().numpy()
    prob = cv2.resize(prob, (w, h))
    return (prob > THRESHOLD).astype(np.uint8) * 255


def overlay(img_bgr, pred, gt, alpha=0.4):
    color = np.zeros_like(img_bgr)
    color[pred > 128] = (0, 0, 255)
    vis = cv2.addWeighted(img_bgr, 1-alpha, color, alpha, 0)
    cnts,  _ = cv2.findContours((pred>128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_gt, _ = cv2.findContours((gt>128).astype(np.uint8),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts,    -1, (0, 255, 255), 2)
    cv2.drawContours(vis, cnts_gt, -1, (0, 255,   0), 2)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model   = load_model()
    results = []

    print("Running UNet++ predictions on test set...")
    for fname in sorted(os.listdir(TEST_IMG)):
        if not fname.endswith(('.jpg', '.png')): continue
        stem     = os.path.splitext(fname)[0]
        img_np   = np.array(Image.open(os.path.join(TEST_IMG, fname)).convert('RGB'))
        msk_path = os.path.join(TEST_MSK, stem + '.png')
        gt_np    = np.array(Image.open(msk_path).convert('L')) if os.path.exists(msk_path) \
                   else np.zeros(img_np.shape[:2], dtype=np.uint8)

        pred = predict_one(model, img_np)
        cv2.imwrite(os.path.join(OUT_DIR, stem + '_mask.png'), pred)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT_DIR, stem + '_overlay.png'),
                    overlay(img_bgr, pred, gt_np))

        if os.path.exists(msk_path):
            m = compute_metrics(pred/255., gt_np/255.)
            results.append(m)
            print(f"  {fname}  Dice: {m['dice']:.4f}  IoU: {m['iou']:.4f}")

    if results: print_metrics(results, 'UNet++')
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
