"""
Unified evaluation — compare all 4 models side by side.
Run AFTER running predict.py for each method.
Run: python evaluate_all.py
"""
import os
import numpy as np
import cv2
from PIL import Image

BASE    = os.path.dirname(os.path.abspath(__file__))
GT_DIR  = os.path.join(BASE, 'dataset', 'test', 'masks')
METHODS = {
    'UNet++':       os.path.join(BASE, 'outputs', 'unetpp'),
    'SegFormer-B5': os.path.join(BASE, 'outputs', 'segformer'),
    'MedSAM':       os.path.join(BASE, 'outputs', 'medsam'),
    'SAM 2':        os.path.join(BASE, 'outputs', 'sam2'),
}


def compute(pred, gt):
    pred = (pred > 128).astype(bool)
    gt   = (gt   > 0  ).astype(bool)
    tp   = (pred & gt ).sum()
    fp   = (pred & ~gt).sum()
    fn   = (~pred & gt).sum()
    return dict(
        dice = 2*tp / (2*tp + fp + fn + 1e-6),
        iou  =   tp / (   tp + fp + fn + 1e-6),
        prec =   tp / (   tp + fp      + 1e-6),
        rec  =   tp / (   tp      + fn + 1e-6),
    )


def evaluate_method(pred_dir, gt_dir):
    results = []
    for fname in os.listdir(pred_dir):
        if not fname.endswith('_mask.png'):
            continue
        # match GT: strip _mask suffix
        stem = fname.replace('_mask.png', '')
        gt_path = os.path.join(gt_dir, stem + '.png')
        if not os.path.exists(gt_path):
            continue
        pred = np.array(Image.open(os.path.join(pred_dir, fname)).convert('L'))
        gt   = np.array(Image.open(gt_path).convert('L'))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        results.append(compute(pred, gt))
    return results


print(f"\n{'Model':<15} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}  {'Samples':>8}")
print('-' * 62)

for name, pred_dir in METHODS.items():
    if not os.path.exists(pred_dir) or not os.listdir(pred_dir):
        print(f"{name:<15} {'(no predictions yet)':>40}")
        continue
    results = evaluate_method(pred_dir, GT_DIR)
    if not results:
        print(f"{name:<15} {'(no matching GT)':>40}")
        continue
    d = np.mean([r['dice'] for r in results])
    i = np.mean([r['iou']  for r in results])
    p = np.mean([r['prec'] for r in results])
    r = np.mean([r['rec']  for r in results])
    print(f"{name:<15} {d:>8.4f} {i:>8.4f} {p:>10.4f} {r:>8.4f}  {len(results):>8}")

print()
