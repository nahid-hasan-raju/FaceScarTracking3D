"""
Threshold Tuning — finds best threshold per model using validation set.
Run: python tune_threshold.py
"""
import os
import numpy as np
import cv2
from PIL import Image

BASE     = os.path.dirname(os.path.abspath(__file__))
VALID_GT = os.path.join(BASE, 'dataset', 'valid', 'masks')

# Point these to where your models save RAW probability maps (0-255 float)
# If your predict.py only saves binary masks, see note below
METHODS = {
    'UNet++':       os.path.join(BASE, 'outputs', 'unetpp'),
    'SegFormer-B5': os.path.join(BASE, 'outputs', 'segformer'),
    'MedSAM':       os.path.join(BASE, 'outputs', 'medsam'),
    'SAM 2':        os.path.join(BASE, 'outputs', 'sam2'),
}

THRESHOLDS = np.arange(0.10, 0.95, 0.05)


def load_prob_map(path):
    """Load grayscale image as float [0,1]."""
    img = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
    return img


def compute_dice(pred_bin, gt_bin):
    tp = (pred_bin & gt_bin).sum()
    fp = (pred_bin & ~gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()
    return 2 * tp / (2 * tp + fp + fn + 1e-6)


def compute_metrics(pred_bin, gt_bin):
    tp = (pred_bin & gt_bin).sum()
    fp = (pred_bin & ~gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return dice, iou, precision, recall


def tune_threshold(pred_dir, gt_dir, method_name):
    """
    Loads probability maps from pred_dir (files ending in _prob.png or _mask.png),
    sweeps thresholds, returns best threshold by Dice on validation set.
    """
    # Try to find probability maps first, fall back to masks
    pred_files = {}
    for f in os.listdir(pred_dir):
        if f.endswith('_prob.png'):
            stem = f.replace('_prob.png', '')
            pred_files[stem] = os.path.join(pred_dir, f)
        elif f.endswith('_mask.png') and f.replace('_mask.png','') not in pred_files:
            stem = f.replace('_mask.png', '')
            pred_files[stem] = os.path.join(pred_dir, f)

    # Match with GT
    pairs = []
    for stem, pred_path in pred_files.items():
        gt_path = os.path.join(gt_dir, stem + '.png')
        if os.path.exists(gt_path):
            pairs.append((pred_path, gt_path))

    if not pairs:
        print(f"  {method_name}: No matching validation pairs found.")
        return None

    print(f"\n  {method_name} — {len(pairs)} validation images")
    print(f"  {'Threshold':>10} {'Dice':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*50}")

    best_thresh = 0.5
    best_dice   = 0.0
    best_row    = None

    for thresh in THRESHOLDS:
        dices, ious, precs, recs = [], [], [], []
        for pred_path, gt_path in pairs:
            prob = load_prob_map(pred_path)
            gt   = (np.array(Image.open(gt_path).convert('L')) > 128)

            # Resize prob map to GT size if needed
            if prob.shape != gt.shape:
                prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]))

            pred_bin = prob > thresh
            d, i, p, r = compute_metrics(pred_bin, gt)
            dices.append(d); ious.append(i); precs.append(p); recs.append(r)

        mean_dice = np.mean(dices)
        mean_iou  = np.mean(ious)
        mean_prec = np.mean(precs)
        mean_rec  = np.mean(recs)

        marker = " ◄ best" if mean_dice > best_dice else ""
        print(f"  {thresh:>10.2f} {mean_dice:>8.4f} {mean_iou:>8.4f} "
              f"{mean_prec:>10.4f} {mean_rec:>8.4f}{marker}")

        if mean_dice > best_dice:
            best_dice   = mean_dice
            best_thresh = thresh
            best_row    = (mean_dice, mean_iou, mean_prec, mean_rec)

    print(f"\n  ✅ Best threshold: {best_thresh:.2f}  →  Dice: {best_row[0]:.4f}  "
          f"Precision: {best_row[2]:.4f}  Recall: {best_row[3]:.4f}")
    return best_thresh, best_row


def evaluate_test_with_threshold(pred_dir, gt_dir, threshold, method_name):
    """Re-evaluate test set using the tuned threshold."""
    print(f"\n  {method_name} TEST results at threshold={threshold:.2f}")
    pred_files = {}
    for f in os.listdir(pred_dir):
        if f.endswith('_prob.png'):
            stem = f.replace('_prob.png', '')
            pred_files[stem] = os.path.join(pred_dir, f)
        elif f.endswith('_mask.png') and f.replace('_mask.png','') not in pred_files:
            stem = f.replace('_mask.png', '')
            pred_files[stem] = os.path.join(pred_dir, f)

    pairs = []
    for stem, pred_path in pred_files.items():
        gt_path = os.path.join(gt_dir, stem + '.png')
        if os.path.exists(gt_path):
            pairs.append((pred_path, gt_path))

    dices, ious, precs, recs = [], [], [], []
    for pred_path, gt_path in pairs:
        prob = load_prob_map(pred_path)
        gt   = (np.array(Image.open(gt_path).convert('L')) > 128)
        if prob.shape != gt.shape:
            prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]))
        pred_bin = prob > threshold
        d, i, p, r = compute_metrics(pred_bin, gt)
        dices.append(d); ious.append(i); precs.append(p); recs.append(r)

    print(f"  Dice: {np.mean(dices):.4f}  IoU: {np.mean(ious):.4f}  "
          f"Precision: {np.mean(precs):.4f}  Recall: {np.mean(recs):.4f}  "
          f"(n={len(pairs)})")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    TEST_GT = os.path.join(BASE, 'dataset', 'test', 'masks')

    print("=" * 60)
    print("  THRESHOLD TUNING — using validation set")
    print("=" * 60)

    best_thresholds = {}
    for name, pred_dir in METHODS.items():
        if not os.path.exists(pred_dir) or not os.listdir(pred_dir):
            print(f"\n  {name}: No outputs found, skipping.")
            continue
        result = tune_threshold(pred_dir, VALID_GT, name)
        if result:
            best_thresholds[name] = result[0]

    # Now re-evaluate test set with tuned thresholds
    print("\n" + "=" * 60)
    print("  TEST SET RESULTS — with tuned thresholds")
    print("=" * 60)

    test_pred_dirs = {
        'UNet++':       os.path.join(BASE, 'outputs', 'unetpp'),
        'SegFormer-B5': os.path.join(BASE, 'outputs', 'segformer'),
        'MedSAM':       os.path.join(BASE, 'outputs', 'medsam'),
        'SAM 2':        os.path.join(BASE, 'outputs', 'sam2'),
    }

    # NOTE: test outputs should be in a separate folder from valid outputs
    # If you run predict.py pointing at test images, point test_pred_dirs there
    for name, thresh in best_thresholds.items():
        if name in test_pred_dirs and os.path.exists(test_pred_dirs[name]):
            evaluate_test_with_threshold(test_pred_dirs[name], TEST_GT, thresh, name)
