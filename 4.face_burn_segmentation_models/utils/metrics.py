"""Dice, IoU, Precision, Recall metrics."""
import numpy as np


def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    """pred_mask: numpy float [0,1] or binary. gt_mask: binary numpy."""
    pred = (pred_mask > threshold).astype(bool)
    gt   = (gt_mask   > 0.5     ).astype(bool)
    tp   = (pred & gt ).sum()
    fp   = (pred & ~gt).sum()
    fn   = (~pred & gt).sum()
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return dict(dice=dice, iou=iou, precision=precision, recall=recall)


def print_metrics(results, model_name):
    keys = ['dice', 'iou', 'precision', 'recall']
    means = {k: np.mean([r[k] for r in results]) for k in keys}
    print(f"\n{'='*50}")
    print(f"  {model_name} Results ({len(results)} images)")
    print(f"{'='*50}")
    for k, v in means.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"{'='*50}")
    return means
