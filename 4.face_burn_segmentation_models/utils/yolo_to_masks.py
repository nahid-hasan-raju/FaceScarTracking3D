"""
Convert YOLOv8 segmentation labels → binary PNG masks.
Each label line: class x1 y1 x2 y2 ... (normalized polygon coords)
Output mask: 0 = background, 255 = burn
"""
import os
import cv2
import numpy as np
from PIL import Image


def yolo_poly_to_mask(label_path, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coords = list(map(float, parts[1:]))  # skip class id
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def convert_split(img_dir, label_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    converted = 0
    for img_file in img_files:
        img = np.array(Image.open(os.path.join(img_dir, img_file)))
        h, w = img.shape[:2]
        stem = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, stem + '.txt')
        mask = yolo_poly_to_mask(label_path, w, h)
        cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask)
        converted += 1
    print(f"  Converted {converted} images → masks in {mask_dir}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base, 'dataset', split, 'images')
        if not os.path.exists(img_dir):
            print(f"  Skipping {split} (folder not found)")
            continue
        print(f"Converting {split}...")
        convert_split(
            img_dir   = img_dir,
            label_dir = os.path.join(base, 'dataset', split, 'labels'),
            mask_dir  = os.path.join(base, 'dataset', split, 'masks'),
        )
    print("Done! Masks saved as PNG files.")
