"""
Shared PyTorch Dataset for burn scar segmentation.
Minimal augmentation — avoids RAM issues on Windows.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size=512, augment=True):
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.4),
            A.Rotate(limit=15, p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class BurnDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, augment=True):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = get_transforms(img_size, augment)
        all_imgs = [f for f in os.listdir(img_dir)
                    if f.endswith(('.jpg', '.png'))]
        self.ids = []
        for f in all_imgs:
            stem = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(mask_dir, stem + '.png')):
                self.ids.append(stem)
        print(f"  Dataset: {len(self.ids)} samples in {img_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        for ext in ['.jpg', '.png']:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB'))
                break
        mask = np.array(Image.open(
            os.path.join(self.mask_dir, stem + '.png')).convert('L'))
        mask = (mask > 128).astype(np.float32)
        out  = self.transform(image=img, mask=mask)
        return out['image'], out['mask'].unsqueeze(0)
