"""
SegFormer-B5 fine-tuning — full image segmentation, no prompt needed.
RUN: python method4_segformer/train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import albumentations as A
import cv2

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS    = 60
LR        = 6e-5
BATCH     = 2
CKPT_OUT  = os.path.join(BASE, 'checkpoints', 'segformer')
TRAIN_IMG = os.path.join(BASE, 'dataset', 'train', 'images')
TRAIN_MSK = os.path.join(BASE, 'dataset', 'train', 'masks')
VALID_IMG = os.path.join(BASE, 'dataset', 'valid', 'images')
VALID_MSK = os.path.join(BASE, 'dataset', 'valid', 'masks')
TEST_IMG  = os.path.join(BASE, 'dataset', 'test',  'images')
TEST_MSK  = os.path.join(BASE, 'dataset', 'test',  'masks')


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.processor = SegformerImageProcessor(
            do_resize=True, size={'height': 512, 'width': 512})
        all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))]
        self.ids = [os.path.splitext(f)[0] for f in all_imgs
                    if os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0]+'.png'))]
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.4),
        ]) if augment else None
        print(f"  Found {len(self.ids)} samples in {img_dir}")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        for ext in ['.jpg', '.png']:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert('RGB'))
                break
        mask = np.array(Image.open(
            os.path.join(self.mask_dir, stem+'.png')).convert('L'))
        mask = (mask > 128).astype(np.uint8)
        if self.aug:
            out = self.aug(image=img, mask=mask)
            img, mask = out['image'], out['mask']
        inputs = self.processor(images=Image.fromarray(img), return_tensors='pt')
        pv     = inputs['pixel_values'].squeeze(0)
        mask_t = torch.from_numpy(mask).long()
        mask_t = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0).float(),
                               size=(128,128), mode='nearest').squeeze().long()
        return pv, mask_t


def iou_score(logits, masks):
    pred = logits.argmax(dim=1)
    tp = ((pred==1)&(masks==1)).float().sum()
    fp = ((pred==1)&(masks==0)).float().sum()
    fn = ((pred==0)&(masks==1)).float().sum()
    return (tp / (tp+fp+fn+1e-6)).item()


def validate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for pv, masks in loader:
            pv, masks = pv.to(DEVICE), masks.to(DEVICE)
            out = model(pixel_values=pv)
            scores.append(iou_score(out.logits, masks))
    return np.mean(scores)


def train():
    print(f"Device: {DEVICE}")
    train_ds = SegDataset(TRAIN_IMG, TRAIN_MSK, augment=True)
    valid_ds = SegDataset(VALID_IMG, VALID_MSK, augment=False)
    test_ds  = SegDataset(TEST_IMG,  TEST_MSK,  augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=1,     shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=1,     shuffle=False, num_workers=0)

    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b5',
        ignore_mismatched_sizes=True,
        num_labels=2,
        id2label={0:'background', 1:'burn'},
        label2id={'background':0, 'burn':1},
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    ce_loss   = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(DEVICE))
    best_iou  = 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for pv, masks in train_dl:
            pv, masks = pv.to(DEVICE), masks.to(DEVICE)
            out  = model(pixel_values=pv)
            loss = ce_loss(out.logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_iou = validate(model, valid_dl)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            model.save_pretrained(CKPT_OUT)
            SegformerImageProcessor(
                do_resize=True, size={'height': 512, 'width': 512}
            ).save_pretrained(CKPT_OUT)
            print(f"  -> Best saved (IoU: {best_iou:.4f})")

    print(f"\nDone! Best Val IoU: {best_iou:.4f}")

    print("\nEvaluating on test set...")
    model = SegformerForSemanticSegmentation.from_pretrained(CKPT_OUT).to(DEVICE)
    test_iou = validate(model, test_dl)
    print(f"Test IoU: {test_iou:.4f}  ({len(test_ds)} images)")


if __name__ == '__main__':
    train()
