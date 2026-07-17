"""
MedSAM Fine-Tuning — HONEST version.
Full image prompt — model must find burn itself.
RUN: python method2_medsam/train.py
"""
import os, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.losses import DiceFocalLoss

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_IN   = os.path.join(BASE, 'checkpoints', 'medsam', 'medsam_vit_b.pth')
CKPT_OUT  = os.path.join(BASE, 'checkpoints', 'medsam', 'best.pth')
TRAIN_IMG = os.path.join(BASE, 'dataset', 'train', 'images')
TRAIN_MSK = os.path.join(BASE, 'dataset', 'train', 'masks')
VALID_IMG = os.path.join(BASE, 'dataset', 'valid', 'images')
VALID_MSK = os.path.join(BASE, 'dataset', 'valid', 'masks')
TEST_IMG  = os.path.join(BASE, 'dataset', 'test',  'images')
TEST_MSK  = os.path.join(BASE, 'dataset', 'test',  'masks')
IMG_SIZE  = 1024
EPOCHS    = 50
LR        = 2e-4
FULL_BOX  = np.array([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=np.float32)


class BurnDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))]
        self.ids = [os.path.splitext(f)[0] for f in all_imgs
                    if os.path.exists(os.path.join(mask_dir,
                       os.path.splitext(f)[0]+'.png'))]
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
        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        img_t  = torch.from_numpy(img).permute(2,0,1).float() / 255.
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


def run_model(model, imgs, box_t):
    with torch.no_grad():
        img_emb = model.image_encoder(imgs)
    sparse, dense = model.prompt_encoder(
        points=None, boxes=box_t.unsqueeze(1), masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=img_emb,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                         mode='bilinear', align_corners=False)


def train():
    torch.cuda.empty_cache()
    if not os.path.exists(CKPT_IN):
        print(f"ERROR: {CKPT_IN} not found"); return
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("Run: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return

    print(f"Device : {DEVICE}")
    print("Training with FULL IMAGE prompt — no location hints")
    model = sam_model_registry['vit_b'](checkpoint=CKPT_IN).to(DEVICE)
    for p in model.image_encoder.parameters():  p.requires_grad = False
    for p in model.mask_decoder.parameters():   p.requires_grad = True
    for p in model.prompt_encoder.parameters(): p.requires_grad = True
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_ds = BurnDataset(TRAIN_IMG, TRAIN_MSK)
    valid_ds = BurnDataset(VALID_IMG, VALID_MSK)
    test_ds  = BurnDataset(TEST_IMG,  TEST_MSK)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = DiceFocalLoss()
    box_t     = torch.from_numpy(FULL_BOX).to(DEVICE)
    best_dice = 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        model.image_encoder.eval()
        total_loss = 0.0

        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = run_model(model, imgs, box_t)
            loss   = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()

        scheduler.step()

        model.eval()
        dice_list = []
        with torch.no_grad():
            for imgs, masks in valid_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = run_model(model, imgs, box_t)
                pred   = (torch.sigmoid(logits) > 0.5).float()
                inter  = (pred * masks).sum()
                dice   = (2*inter / (pred.sum() + masks.sum() + 1e-6)).item()
                dice_list.append(dice)

        val_dice = np.mean(dice_list)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | Val Dice: {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  -> Best saved (Dice: {best_dice:.4f})")

    print(f"\nDone! Best Val Dice: {best_dice:.4f}")
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(CKPT_OUT, map_location=DEVICE))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for imgs, masks in test_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = run_model(model, imgs, box_t)
            pred   = (torch.sigmoid(logits) > 0.5).float()
            inter  = (pred * masks).sum()
            dice   = (2*inter / (pred.sum() + masks.sum() + 1e-6)).item()
            test_scores.append(dice)
    print(f"Test Dice: {np.mean(test_scores):.4f}  ({len(test_scores)} images)")


if __name__ == '__main__':
    train()
