"""
SAM2 Fine-Tuning — HONEST version.
No bbox prompt. Model sees full image and must find burn itself.
Train and predict both use identical full-image prompt.
RUN: python method1_sam2/train.py
"""
import os, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_DIR = os.path.join(BASE, 'segment-anything-2')
sys.path.insert(0, BASE)
sys.path.insert(0, SAM2_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.losses import DiceFocalLoss

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SAM2_CFG  = os.path.join(SAM2_DIR, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
CKPT_IN   = os.path.join(BASE, 'checkpoints', 'sam2', 'sam2_hiera_large.pt')
CKPT_OUT  = os.path.join(BASE, 'checkpoints', 'sam2', 'best.pth')
TRAIN_IMG = os.path.join(BASE, 'dataset', 'train', 'images')
TRAIN_MSK = os.path.join(BASE, 'dataset', 'train', 'masks')
VALID_IMG = os.path.join(BASE, 'dataset', 'valid', 'images')
VALID_MSK = os.path.join(BASE, 'dataset', 'valid', 'masks')
TEST_IMG  = os.path.join(BASE, 'dataset', 'test',  'images')
TEST_MSK  = os.path.join(BASE, 'dataset', 'test',  'masks')
IMG_SIZE  = 1024
EPOCHS    = 50
LR        = 1e-4
# Full image — model must find burn on its own
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
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask_t


def forward_pass(model, predictor, img_np, box_np):
    predictor.set_image(img_np)
    feats    = predictor._features
    img_emb  = feats['image_embed']
    high_res = feats['high_res_feats']
    box_t    = torch.from_numpy(box_np).to(DEVICE)
    sparse, dense = model.sam_prompt_encoder(
        points=None, boxes=box_t, masks=None)
    out = model.sam_mask_decoder(
        image_embeddings=img_emb,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res,
    )
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                         mode='bilinear', align_corners=False)


def train():
    torch.cuda.empty_cache()
    if not os.path.exists(SAM2_DIR):
        print(f"ERROR: {SAM2_DIR} not found"); return
    if not os.path.exists(CKPT_IN):
        print(f"ERROR: {CKPT_IN} not found"); return
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("Run: cd segment-anything-2 && python -m pip install -e ."); return

    print(f"Device : {DEVICE}")
    print("Training with FULL IMAGE prompt — no location hints")
    model     = build_sam2(SAM2_CFG, CKPT_IN, device=DEVICE)
    predictor = SAM2ImagePredictor(model)

    for name, p in model.named_parameters():
        p.requires_grad = 'mask_decoder' in name
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    collate  = lambda x: x
    train_ds = BurnDataset(TRAIN_IMG, TRAIN_MSK)
    valid_ds = BurnDataset(VALID_IMG, VALID_MSK)
    test_ds  = BurnDataset(TEST_IMG,  TEST_MSK)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = DiceFocalLoss()
    best_dice = 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            img, mask_gt = batch[0]
            mask_gt = mask_gt.unsqueeze(0).to(DEVICE)

            with torch.inference_mode(False):
                with torch.no_grad():
                    predictor.set_image(img)
                feats    = predictor._features
                img_emb  = feats['image_embed'].clone()
                high_res = [f.clone() for f in feats['high_res_feats']]

            box_t = torch.from_numpy(FULL_BOX).to(DEVICE)
            sparse, dense = model.sam_prompt_encoder(
                points=None, boxes=box_t, masks=None)
            out = model.sam_mask_decoder(
                image_embeddings=img_emb,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res,
            )
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                                   mode='bilinear', align_corners=False)
            loss = criterion(logits, mask_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()

        scheduler.step()

        model.eval()
        dice_list = []
        with torch.no_grad():
            for batch in valid_dl:
                img, mask_gt = batch[0]
                mask_gt = mask_gt.unsqueeze(0).to(DEVICE)
                logits  = forward_pass(model, predictor, img, FULL_BOX)
                pred    = (torch.sigmoid(logits) > 0.5).float()
                inter   = (pred * mask_gt).sum()
                dice    = (2*inter / (pred.sum() + mask_gt.sum() + 1e-6)).item()
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
        for batch in test_dl:
            img, mask_gt = batch[0]
            mask_gt = mask_gt.unsqueeze(0).to(DEVICE)
            logits  = forward_pass(model, predictor, img, FULL_BOX)
            pred    = (torch.sigmoid(logits) > 0.5).float()
            inter   = (pred * mask_gt).sum()
            dice    = (2*inter / (pred.sum() + mask_gt.sum() + 1e-6)).item()
            test_scores.append(dice)
    print(f"Test Dice: {np.mean(test_scores):.4f}  ({len(test_scores)} images)")


if __name__ == '__main__':
    train()
