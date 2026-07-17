#!/usr/bin/env python3
"""
burn_3d_pipeline.py
===================
Full pipeline: Unwrapped 2D TIFF  →  Burn Segmentation  →  3D Point Cloud

WHAT IT DOES:
  1. Loads the unwrapped 2D TIFF (face scan texture)
  2. Runs your trained model (.pth) to get a burn segmentation mask
  3. Colors burn pixels RED, keeps normal skin colors from the original TIF
  4. Wraps the colored image back onto the Cyberware 3D geometry
  5. Applies front-facing alignment (Rx=-90°, Ry=+90°)
  6. Saves: _burn3d.ply  +  _front.png snapshot

FOLDER STRUCTURE:
  burn_3d_pipeline/
    burn_3d_pipeline.py       ← this script
    utils/
      cyberware.py            ← 3D reconstruction (from step1)
      alignment.py            ← rotation alignment (from step2)
      writers.py              ← PLY writer + snapshot renderer
      models/
        unetpp.py             ← UNet++ inference
        segformer.py          ← SegFormer inference
        medsam.py             ← MedSAM inference
        sam2.py               ← SAM2 inference

USAGE:
  # UNet++ (single .pth file)
  python burn_3d_pipeline.py \\
      --range   "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C" \\
      --tif     "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C.tif" \\
      --model   unetpp \\
      --ckpt    "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/unetpp/best.pth" \\
      --output  "D:/NahidW/Dataset/3d_scans/burn_3d"

  # SegFormer (checkpoint is a FOLDER, not a .pth file)
  python burn_3d_pipeline.py \\
      --range   "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C" \\
      --tif     "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C.tif" \\
      --model   segformer \\
      --ckpt    "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/segformer" \\
      --output  "D:/NahidW/Dataset/3d_scans/burn_3d"

  # MedSAM (needs base weights + fine-tuned weights)
  python burn_3d_pipeline.py \\
      --range      "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C" \\
      --tif        "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C.tif" \\
      --model      medsam \\
      --ckpt       "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/medsam/best.pth" \\
      --base-ckpt  "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/medsam/medsam_vit_b.pth" \\
      --output     "D:/NahidW/Dataset/3d_scans/burn_3d"

  # SAM2 (needs base weights + fine-tuned weights + repo path)
  python burn_3d_pipeline.py \\
      --range      "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C" \\
      --tif        "D:/NahidW/Dataset/pat1/pat1day0C/pat1day0C.tif" \\
      --model      sam2 \\
      --ckpt       "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/sam2/best.pth" \\
      --base-ckpt  "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/sam2/sam2_hiera_large.pt" \\
      --sam2-repo  "D:/NahidW/Coding/5.face_burn_segmentation/segment-anything-2" \\
      --output     "D:/NahidW/Dataset/3d_scans/burn_3d"

  # Optional: fine-tune yaw if face isn't centered
  python burn_3d_pipeline.py ... --fine-yaw -12.0

  # Save intermediate mask/overlay images too
  python burn_3d_pipeline.py ... --save-mask

REQUIREMENTS:
  pip install numpy pillow open3d opencv-python

  + depending on model:
  UNet++    : pip install segmentation-models-pytorch albumentations
  SegFormer : pip install transformers==4.40.0
  MedSAM    : pip install git+https://github.com/facebookresearch/segment-anything.git
  SAM2      : git clone + pip install -e segment-anything-2
"""

import sys
import argparse
import re
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from PIL import Image

# Add pipeline root to path so `utils` imports work
sys.path.insert(0, str(Path(__file__).parent))

from utils.cyberware import range_to_3d
from utils.alignment  import build_pcd, align_front
from utils.writers    import write_ply, render_snapshot


# ─────────────────────────────────────────────────────────────────────────────
#  BURN COLOR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BURN_COLOR   = np.array([220, 50, 50],  dtype=np.uint8)   # Red for burn areas
NORMAL_ALPHA = 1.0                                          # Keep original skin color


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: LOAD TIFF → RGB NUMPY
# ─────────────────────────────────────────────────────────────────────────────

def load_tiff(tif_path: Path) -> np.ndarray:
    """
    Load TIFF (any bit depth) → 8-bit RGB numpy array.
    Handles 16-bit TIFF correctly — converts to 8-bit range.
    """
    img = Image.open(tif_path).convert("RGB")   # forces 8-bit RGB regardless of TIFF type
    arr = np.array(img)
    print(f"  [TIF] Loaded: {tif_path.name}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: RUN SEGMENTATION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(img_rgb: np.ndarray, args) -> np.ndarray:
    """
    Load the requested model and run inference on img_rgb.
    Returns binary mask (H x W, uint8) — 255 = burn, 0 = normal.
    """
    model_name = args.model.lower()
    print(f"\n  [SEG] Running model: {model_name.upper()}")

    if model_name == "unetpp":
        from utils.models.unetpp import load_model, predict
        model = load_model(args.ckpt)
        mask  = predict(model, img_rgb, threshold=args.threshold)

    elif model_name == "segformer":
        from utils.models.segformer import load_model, predict
        model, processor = load_model(args.ckpt)   # ckpt is a directory for SegFormer
        mask = predict(model, processor, img_rgb, threshold=args.threshold)

    elif model_name == "medsam":
        if not args.base_ckpt:
            raise ValueError("--base-ckpt required for MedSAM (path to medsam_vit_b.pth)")
        from utils.models.medsam import load_model, predict
        model, predictor = load_model(args.base_ckpt, args.ckpt)
        mask = predict(model, predictor, img_rgb, threshold=args.threshold)

    elif model_name == "sam2":
        if not args.base_ckpt:
            raise ValueError("--base-ckpt required for SAM2 (path to sam2_hiera_large.pt)")
        if not args.sam2_repo:
            raise ValueError("--sam2-repo required for SAM2 (path to cloned segment-anything-2 repo)")
        from utils.models.sam2 import load_model, predict
        model, predictor = load_model(args.base_ckpt, args.ckpt, args.sam2_repo)
        mask = predict(model, predictor, img_rgb, threshold=args.threshold)

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: unetpp, segformer, medsam, sam2")

    burn_px = int((mask > 128).sum())
    total   = mask.size
    pct     = 100.0 * burn_px / total
    print(f"  [SEG] Burn pixels: {burn_px:,} / {total:,}  ({pct:.1f}%)")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: APPLY MASK → COLORED TEXTURE
# ─────────────────────────────────────────────────────────────────────────────

def apply_burn_colors(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Returns a new image where:
      burn pixels  (mask > 128) → BURN_COLOR (red)
      normal pixels             → original skin color from img_rgb
    """
    colored = img_rgb.copy()
    burn_px = mask > 128
    colored[burn_px] = BURN_COLOR
    return colored


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 & 5: WRAP TO 3D + ALIGN
# ─────────────────────────────────────────────────────────────────────────────

def wrap_and_align(range_path: Path, colored_img: np.ndarray,
                   fine_yaw: float = 0.0):
    """
    Reconstruct 3D from Cyberware range file using colored_img as texture.
    Applies front-facing alignment.
    Returns aligned open3d PointCloud.
    """
    print(f"\n  [3D] Reconstructing from range file...")
    pts, colors, _, _, _ = range_to_3d(range_path, colored_img)
    print(f"  [3D] Valid points: {len(pts):,}")

    pcd = build_pcd(pts, colors)
    pcd = align_front(pcd, fine_yaw=fine_yaw)
    print(f"  [3D] Aligned: Rx=-90°, Ry=+90°" +
          (f", fine Ry={fine_yaw:+.1f}°" if fine_yaw != 0.0 else ""))
    return pcd, pts, colors




def process_one_scan(range_path, tif_path, args):
    scan_name  = range_path.stem
    m          = re.match(r"(pat\d+)", scan_name, re.IGNORECASE)
    patient_id = m.group(1).lower() if m else "unknown"

    output_dir = Path(args.output) / args.model / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Scan: {scan_name}  |  Patient: {patient_id.upper()}")
    print(f"{'─'*60}")

    img_rgb     = load_tiff(tif_path)
    mask        = run_segmentation(img_rgb, args)
    colored_img = apply_burn_colors(img_rgb, mask)
    print(f"\n  [COLOR] Burn areas colored red on texture")

    if args.save_mask:
        mask_path    = output_dir / f"{scan_name}_burn_mask.png"
        overlay_path = output_dir / f"{scan_name}_burn_overlay.png"
        colored_path = output_dir / f"{scan_name}_burn_texture.png"
        cv2.imwrite(str(mask_path), mask)
        overlay = img_rgb.copy()
        overlay[mask > 128] = (
            overlay[mask > 128].astype(np.float32) * 0.4 +
            np.array(BURN_COLOR, dtype=np.float32) * 0.6
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            (mask > 128).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            contours, -1, (0, 255, 255), 2
        )
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(colored_path), cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE] Mask    → {mask_path.name}")
        print(f"  [SAVE] Overlay → {overlay_path.name}")
        print(f"  [SAVE] Texture → {colored_path.name}")

    pcd, pts, colors = wrap_and_align(range_path, colored_img, fine_yaw=args.fine_yaw)
    ply_path = output_dir / f"{scan_name}_burn3d.ply"
    png_path = output_dir / f"{scan_name}_front.png"
    pts_aligned    = np.asarray(pcd.points).astype(np.float32)
    colors_aligned = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    sz = write_ply(pts_aligned, colors_aligned, ply_path)
    render_snapshot(pcd, png_path)
    print(f"  [OUT] PLY      → {ply_path.name}  ({sz/1e6:.1f} MB)")
    print(f"  [OUT] Snapshot → {png_path.name}")

    if args.show:
        o3d.visualization.draw_geometries(
            [pcd], window_name=f"{scan_name} — burn areas in red",
            width=900, height=700)
        
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Burn segmentation → 3D point cloud pipeline"
    )
    p.add_argument("--config",     default=None,
                   help="Path to config.yaml (all other args optional if config provided)")
    p.add_argument("--range",      default=None,
                   help="Path to Cyberware range file (no extension)")
    p.add_argument("--tif",        default=None,
                   help="Path to unwrapped 2D TIFF image")
    p.add_argument("--model",      default=None,
                   choices=["unetpp", "segformer", "medsam", "sam2"],
                   help="Which segmentation model to use")
    p.add_argument("--ckpt",       default=None,
                   help="Path to fine-tuned checkpoint (.pth for unetpp/medsam/sam2, folder for segformer)")
    p.add_argument("--base-ckpt",  default=None,
                   help="Base pretrained weights (required for medsam and sam2)")
    p.add_argument("--sam2-repo",  default=None,
                   help="Path to cloned segment-anything-2 repo (required for sam2)")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Segmentation threshold (default: 0.5, UNet++/SegFormer only)")
    p.add_argument("--output",     default=None,
                   help="Output directory (default: ./outputs)")
    p.add_argument("--fine-yaw",   type=float, default=None,
                   help="Fine yaw correction in degrees (positive=left, negative=right)")
    p.add_argument("--save-mask",  action="store_true",
                   help="Also save intermediate mask and overlay images")
    p.add_argument("--show",       action="store_true",
                   help="Open 3D viewer after processing (requires display)")
    p.add_argument("--dataset", default=None,
               help="Dataset root folder to process all scans (batch mode)")

    args = p.parse_args()

    # Load config file — CLI args override config values
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if args.range     is None: args.range     = cfg.get("range")
        if not hasattr(args, 'dataset') or args.dataset is None: args.dataset = cfg.get("dataset")
        if args.tif       is None: args.tif       = cfg.get("tif")
        if args.model     is None: args.model     = cfg.get("model")
        if args.ckpt      is None: args.ckpt      = cfg.get("ckpt")
        if args.base_ckpt is None: args.base_ckpt = cfg.get("base_ckpt")
        if args.sam2_repo is None: args.sam2_repo = cfg.get("sam2_repo")
        if args.threshold is None: args.threshold = cfg.get("threshold", 0.5)
        if args.output    is None: args.output    = cfg.get("output", "./outputs")
        if args.fine_yaw  is None: args.fine_yaw  = cfg.get("fine_yaw", 0.0)
        if not args.save_mask:     args.save_mask = cfg.get("save_mask", False)
        if not args.show:          args.show      = cfg.get("show", False)

    # Apply defaults for anything still None
    if args.threshold is None: args.threshold = 0.5
    if args.output    is None: args.output    = "./outputs"
    if args.fine_yaw  is None: args.fine_yaw  = 0.0

    # Validate required fields
    is_batch = hasattr(args, 'dataset') and args.dataset
    for field, name in [(args.model, "--model"), (args.ckpt, "--ckpt")]:
        if not field:
            p.error(f"{name} is required (set in config.yaml or pass as argument)")
    if not is_batch:
        for field, name in [(args.range, "--range"), (args.tif, "--tif")]:
            if not field:
                p.error(f"{name} is required in single mode (or set dataset for batch mode)")

    return args


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Burn 3D Pipeline")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Output  : {args.output}")
    print(f"{'='*60}")

    # ── Batch mode: dataset folder ────────────────────────────────────────────
    if hasattr(args, 'dataset') and args.dataset:
        dataset_dir = Path(args.dataset)
        scans = []
        for scan_folder in sorted(dataset_dir.glob("*/*/")):
            scan_name = scan_folder.name
            if not re.match(r"pat\d+day\d+[A-Z][A-Z0-9]?$", scan_name, re.IGNORECASE):
                continue
            range_path = scan_folder / scan_name
            tif_path   = scan_folder / f"{scan_name}.tif"
            if range_path.exists() and tif_path.exists():
                scans.append((range_path, tif_path))

        if not scans:
            print("  No scans found in dataset folder.")
            sys.exit(1)

        print(f"  Found {len(scans)} scan(s)\n")
        for range_path, tif_path in scans:
            try:
                process_one_scan(range_path, tif_path, args)
            except Exception as e:
                print(f"  ✗ {range_path.stem} failed: {e}")

    # ── Single mode: explicit range + tif ────────────────────────────────────
    else:
        process_one_scan(Path(args.range), Path(args.tif), args)

    print(f"\n{'='*60}")
    print(f"  ALL DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()