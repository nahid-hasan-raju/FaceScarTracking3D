#!/usr/bin/env python3
"""
step1_burn_segmented_into_3d_pipeline.py
=========================================
Full pipeline: Unwrapped 2D TIFF  →  Burn Segmentation  →  3D Point Cloud

WHAT IT DOES:
  1. Loads the unwrapped 2D TIFF (face scan texture)
  2. Runs your trained model (.pth) to get a burn segmentation mask
  3. Colors burn pixels solid RED, keeps normal skin colors from the original TIF
  4. Wraps the colored image back onto the Cyberware 3D geometry
  5. Applies front-facing alignment (Rx=-90°, Ry=+90°)
  6. Saves to output folder: _burn3d.ply  +  _front.png snapshot

DATASET FOLDER STRUCTURE EXPECTED:
  <dataset>/
    PAT01/
      D00/
        PAT01_D00_A/
          PAT01_D00_A.tif        ← texture scan
          PAT01_D00_A            ← Cyberware range file (no extension)
        PAT01_D00_B/
          ...
      D14/
        ...
    PAT02/
      ...

OUTPUTS (written to --output folder, mirroring patient structure):
  <output>/<model>/pat01/
    PAT01_D00_A_burn3d.ply
    PAT01_D00_A_front.png
    PAT01_D00_A_burn_mask.png      (--save-mask only)
    PAT01_D00_A_burn_overlay.png   (--save-mask only)
    PAT01_D00_A_burn_texture.png   (--save-mask only)

USAGE:
  # Batch — all patients:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml

  # Batch — one patient:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml --patient PAT01

  # Batch — one patient, one timepoint:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml --patient PAT01 --timepoint D00

  # Single scan subfolder:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml --scandir D:/.../PAT01/D00/PAT01_D00_A

REQUIREMENTS:
  pip install numpy pillow open3d opencv-python pyyaml imagecodecs tifffile
  + model-specific: see README.md
"""

import sys
import argparse
import re
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

from utils.cyberware import range_to_3d
from utils.alignment  import build_pcd, align_front
from utils.writers    import write_ply, render_snapshot


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BURN_COLOR = np.array([220, 50, 50], dtype=np.uint8)   # solid red for burn areas

# Matches:  PAT01_D00_A  /  PAT03_M06_B2
SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)

def _tp_key(name):
    """Sort timepoints D00 < D14 < D28 < M01 … M24"""
    m = re.match(r'([DM])(\d+)', name)
    if not m:
        return (99, 0)
    return (0 if m.group(1).upper() == 'D' else 1, int(m.group(2)))


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD TIFF
# ─────────────────────────────────────────────────────────────────────────────

def load_tiff(tif_path: Path) -> np.ndarray:
    """Load .tif as uint8 RGB array, normalising whatever bit-depth it has."""
    raw = tifffile.imread(str(tif_path))

    # Multi-page: take first page
    if raw.ndim == 3 and raw.shape[0] < 10:
        raw = raw[0]

    # Normalise to 0-255
    raw = raw.astype(np.float32)
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        raw = (raw - lo) / (hi - lo) * 255.0
    raw = raw.astype(np.uint8)

    # Ensure RGB (H, W, 3)
    if raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=-1)
    elif raw.shape[-1] == 4:
        raw = raw[..., :3]

    print(f"  [TIF] Loaded: {tif_path.name}  shape={raw.shape}  dtype={raw.dtype}")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(img_rgb: np.ndarray, args) -> np.ndarray:
    """
    Load the requested model and run inference on img_rgb.
    Returns binary mask (H x W, uint8) — 255 = burn, 0 = normal.
    """
    model_name = args.model.lower()
    print(f"\n  [SEG] Running model: {model_name.upper()}  threshold={args.threshold}")

    if model_name == "unetpp":
        from utils.models.unetpp import load_model, predict
        model = load_model(args.ckpt)
        mask  = predict(model, img_rgb, threshold=args.threshold)

    elif model_name == "segformer":
        from utils.models.segformer import load_model, predict
        model, processor = load_model(args.ckpt)
        mask = predict(model, processor, img_rgb, threshold=args.threshold)

    elif model_name == "medsam":
        if not args.base_ckpt:
            raise ValueError("--base-ckpt required for MedSAM")
        from utils.models.medsam import load_model, predict
        model, predictor = load_model(args.base_ckpt, args.ckpt)
        mask = predict(model, predictor, img_rgb, threshold=args.threshold)

    elif model_name == "sam2":
        if not args.base_ckpt:
            raise ValueError("--base-ckpt required for SAM2")
        if not args.sam2_repo:
            raise ValueError("--sam2-repo required for SAM2")
        from utils.models.sam2 import load_model, predict
        model, predictor = load_model(args.base_ckpt, args.ckpt, args.sam2_repo)
        mask = predict(model, predictor, img_rgb, threshold=args.threshold)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    burn_px = int((mask > 128).sum())
    pct     = 100.0 * burn_px / mask.size
    print(f"  [SEG] Burn pixels: {burn_px:,} / {mask.size:,}  ({pct:.1f}%)")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — APPLY MASK → COLORED TEXTURE (solid red on burn, skin elsewhere)
# ─────────────────────────────────────────────────────────────────────────────

def apply_burn_colors(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Returns a new image where:
      burn pixels  (mask > 128) → solid BURN_COLOR (red)
      normal pixels             → original skin color from img_rgb
    """
    colored = img_rgb.copy()
    colored[mask > 128] = BURN_COLOR
    return colored


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4+5 — WRAP TO 3D + ALIGN
# ─────────────────────────────────────────────────────────────────────────────

def wrap_and_align(range_path: Path, colored_img: np.ndarray, fine_yaw: float = 0.0):
    """
    Reconstruct 3D from Cyberware range file using colored_img as texture.
    Returns (aligned PointCloud, raw pts, raw colors).
    """
    print(f"\n  [3D] Reconstructing from range file...")
    pts, colors, _, _, _ = range_to_3d(range_path, colored_img)
    print(f"  [3D] Valid points: {len(pts):,}")
    pcd = build_pcd(pts, colors)
    pcd = align_front(pcd, fine_yaw=fine_yaw)
    print(f"  [3D] Aligned: Rx=-90°, Ry=+90°" +
          (f", fine Ry={fine_yaw:+.1f}°" if fine_yaw != 0.0 else ""))
    return pcd, pts, colors


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_one_scan(scan_dir: Path, args):
    """
    scan_dir — e.g. .../PAT01/D00/PAT01_D00_A/
    Expects:
        PAT01_D00_A.tif     (texture)
        PAT01_D00_A         (Cyberware range file, no extension)
    Writes to args.output/<model>/<patient_lower>/:
        PAT01_D00_A_burn3d.ply
        PAT01_D00_A_front.png
        PAT01_D00_A_burn_mask.png      (--save-mask)
        PAT01_D00_A_burn_overlay.png   (--save-mask)
        PAT01_D00_A_burn_texture.png   (--save-mask)
    """
    scan_name  = scan_dir.name
    tif_path   = scan_dir / f"{scan_name}.tif"
    range_path = scan_dir / scan_name

    if not tif_path.exists():
        raise FileNotFoundError(f"TIF not found: {tif_path}")
    if not range_path.exists():
        raise FileNotFoundError(f"Range file not found: {range_path}")

    m          = SCAN_RE.match(scan_name)
    patient_id = m.group("patient").upper() if m else "UNKNOWN"
    timepoint  = m.group("timepoint").upper() if m else "?"

    output_dir = scan_dir   # save in-place inside the scan subfolder

    print(f"\n{'─'*60}")
    print(f"  Scan     : {scan_name}")
    print(f"  Patient  : {patient_id}  |  Timepoint: {timepoint}  |  Model: {args.model.upper()}")
    print(f"  Output   : {output_dir}  (in-place)")
    print(f"{'─'*60}")

    # Step 1 — load
    img_rgb = load_tiff(tif_path)

    # Step 2 — segment
    mask = run_segmentation(img_rgb, args)

    # Step 3 — colour (solid red on burn)
    colored_img = apply_burn_colors(img_rgb, mask)
    print(f"\n  [COLOR] Burn areas colored solid red on texture")

    # Optional: save mask + overlay + colored texture
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
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 255), 2)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        cv2.imwrite(str(colored_path), cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))

        print(f"  [SAVE] Mask    → {mask_path.name}")
        print(f"  [SAVE] Overlay → {overlay_path.name}")
        print(f"  [SAVE] Texture → {colored_path.name}")

    # Steps 4+5 — 3D
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
#  DISCOVER ALL SCAN SUBFOLDERS
# ─────────────────────────────────────────────────────────────────────────────

def discover_scans(dataset_dir: Path, patient_filter=None, timepoint_filter=None):
    """
    Walk:  dataset_dir / PAT* / [DM]* / PAT*_[DM]*_* /
    Return sorted list of valid scan Path objects.
    """
    scans = []
    for pat_dir in sorted(dataset_dir.iterdir()):
        if not pat_dir.is_dir():
            continue
        if patient_filter and pat_dir.name.upper() != patient_filter.upper():
            continue

        for tp_dir in sorted(pat_dir.iterdir(), key=lambda p: _tp_key(p.name)):
            if not tp_dir.is_dir():
                continue
            if timepoint_filter and tp_dir.name.upper() != timepoint_filter.upper():
                continue

            for scan_dir in sorted(tp_dir.iterdir()):
                if not scan_dir.is_dir():
                    continue
                if not SCAN_RE.match(scan_dir.name):
                    continue
                tif_path   = scan_dir / f"{scan_dir.name}.tif"
                range_path = scan_dir / scan_dir.name
                if tif_path.exists() and range_path.exists():
                    scans.append(scan_dir)
                else:
                    if not tif_path.exists():
                        print(f"  ⚠  TIF missing  : {tif_path}")
                    if not range_path.exists():
                        print(f"  ⚠  Range missing: {range_path}")
    return scans


# ─────────────────────────────────────────────────────────────────────────────
#  PARSE ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Burn segmentation → 3D point cloud pipeline")
    p.add_argument("--config",      default=None, help="Path to config.yaml")
    p.add_argument("--dataset",     default=None, help="Dataset root folder (batch mode)")
    p.add_argument("--patient",     default=None, help="Filter to one patient e.g. PAT01")
    p.add_argument("--timepoint",   default=None, help="Filter to one timepoint e.g. D00")
    p.add_argument("--scandir",     default=None, help="Single scan subfolder path")
    p.add_argument("--model",       default=None, choices=["unetpp", "segformer", "medsam", "sam2"])
    p.add_argument("--ckpt",        default=None, help=".pth file or folder (segformer)")
    p.add_argument("--base-ckpt",   default=None, help="Base weights for medsam/sam2")
    p.add_argument("--sam2-repo",   default=None, help="segment-anything-2 repo path")
    p.add_argument("--threshold",   type=float, default=None, help="Burn threshold (default 0.5)")
    p.add_argument("--output",      default=None, help="Output root directory")
    p.add_argument("--fine-yaw",    type=float, default=None, help="Fine yaw correction degrees")
    p.add_argument("--save-mask",   action="store_true", help="Save mask + overlay + texture PNGs")
    p.add_argument("--show",        action="store_true", help="Open 3D viewer after each scan")

    args = p.parse_args()

    # Merge config.yaml — CLI args always win
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if args.dataset    is None: args.dataset    = cfg.get("dataset")
        if args.patient    is None: args.patient    = cfg.get("patient")
        if args.timepoint  is None: args.timepoint  = cfg.get("timepoint")
        if args.scandir    is None: args.scandir    = cfg.get("scandir")
        if args.model      is None: args.model      = cfg.get("model")
        if args.ckpt       is None: args.ckpt       = cfg.get("ckpt")
        if args.base_ckpt  is None: args.base_ckpt  = cfg.get("base_ckpt")
        if args.sam2_repo  is None: args.sam2_repo  = cfg.get("sam2_repo")
        if args.threshold  is None: args.threshold  = cfg.get("threshold", 0.5)
        if args.output     is None: args.output     = cfg.get("output", "./outputs")
        if args.fine_yaw   is None: args.fine_yaw   = cfg.get("fine_yaw", 0.0)
        if not args.save_mask:      args.save_mask  = cfg.get("save_mask", False)
        if not args.show:           args.show       = cfg.get("show", False)

    # Defaults
    if args.threshold is None: args.threshold = 0.5
    if args.fine_yaw  is None: args.fine_yaw  = 0.0

    # scandir always beats dataset — single scan takes priority
    if args.scandir:
        args.dataset = None

    # Validate
    for field, name in [(args.model, "--model"), (args.ckpt, "--ckpt")]:
        if not field:
            p.error(f"{name} is required (pass as argument or set in config.yaml)")
    if not args.dataset and not args.scandir:
        p.error("Provide --dataset (batch) or --scandir (single scan)")

    return args


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Burn 3D Pipeline")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Outputs : saved in-place inside each scan subfolder")
    print(f"{'='*60}")

    if args.scandir:
        # ── Single scan mode ─────────────────────────────────────────────────
        print(f"  Mode    : SINGLE")
        process_one_scan(Path(args.scandir), args)

    else:
        # ── Batch mode ────────────────────────────────────────────────────────
        dataset_dir = Path(args.dataset)
        if not dataset_dir.exists():
            print(f"  Dataset not found: {dataset_dir}")
            sys.exit(1)

        scans = discover_scans(dataset_dir,
                               patient_filter=args.patient,
                               timepoint_filter=args.timepoint)
        if not scans:
            print("  No valid scans found.")
            sys.exit(1)

        print(f"  Mode    : BATCH  ({len(scans)} scan(s) found)")
        if args.patient:   print(f"  Patient  : {args.patient}")
        if args.timepoint: print(f"  Timepoint: {args.timepoint}")

        ok, failed = 0, []
        for scan_dir in scans:
            try:
                process_one_scan(scan_dir, args)
                ok += 1
            except Exception as e:
                print(f"  ✗ {scan_dir.name} failed: {e}")
                failed.append((scan_dir.name, str(e)))

        print(f"\n{'='*60}")
        print(f"  ALL DONE  —  {ok} succeeded, {len(failed)} failed")
        if failed:
            for name, err in failed:
                print(f"    ✗ {name}: {err}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()