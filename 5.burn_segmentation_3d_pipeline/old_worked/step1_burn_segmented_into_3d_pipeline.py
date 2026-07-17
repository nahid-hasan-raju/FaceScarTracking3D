#!/usr/bin/env python3
"""
burn_3d_pipeline.py
===================
Full pipeline: Unwrapped 2D TIFF  →  Burn Segmentation  →  3D Point Cloud

Dataset folder structure expected:
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

Outputs saved IN-PLACE inside each scan subfolder:
  PAT01_D00_A_seg.tif            ← segmented texture (burn region coloured)
  PAT01_D00_A_burn3d.ply         ← 3-D point cloud with burn colours
  PAT01_D00_A_burn_polygons.json ← burn region polygons (always)
  PAT01_D00_A_burn_mask.png      ← binary mask (--save-mask only)

USAGE:
  # Batch — all patients:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml --dataset D:/NahidW/Dataset/face_burn_dataset

  # Batch — one patient:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml --dataset D:/... --patient PAT01

  # Single scan subfolder:
  python step1_burn_segmented_into_3d_pipeline.py --config config.yaml --scandir D:/.../PAT01/D00/PAT01_D00_A

REQUIREMENTS:
  pip install numpy pillow open3d opencv-python pyyaml imagecodecs tifffile
  + model-specific: see README.md
"""

import sys
import re
import json
import argparse
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from PIL import Image
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

from utils.cyberware import range_to_3d
from utils.alignment  import build_pcd, align_front
from utils.writers    import write_ply, render_snapshot


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BURN_TINT      = np.array([255, 30,  30],  dtype=np.float32)   # vivid red tint
BOUNDARY_COLOR = np.array([255, 255,  0],  dtype=np.uint8)     # yellow boundary
BLEND_ALPHA    = 0.55                                            # 55% tint + 45% skin
BOUNDARY_WIDTH = 1                                               # px — wide enough on high-res TIF

# Matches scan subfolder names like  PAT01_D00_A  or  PAT03_M06_B2
SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)

# Sort timepoints D00 < D14 < D28 < M01 … M24
def _tp_key(name):
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

    print(f"  [TIF] {tif_path.name}  shape={raw.shape}  dtype={raw.dtype}")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(img_rgb: np.ndarray, args):
    """
    Returns (mask, prob_map):
      mask     — uint8 binary mask (0/255), same as before
      prob_map — float32 array, per-pixel probability in [0, 1], same H×W as mask
    """
    model_name = args.model.lower()
    print(f"\n  [SEG] Model: {model_name.upper()}  threshold={args.threshold}")

    if model_name == "unetpp":
        from utils.models.unetpp import load_model, predict
        model = load_model(args.ckpt)
        mask, prob_map = predict(model, img_rgb, threshold=args.threshold)

    elif model_name == "segformer":
        from utils.models.segformer import load_model, predict
        model, processor = load_model(args.ckpt)
        mask, prob_map = predict(model, processor, img_rgb, threshold=args.threshold)

    elif model_name == "medsam":
        if not args.base_ckpt:
            raise ValueError("--base-ckpt required for MedSAM")
        from utils.models.medsam import load_model, predict
        model, predictor = load_model(args.base_ckpt, args.ckpt)
        mask, prob_map = predict(model, predictor, img_rgb, threshold=args.threshold)

    elif model_name == "sam2":
        if not args.base_ckpt:
            raise ValueError("--base-ckpt required for SAM2")
        if not args.sam2_repo:
            raise ValueError("--sam2-repo required for SAM2")
        from utils.models.sam2 import load_model, predict
        model, predictor = load_model(args.base_ckpt, args.ckpt, args.sam2_repo)
        mask, prob_map = predict(model, predictor, img_rgb, threshold=args.threshold)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    burn_px = int((mask > 128).sum())
    pct     = 100.0 * burn_px / mask.size
    print(f"  [SEG] Burn pixels: {burn_px:,} / {mask.size:,}  ({pct:.1f}%)")
    return mask, prob_map


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — APPLY BURN COLOURS + POLYGONS
# ─────────────────────────────────────────────────────────────────────────────

def get_burn_polygons(mask: np.ndarray):
    contours, _ = cv2.findContours(
        (mask > 128).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) > 50]


def apply_burn_colors(img_rgb: np.ndarray, mask: np.ndarray):
    """
    Burn region  → skin colour blended with vivid red tint (texture still visible)
    Burn boundary → bright yellow outline drawn LAST so it is never overwritten
    Normal skin  → unchanged
    Returns (coloured_img uint8 RGB, contours)
    """
    # Step A: blend burn region — original skin fades toward red
    colored = img_rgb.copy().astype(np.float32)
    burn_px = mask > 128
    colored[burn_px] = (
        colored[burn_px] * (1.0 - BLEND_ALPHA) +
        BURN_TINT         * BLEND_ALPHA
    )
    colored = colored.clip(0, 255).astype(np.uint8)

    # Step B: find contours on the raw binary mask
    contours = get_burn_polygons(mask)

    # Step C: draw boundary AFTER the blend is locked in
    # The array stays in RGB order throughout — pass (R, G, B) directly
    boundary_rgb = (int(BOUNDARY_COLOR[0]), int(BOUNDARY_COLOR[1]), int(BOUNDARY_COLOR[2]))
    cv2.drawContours(colored, contours, -1, boundary_rgb, BOUNDARY_WIDTH)

    return colored, contours


def polygons_to_json(contours, scan_name: str, tif_shape: tuple, prob_map: np.ndarray,
                      patient_id: str = None, timepoint: str = None,
                      model_name: str = None, threshold: float = None) -> dict:
    regions = []
    for i, cnt in enumerate(contours):
        pts = cnt.squeeze()

        # Per-region confidence: mean/min/max of the probability map inside this contour
        region_mask = np.zeros(prob_map.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, 1, thickness=cv2.FILLED)
        region_probs = prob_map[region_mask.astype(bool)]
        if region_probs.size > 0:
            confidence_mean = float(region_probs.mean())
            confidence_min  = float(region_probs.min())
            confidence_max  = float(region_probs.max())
        else:
            confidence_mean = confidence_min = confidence_max = None

        regions.append({
            "region_id"      : i + 1,
            "area_pixels"    : int(cv2.contourArea(cnt)),
            "confidence_mean": confidence_mean,
            "confidence_min" : confidence_min,
            "confidence_max" : confidence_max,
            "polygon"        : pts.tolist() if pts.ndim == 2 else [pts.tolist()],
        })
    return {
        "scan"        : scan_name,
        "patient_id"  : patient_id,
        "timepoint"   : timepoint,
        "model"       : model_name,
        "threshold"   : threshold,
        "image_size"  : {"height": tif_shape[0], "width": tif_shape[1]},
        "burn_regions": len(regions),
        "regions"     : regions,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4+5 — WRAP TO 3D + ALIGN
# ─────────────────────────────────────────────────────────────────────────────

def wrap_and_align(range_path: Path, colored_img: np.ndarray, fine_yaw: float = 0.0):
    print(f"\n  [3D] Reconstructing from range file...")
    pts, colors, _, _, _ = range_to_3d(range_path, colored_img)
    print(f"  [3D] Valid points: {len(pts):,}")
    pcd = build_pcd(pts, colors)
    pcd = align_front(pcd, fine_yaw=fine_yaw)
    print(f"  [3D] Aligned: Rx=-90°, Ry=+90°" +
          (f"  fine Ry={fine_yaw:+.1f}°" if fine_yaw != 0.0 else ""))
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN SUBFOLDER
# ─────────────────────────────────────────────────────────────────────────────

def process_one_scan(scan_dir: Path, args):
    """
    scan_dir  — e.g. .../PAT01/D00/PAT01_D00_A/
    Expects inside it:
        PAT01_D00_A.tif      (texture)
        PAT01_D00_A          (Cyberware range file, no extension)
    Writes back into scan_dir:
        PAT01_D00_A_seg.tif
        PAT01_D00_A_burn3d.ply
        PAT01_D00_A_burn_polygons.json
        PAT01_D00_A_burn_mask.png   (if --save-mask)
    """
    scan_name  = scan_dir.name
    tif_path   = scan_dir / f"{scan_name}.tif"
    range_path = scan_dir / scan_name          # no extension

    if not tif_path.exists():
        raise FileNotFoundError(f"TIF not found: {tif_path}")
    if not range_path.exists():
        raise FileNotFoundError(f"Range file not found: {range_path}")

    # Parse patient / timepoint from folder name
    m          = SCAN_RE.match(scan_name)
    patient_id = m.group("patient").upper() if m else "UNKNOWN"
    timepoint  = m.group("timepoint").upper() if m else "?"

    print(f"\n{'─'*60}")
    print(f"  Scan     : {scan_name}")
    print(f"  Patient  : {patient_id}  |  Timepoint: {timepoint}  |  Model: {args.model.upper()}")
    print(f"  Output   : {scan_dir}  (in-place)")
    print(f"{'─'*60}")

    # Step 1 — load
    img_rgb = load_tiff(tif_path)

    # Step 2 — segment
    mask, prob_map = run_segmentation(img_rgb, args)

    # Step 3 — colour
    colored_img, contours = apply_burn_colors(img_rgb, mask)
    print(f"\n  [COLOR] Burn regions found: {len(contours)}")

    # ── Save segmented TIFF (in-place) ───────────────────────────────────────
    seg_tif_path = scan_dir / f"{scan_name}_seg.tif"
    tifffile.imwrite(str(seg_tif_path), colored_img)
    print(f"  [SAVE] Seg TIF  → {seg_tif_path.name}")

    # ── Save polygon JSON (always) ────────────────────────────────────────────
    json_data = polygons_to_json(
        contours, scan_name, img_rgb.shape, prob_map,
        patient_id=patient_id, timepoint=timepoint,
        model_name=args.model, threshold=args.threshold,
    )
    json_path = scan_dir / f"{scan_name}_burn_polygons.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [SAVE] Polygons → {json_path.name}  ({len(contours)} region(s))")

    # ── Optional: binary mask PNG ─────────────────────────────────────────────
    if args.save_mask:
        mask_path = scan_dir / f"{scan_name}_burn_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"  [SAVE] Mask     → {mask_path.name}")

    # Step 4+5 — 3D reconstruction with burn colours baked in
    pcd = wrap_and_align(range_path, colored_img, fine_yaw=args.fine_yaw)

    # ── Save PLY + snapshot (in-place) ────────────────────────────────────────
    ply_path = scan_dir / f"{scan_name}_burn3d.ply"
    png_path = scan_dir / f"{scan_name}_front.png"
    pts_out    = np.asarray(pcd.points).astype(np.float32)
    colors_out = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    sz = write_ply(pts_out, colors_out, ply_path)
    render_snapshot(pcd, png_path)

    print(f"\n  [OUT] PLY       → {ply_path.name}  ({sz/1e6:.1f} MB)")
    print(f"  [OUT] Snapshot  → {png_path.name}")

    if args.show:
        o3d.visualization.draw_geometries(
            [pcd], window_name=f"{scan_name} — burn segmentation",
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
    p.add_argument("--fine-yaw",    type=float, default=None, help="Fine yaw correction degrees")
    p.add_argument("--save-mask",   action="store_true", help="Save binary burn mask PNG")
    p.add_argument("--show",        action="store_true", help="Open 3D viewer after each scan")

    args = p.parse_args()

    # Merge config.yaml — only model/checkpoint/parameter settings, NEVER run-mode paths
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        # ↓ dataset / patient / timepoint / scandir intentionally excluded
        if args.model      is None: args.model      = cfg.get("model")
        if args.ckpt       is None: args.ckpt       = cfg.get("ckpt")
        if args.base_ckpt  is None: args.base_ckpt  = cfg.get("base_ckpt")
        if args.sam2_repo  is None: args.sam2_repo  = cfg.get("sam2_repo")
        if args.threshold  is None: args.threshold  = cfg.get("threshold", 0.5)
        if args.fine_yaw   is None: args.fine_yaw   = cfg.get("fine_yaw", 0.0)
        if not args.save_mask:      args.save_mask  = cfg.get("save_mask", False)
        if not args.show:           args.show       = cfg.get("show", False)

    # Defaults
    if args.threshold is None: args.threshold = 0.5
    if args.fine_yaw  is None: args.fine_yaw  = 0.0

    # --scandir always wins over dataset — single scan takes priority
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

    # DEBUG
    print(f"  DEBUG dataset  : '{args.dataset}'")
    print(f"  DEBUG scandir  : '{args.scandir}'")
    print(f"  DEBUG patient  : '{args.patient}'")
    print(f"  DEBUG model    : '{args.model}'")
    print(f"  DEBUG config   : '{args.config}'")
    # END DEBUG

    print(f"\n{'='*60}")
    print(f"  Burn 3D Pipeline")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Outputs : saved in-place inside each scan subfolder")
    print(f"{'='*60}")

    if args.dataset:
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
        print(f"{'='*60}\n")

    else:
        # ── Single scan mode ─────────────────────────────────────────────────
        print(f"  Mode    : SINGLE")
        process_one_scan(Path(args.scandir), args)
        print(f"\n{'='*60}")
        print(f"  DONE")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()