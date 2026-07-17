#!/usr/bin/env python3
"""
burn_3d_pipeline.py
===================
Full pipeline: Unwrapped 2D TIFF  →  Burn Segmentation  →  3D Point Cloud

USAGE:
  # Batch mode (all patients/scans):
  python burn_3d_pipeline.py --config config.yaml

  # Single scan:
  python burn_3d_pipeline.py --config config.yaml --range "..." --tif "..."

REQUIREMENTS:
  pip install numpy pillow open3d opencv-python pyyaml
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

sys.path.insert(0, str(Path(__file__).parent))

from utils.cyberware import range_to_3d
from utils.alignment  import build_pcd, align_front
from utils.writers    import write_ply, render_snapshot


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BURN_TINT      = np.array([220, 50,  50],  dtype=np.float32)  # red tint
BOUNDARY_COLOR = np.array([255,  255, 0], dtype=np.uint8)    # cyan boundary
BLEND_ALPHA    = 0.5                                           # 50% skin + 50% tint
BOUNDARY_WIDTH = 1                                             # px
SCAN_RE        = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD TIFF
# ─────────────────────────────────────────────────────────────────────────────

def load_tiff(tif_path: Path) -> np.ndarray:
    img = Image.open(tif_path).convert("RGB")
    arr = np.array(img)
    print(f"  [TIF] {tif_path.name}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(img_rgb: np.ndarray, args) -> np.ndarray:
    model_name = args.model.lower()
    print(f"\n  [SEG] Model: {model_name.upper()}  threshold={args.threshold}")

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
#  STEP 3 — APPLY BURN COLORS + POLYGONS
# ─────────────────────────────────────────────────────────────────────────────

def get_burn_polygons(mask: np.ndarray):
    contours, _ = cv2.findContours(
        (mask > 128).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) > 200]


def apply_burn_colors(img_rgb: np.ndarray, mask: np.ndarray):
    """
    Burn region  → skin color blended 50% with red tint (texture still visible)
    Burn boundary → bright cyan outline
    Normal skin  → unchanged
    Returns (colored_img, contours)
    """
    colored = img_rgb.copy().astype(np.float32)
    burn_px = mask > 128

    # Blend burn region
    colored[burn_px] = (
        colored[burn_px] * (1.0 - BLEND_ALPHA) +
        BURN_TINT         * BLEND_ALPHA
    )
    colored = colored.clip(0, 255).astype(np.uint8)

    # Draw cyan boundary
    contours = get_burn_polygons(mask)
    cv2.drawContours(colored, contours, -1, BOUNDARY_COLOR.tolist(), BOUNDARY_WIDTH)

    return colored, contours


def polygons_to_json(contours, scan_name: str, tif_shape: tuple) -> dict:
    regions = []
    for i, cnt in enumerate(contours):
        pts = cnt.squeeze()
        regions.append({
            "region_id"  : i + 1,
            "area_pixels": int(cv2.contourArea(cnt)),
            "polygon"    : pts.tolist() if pts.ndim == 2 else [pts.tolist()],
        })
    return {
        "scan"        : scan_name,
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
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_one_scan(range_path: Path, tif_path: Path, args):
    scan_name  = range_path.stem
    m          = SCAN_RE.match(scan_name)
    patient_id = m.group("patient").lower() if m else "unknown"

    output_dir = Path(args.output) / args.model / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Scan: {scan_name}  |  Patient: {patient_id.upper()}  |  Model: {args.model.upper()}")
    print(f"  Output: {output_dir}")
    print(f"{'─'*60}")

    # Step 1
    img_rgb = load_tiff(tif_path)

    # Step 2
    mask = run_segmentation(img_rgb, args)

    # Step 3
    colored_img, contours = apply_burn_colors(img_rgb, mask)
    print(f"\n  [COLOR] Burn regions found: {len(contours)}")

    # Save polygon JSON (always)
    json_data = polygons_to_json(contours, scan_name, img_rgb.shape)
    json_path = output_dir / f"{scan_name}_burn_polygons.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  [SAVE] Polygons → {json_path.name}  ({len(contours)} region(s))")

    # Optional: save 2D mask + colored texture
    if args.save_mask:
        mask_path    = output_dir / f"{scan_name}_burn_mask.png"
        colored_path = output_dir / f"{scan_name}_burn_texture.png"
        cv2.imwrite(str(mask_path),    mask)
        cv2.imwrite(str(colored_path), cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE] Mask    → {mask_path.name}")
        print(f"  [SAVE] Texture → {colored_path.name}")

    # Step 4+5
    pcd = wrap_and_align(range_path, colored_img, fine_yaw=args.fine_yaw)       # Replace the original pixel color with red/yellow
    # pcd = wrap_and_align(range_path, img_rgb, fine_yaw=args.fine_yaw)           # Have the original pixel color  


    # Save PLY + snapshot
    ply_path = output_dir / f"{scan_name}_burn3d.ply"
    png_path = output_dir / f"{scan_name}_front.png"
    pts_out    = np.asarray(pcd.points).astype(np.float32)
    colors_out = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    sz = write_ply(pts_out, colors_out, ply_path)
    render_snapshot(pcd, png_path)

    print(f"\n  [OUT] PLY      → {ply_path.name}  ({sz/1e6:.1f} MB)")
    print(f"  [OUT] Snapshot → {png_path.name}")
    print(f"  [OUT] Polygons → {json_path.name}")

    if args.show:
        o3d.visualization.draw_geometries(
            [pcd], window_name=f"{scan_name} — burn segmentation",
            width=900, height=700)


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVER ALL SCANS IN DATASET FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def discover_scans(dataset_dir: Path, patient_filter=None):
    scans = []
    for scan_folder in sorted(dataset_dir.glob("*/*/")):
        scan_name = scan_folder.name
        if not SCAN_RE.match(scan_name):
            continue
        pid = SCAN_RE.match(scan_name).group("patient").lower()
        if patient_filter and pid != patient_filter.lower():
            continue
        range_path = scan_folder / scan_name
        tif_path   = scan_folder / f"{scan_name}.tif"
        if range_path.exists() and tif_path.exists():
            scans.append((range_path, tif_path))
        else:
            if not range_path.exists():
                print(f"  ⚠ Range missing: {range_path}")
            if not tif_path.exists():
                print(f"  ⚠ TIF missing: {tif_path}")
    return sorted(scans, key=lambda x: (x[0].parent.parent.name, x[0].name))


# ─────────────────────────────────────────────────────────────────────────────
#  PARSE ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Burn segmentation → 3D point cloud pipeline")
    p.add_argument("--config",    default=None,  help="Path to config.yaml")
    p.add_argument("--dataset",   default=None,  help="Dataset root folder — batch mode (all patients)")
    p.add_argument("--patient",   default=None,  help="Filter to one patient e.g. pat1 (batch mode only)")
    p.add_argument("--range",     default=None,  help="Cyberware range file path (single mode)")
    p.add_argument("--tif",       default=None,  help="TIFF image path (single mode)")
    p.add_argument("--model",     default=None,  choices=["unetpp", "segformer", "medsam", "sam2"])
    p.add_argument("--ckpt",      default=None,  help=".pth file or folder (segformer)")
    p.add_argument("--base-ckpt", default=None,  help="Base weights for medsam/sam2")
    p.add_argument("--sam2-repo", default=None,  help="segment-anything-2 repo path")
    p.add_argument("--threshold", type=float, default=None, help="Burn threshold (default 0.5)")
    p.add_argument("--output",    default=None,  help="Output root directory")
    p.add_argument("--fine-yaw",  type=float, default=None, help="Fine yaw correction degrees")
    p.add_argument("--save-mask", action="store_true", help="Save 2D mask + texture images")
    p.add_argument("--show",      action="store_true", help="Open 3D viewer after each scan")

    args = p.parse_args()

    # Load config file — CLI args override config values
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        if args.dataset   is None: args.dataset   = cfg.get("dataset")
        if args.patient   is None: args.patient   = cfg.get("patient")
        if args.range     is None: args.range     = cfg.get("range")
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

    # Defaults
    if args.threshold is None: args.threshold = 0.5
    if args.output    is None: args.output    = "./outputs"
    if args.fine_yaw  is None: args.fine_yaw  = 0.0

    # Validate
    is_batch = bool(args.dataset)
    for field, name in [(args.model, "--model"), (args.ckpt, "--ckpt")]:
        if not field:
            p.error(f"{name} is required (set in config.yaml or pass as argument)")
    if not is_batch:
        for field, name in [(args.range, "--range"), (args.tif, "--tif")]:
            if not field:
                p.error(f"{name} is required in single mode (or set dataset: in config.yaml for batch mode)")

    return args


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Burn 3D Pipeline")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Output  : {args.output}")
    print(f"{'='*60}")

    if args.dataset:
        # ── Batch mode ────────────────────────────────────────────────────────
        dataset_dir = Path(args.dataset)
        if not dataset_dir.exists():
            print(f"  Dataset not found: {dataset_dir}")
            sys.exit(1)

        scans = discover_scans(dataset_dir, patient_filter=args.patient)
        if not scans:
            print("  No valid scans found.")
            sys.exit(1)

        print(f"  Mode    : BATCH  ({len(scans)} scan(s) found)")
        if args.patient:
            print(f"  Patient : {args.patient}")

        ok, failed = 0, []
        for range_path, tif_path in scans:
            try:
                process_one_scan(range_path, tif_path, args)
                ok += 1
            except Exception as e:
                print(f"  ✗ {range_path.stem} failed: {e}")
                failed.append((range_path.stem, str(e)))

        print(f"\n{'='*60}")
        print(f"  ALL DONE  —  {ok} succeeded, {len(failed)} failed")
        if failed:
            for name, err in failed:
                print(f"    ✗ {name}: {err}")
        print(f"{'='*60}\n")

    else:
        # ── Single mode ───────────────────────────────────────────────────────
        print(f"  Mode    : SINGLE")
        process_one_scan(Path(args.range), Path(args.tif), args)
        print(f"\n{'='*60}")
        print(f"  DONE")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()