#!/usr/bin/env python3
"""
step0b_apply_alignment.py
==========================
Reads each scan's:
  - <scan>_burn_polygons.json   (2D polygon in pixel space, from step 1)
  - <scan>_alignment.json       (R, t transform to reference frame, from step 0)
  - <scan>.tif + <scan>         (TIF texture + Cyberware range file)

For each burn region:
  1. Rasterises the 2D polygon -> pixel mask
  2. Finds which 3D points (from range_to_3d) project into that mask
  3. Computes area in real mm² using cylindrical surface-patch formula:
       area_mm2 = Σ radius_i × dθ × dz   (exact for a cylindrical scanner)
  4. Applies (R, t) to move those 3D points into the patient reference frame
     (PAT01_D00_A for PAT01)
  5. Computes the aligned 3D centroid

Writes <scan>_burn_polygons_aligned.json alongside the other scan outputs.
This file has the same structure as _burn_polygons.json PLUS:
  - area_mm2               (real surface area)
  - n_points_3d            (how many scanner points fell in this region)
  - aligned_centroid_xyz   ([x, y, z] in reference frame)
  - alignment_status       (ok / low_confidence / failed / no_alignment_file)

Step 3 (step3_track_progress.py) reads this file preferentially when present,
falling back to the original _burn_polygons.json for any scan that lacks it.

USAGE:
  # one patient, batch over all scan subfolders
  python step0b_apply_alignment.py --dataset D:/.../face_burn_dataset --patient PAT01

  # single scan
  python step0b_apply_alignment.py --scandir D:/.../PAT01/D00/PAT01_D00_A
"""

import sys
import json
import math
import argparse
import numpy as np
import cv2
import tifffile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "3.burn_segmentation_3d_pipeline"))
from utils.cyberware import range_to_3d, SCANNER_HEIGHT_MM


def load_tiff_rgb(tif_path: Path) -> np.ndarray:
    raw = tifffile.imread(str(tif_path))
    if raw.ndim == 3 and raw.shape[0] < 10:
        raw = raw[0]
    raw = raw.astype(np.float32)
    lo, hi = raw.min(), raw.max()
    raw = ((raw - lo) / (hi - lo) * 255.0).astype(np.uint8) if hi > lo else raw.astype(np.uint8)
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=-1)
    return raw[..., :3]


def polygon_to_mask(polygon_pts: list, shape: tuple) -> np.ndarray:
    """Rasterise a list of [x, y] points into a binary HxW mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cnt = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [cnt], 1)
    return mask.astype(bool)


def compute_region_3d(polygon_pts, pts, tif_row, tif_col, valid_mask,
                      R, t, img_shape):
    """
    For one burn region:
      - find the 3D points whose (tif_row, tif_col) falls inside the polygon
      - compute area_mm2 from the cylindrical patch formula (before transform)
      - apply (R, t) to get aligned 3D positions
      - return area_mm2, aligned_centroid_xyz, n_points_3d
    """
    poly_mask = polygon_to_mask(polygon_pts, img_shape)
    in_region = poly_mask[tif_row, tif_col]
    n3d = int(in_region.sum())

    if n3d == 0:
        return None, None, 0

    region_pts = pts[in_region]

    NLG, NLT = valid_mask.shape
    dtheta = 2.0 * math.pi / NLG
    dz = SCANNER_HEIGHT_MM / NLT

    radii = np.sqrt(region_pts[:, 0] ** 2 + region_pts[:, 1] ** 2)
    area_mm2 = float(np.sum(radii * dtheta * dz))

    if R is not None and t is not None:
        R_arr = np.array(R)
        t_arr = np.array(t)
        aligned_pts = (R_arr @ region_pts.T).T + t_arr
    else:
        aligned_pts = region_pts

    centroid = np.median(aligned_pts, axis=0).tolist()
    return area_mm2, centroid, n3d


def process_scan(scan_dir: Path) -> dict:
    name = scan_dir.name
    tif_path = scan_dir / f"{name}.tif"
    range_path = scan_dir / name
    poly_path = scan_dir / f"{name}_burn_polygons.json"
    align_path = scan_dir / f"{name}_alignment.json"

    if not poly_path.exists():
        return {"status": "skipped", "reason": "no _burn_polygons.json"}
    if not tif_path.exists():
        return {"status": "skipped", "reason": "no .tif"}
    if not range_path.exists():
        return {"status": "skipped", "reason": "no range file"}

    poly_data = json.load(open(poly_path))
    img_shape = (poly_data["image_size"]["height"],
                 poly_data["image_size"]["width"])

    if align_path.exists():
        align = json.load(open(align_path))
        align_status = align.get("status", "unknown")
        R = align.get("R")
        t = align.get("t")
        if align_status == "failed" or R is None:
            R, t = None, None
    else:
        align_status = "no_alignment_file"
        R, t = None, None

    img_rgb = load_tiff_rgb(tif_path)
    pts, colors, tif_row, tif_col, valid_mask = range_to_3d(range_path, img_rgb)

    regions_out = []
    for region in poly_data.get("regions", []):
        area_mm2, centroid, n3d = compute_region_3d(
            region["polygon"], pts, tif_row, tif_col, valid_mask,
            R, t, img_shape
        )
        r_out = dict(region)
        r_out["area_mm2"] = round(area_mm2, 2) if area_mm2 is not None else None
        r_out["n_points_3d"] = n3d
        r_out["aligned_centroid_xyz"] = (
            [round(v, 3) for v in centroid] if centroid is not None else None
        )
        regions_out.append(r_out)

    total_mm2 = sum(r["area_mm2"] or 0 for r in regions_out)

    out = dict(poly_data)
    out["alignment_status"] = align_status
    out["total_burn_area_mm2"] = round(total_mm2, 2)
    out["regions"] = regions_out

    out_path = scan_dir / f"{name}_burn_polygons_aligned.json"
    json.dump(out, open(out_path, "w"), indent=2)
    return {"status": "ok", "total_mm2": total_mm2,
            "n_regions": len(regions_out), "align_status": align_status}


def batch(dataset_dir: Path, patient: str):
    pat_dir = dataset_dir / patient
    scan_dirs = sorted([p for p in pat_dir.glob("*/*") if p.is_dir()])
    print(f"  Found {len(scan_dirs)} scan folder(s) for {patient}\n")

    ok, skipped, failed = 0, 0, []
    for sd in scan_dirs:
        try:
            r = process_scan(sd)
            if r["status"] == "ok":
                align_sym = {"ok": "✓", "low_confidence": "⚠",
                             "failed": "✗", "no_alignment_file": "–"}.get(
                    r["align_status"], "?")
                print(f"  {align_sym} {sd.name}: "
                      f"{r['n_regions']} region(s), "
                      f"area={r['total_mm2']:.1f}mm²  "
                      f"(align={r['align_status']})")
                ok += 1
            else:
                print(f"  – {sd.name}: skipped ({r['reason']})")
                skipped += 1
        except Exception as e:
            print(f"  ✗ {sd.name}: ERROR — {e}")
            failed.append((sd.name, str(e)))

    print(f"\n  DONE — {ok} processed, {skipped} skipped, {len(failed)} errors")
    for name, err in failed:
        print(f"    ✗ {name}: {err}")


def main():
    p = argparse.ArgumentParser(description="Step 0b — apply alignment to burn polygons")
    p.add_argument("--dataset", default=None)
    p.add_argument("--patient", default=None)
    p.add_argument("--scandir", default=None)
    args = p.parse_args()

    if args.scandir:
        sd = Path(args.scandir)
        r = process_scan(sd)
        if r["status"] == "ok":
            print(f"  ✓ {sd.name}: {r['n_regions']} region(s), "
                  f"area={r['total_mm2']:.1f}mm²  (align={r['align_status']})")
        else:
            print(f"  – {sd.name}: {r['reason']}")
    elif args.dataset and args.patient:
        batch(Path(args.dataset), args.patient)
    else:
        p.error("Provide --scandir, or --dataset + --patient")


if __name__ == "__main__":
    main()