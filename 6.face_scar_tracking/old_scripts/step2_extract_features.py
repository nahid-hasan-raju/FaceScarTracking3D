#!/usr/bin/env python3
"""
step2_extract_features.py
==========================
Reads each scan's <scan>_burn_polygons.json (produced by step 1) and computes
a richer per-scan feature file: <scan>_scan_features.json, saved next to it.

Adds, per region:-
  - area_pixels, area_pct_of_image
  - perimeter_pixels
  - compactness          (4*pi*Area / Perimeter^2 -- 1.0 = perfect circle,
                           lower = more irregular border)
  - centroid_xy
  - bbox  (x, y, w, h)
  - confidence_mean / min / max   (carried over from step 1, if present)

Adds, per scan:
  - total_burn_area_pixels, total_burn_area_pct_of_image
  - num_regions
  - confidence_mean_weighted   (area-weighted across regions)
  - confidence_min / confidence_max   (global across regions)
  - low_confidence_regions     (count of regions whose confidence_mean
                                 sits within LOW_CONF_MARGIN of the
                                 segmentation threshold -- a QC flag,
                                 not a tracking decision)

USAGE:
  # one patient, batch over every scan subfolder found under it
  python step2_extract_features.py --dataset D:/.../face_burn_dataset --patient PAT01

  # single scan
  python step2_extract_features.py --polygons D:/.../PAT01_D00_A_burn_polygons.json
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path

LOW_CONF_MARGIN = 0.03   # flag a region if confidence_mean is within this of threshold


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def region_geometry(polygon_pts: list) -> dict:
    """polygon_pts: list of [x, y] ints, as stored in the polygon JSON."""
    cnt = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)

    area      = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, closed=True))
    compactness = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else None

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        pts_arr = np.array(polygon_pts, dtype=np.float64)
        cx, cy = pts_arr[:, 0].mean(), pts_arr[:, 1].mean()

    x, y, w, h = cv2.boundingRect(cnt)

    return {
        "perimeter_pixels": round(perimeter, 2),
        "compactness"     : round(compactness, 4) if compactness is not None else None,
        "centroid_xy"     : [round(cx, 2), round(cy, 2)],
        "bbox_xywh"       : [int(x), int(y), int(w), int(h)],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PER-SCAN FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_scan_features(polygons_json_path: Path) -> dict:
    with open(polygons_json_path, "r") as f:
        data = json.load(f)

    img_h = data["image_size"]["height"]
    img_w = data["image_size"]["width"]
    img_area = img_h * img_w
    threshold = data.get("threshold")

    regions_out = []
    total_area = 0
    conf_means, conf_mins, conf_maxs, conf_weights = [], [], [], []
    low_conf_count = 0

    for region in data.get("regions", []):
        area_px = region["area_pixels"]
        total_area += area_px

        geom = region_geometry(region["polygon"])

        conf_mean = region.get("confidence_mean")
        conf_min  = region.get("confidence_min")
        conf_max  = region.get("confidence_max")

        if conf_mean is not None:
            conf_means.append(conf_mean)
            conf_weights.append(area_px)
        if conf_min is not None:
            conf_mins.append(conf_min)
        if conf_max is not None:
            conf_maxs.append(conf_max)
        if conf_mean is not None and threshold is not None and (conf_mean - threshold) <= LOW_CONF_MARGIN:
            low_conf_count += 1

        regions_out.append({
            "region_id"        : region["region_id"],
            "area_pixels"      : area_px,
            "area_pct_of_image": round(100.0 * area_px / img_area, 4),
            "confidence_mean"  : conf_mean,
            "confidence_min"   : conf_min,
            "confidence_max"   : conf_max,
            **geom,
        })

    scan_confidence_mean_weighted = (
        float(np.average(conf_means, weights=conf_weights))
        if conf_means else None
    )

    return {
        "scan"        : data.get("scan"),
        "patient_id"  : data.get("patient_id"),
        "timepoint"   : data.get("timepoint"),
        "variant"     : (data.get("scan") or "").split("_")[-1] if data.get("scan") else None,
        "model"       : data.get("model"),
        "threshold"   : threshold,
        "image_size"  : {"height": img_h, "width": img_w},
        "num_regions" : len(regions_out),
        "total_burn_area_pixels"      : total_area,
        "total_burn_area_pct_of_image": round(100.0 * total_area / img_area, 4),
        "confidence_mean_weighted"    : (round(scan_confidence_mean_weighted, 4)
                                          if scan_confidence_mean_weighted is not None else None),
        "confidence_min_global"       : (round(min(conf_mins), 4) if conf_mins else None),
        "confidence_max_global"       : (round(max(conf_maxs), 4) if conf_maxs else None),
        "low_confidence_region_count" : low_conf_count,
        "regions"     : regions_out,
    }


def process_polygons_file(polygons_json_path: Path):
    features = extract_scan_features(polygons_json_path)
    out_path = polygons_json_path.parent / (
        polygons_json_path.name.replace("_burn_polygons.json", "_scan_features.json")
    )
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)
    print(f"  ✓ {polygons_json_path.name}  →  {out_path.name}"
          f"   (regions={features['num_regions']}, "
          f"area={features['total_burn_area_pixels']}px, "
          f"conf={features['confidence_mean_weighted']})")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVER + BATCH
# ─────────────────────────────────────────────────────────────────────────────

def discover_polygon_files(dataset_dir: Path, patient: str):
    pat_dir = dataset_dir / patient
    if not pat_dir.exists():
        raise FileNotFoundError(f"Patient folder not found: {pat_dir}")
    return sorted(pat_dir.glob("*/*/*_burn_polygons.json"))


def main():
    p = argparse.ArgumentParser(description="Step 2 — per-scan feature extraction")
    p.add_argument("--dataset",  default=None, help="Dataset root (batch mode)")
    p.add_argument("--patient",  default=None, help="Patient ID, e.g. PAT01 (used with --dataset)")
    p.add_argument("--polygons", default=None, help="Path to a single *_burn_polygons.json file")
    args = p.parse_args()

    if args.polygons:
        process_polygons_file(Path(args.polygons))
        return

    if not args.dataset or not args.patient:
        p.error("Provide --polygons for a single file, or --dataset + --patient for batch mode")

    files = discover_polygon_files(Path(args.dataset), args.patient)
    if not files:
        print(f"  No *_burn_polygons.json files found for {args.patient}")
        return

    print(f"  Found {len(files)} scan(s) for {args.patient}\n")
    ok, failed = 0, []
    for f in files:
        try:
            process_polygons_file(f)
            ok += 1
        except Exception as e:
            print(f"  ✗ {f.name} failed: {e}")
            failed.append((f.name, str(e)))

    print(f"\n  DONE — {ok} succeeded, {len(failed)} failed")
    for name, err in failed:
        print(f"    ✗ {name}: {err}")


if __name__ == "__main__":
    main()
