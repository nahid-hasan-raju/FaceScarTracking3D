#!/usr/bin/env python3
"""
step3_track_progress.py
=========================
Tracks burn regions over time, per camera-angle "variant" (A, B, C, D, ...),
for one patient. Reads the *_burn_polygons.json files produced by step 1
(must include confidence_mean/min/max — i.e. run after the step-1 patch).

MATCHING STRATEGY (region_t  <->  region_{t-1}):
  1. Rasterize each region's polygon into a binary mask (same image_size).
  2. Score every (prev_region, curr_region) pair by IoU.
  3. Greedily assign pairs with IoU > MIN_IOU, highest IoU first.
  4. For anything left unmatched, fall back to centroid distance:
     assign greedily if distance < MAX_CENTROID_DIST, closest first.
  5. Anything still unmatched:
       - a curr region with no match  -> new track ("first_seen")
       - a prev region with no match  -> track has a gap this scan
         ("not_detected" -- could mean healed/below-threshold, or a
         genuine tracking miss; check the match log)

OUTPUT (per variant, e.g. PAT01_A):
  <patient>_<variant>_tracking.json
    - tracks: per-track time series (area, area_pct, confidence,
              compactness, elapsed_days, pct_change_from_baseline,
              pct_change_from_previous)
    - match_log: every pairwise match decision + score, for manual QC

USAGE:
  python step3_track_progress.py --dataset D:/.../face_burn_dataset --patient PAT01
  # writes one tracking JSON per variant found (A, B, C, D, ...)

  # or test/inspect a single variant only:
  python step3_track_progress.py --dataset D:/.../face_burn_dataset --patient PAT01 --variant A
"""

import json
import argparse
import re
import numpy as np
import cv2
from pathlib import Path
from itertools import groupby

MIN_IOU             = 0.05   # minimum overlap to accept an IoU-based match
MAX_CENTROID_DIST   = 100.0  # mm for aligned 3D data; also reasonable for 2D px fallback

SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  TIME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def elapsed_days(timepoint: str) -> int:
    """D00 -> 0, D14 -> 14, M02 -> 60 (approx, 30 days/month), etc.
    NOTE: month timepoints are approximate (30 days/month). Fine for sorting
    and rough trend plotting; not a substitute for real visit dates if you
    have them."""
    m = re.match(r'([DM])(\d+)', timepoint.upper())
    if not m:
        return -1
    unit, n = m.group(1), int(m.group(2))
    return n if unit == 'D' else n * 30


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY (shared logic with step 2, kept self-contained here)
# ─────────────────────────────────────────────────────────────────────────────

def region_geometry(polygon_pts: list) -> dict:
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
    return {
        "perimeter_pixels": round(perimeter, 2),
        "compactness"     : round(compactness, 4) if compactness is not None else None,
        "centroid_xy"     : [round(cx, 2), round(cy, 2)],
    }


def rasterize(polygon_pts: list, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cnt = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [cnt], 1)
    return mask


def iou(maskA: np.ndarray, maskB: np.ndarray) -> float:
    inter = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    return float(inter / union) if union > 0 else 0.0


def centroid_dist(c1, c2) -> float:
    """Works for both 2D [x,y] and 3D [x,y,z] centroids."""
    a, b = np.array(c1, dtype=float), np.array(c2, dtype=float)
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD A SCAN'S REGIONS (with rasterized mask + geometry attached)
# ─────────────────────────────────────────────────────────────────────────────

def load_scan_regions(polygons_json_path: Path) -> dict:
    # prefer _burn_polygons_aligned.json when available (has area_mm2 + aligned centroid)
    aligned_path = polygons_json_path.parent / polygons_json_path.name.replace(
        "_burn_polygons.json", "_burn_polygons_aligned.json"
    )
    path = aligned_path if aligned_path.exists() else polygons_json_path
    is_aligned = path == aligned_path

    with open(path, "r") as f:
        data = json.load(f)
    shape = (data["image_size"]["height"], data["image_size"]["width"])

    regions = []
    for r in data.get("regions", []):
        geom = region_geometry(r["polygon"])
        # use aligned 3D centroid if available, fall back to 2D pixel centroid
        centroid = r.get("aligned_centroid_xyz") or geom["centroid_xy"]
        regions.append({
            "region_id"           : r["region_id"],
            "area_pixels"         : r["area_pixels"],
            "area_mm2"            : r.get("area_mm2"),
            "area_pct"            : round(100.0 * r["area_pixels"] / (shape[0] * shape[1]), 4),
            "confidence_mean"     : r.get("confidence_mean"),
            "confidence_min"      : r.get("confidence_min"),
            "confidence_max"      : r.get("confidence_max"),
            "aligned_centroid_xyz": r.get("aligned_centroid_xyz"),
            "centroid_xy"         : geom["centroid_xy"],
            "centroid"            : centroid,   # unified: 3D if aligned, else 2D
            "mask"                : rasterize(r["polygon"], shape),
            **geom,
        })
    return {
        "scan"       : data.get("scan"),
        "timepoint"  : data.get("timepoint"),
        "is_aligned" : is_aligned,
        "align_status": data.get("alignment_status", "unknown"),
        "image_size" : shape,
        "regions"    : regions,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MATCH ONE SCAN'S REGIONS AGAINST THE PREVIOUS SCAN'S
# ─────────────────────────────────────────────────────────────────────────────

def match_regions(prev_regions: list, curr_regions: list):
    """
    Returns:
      matches      : list of (prev_idx, curr_idx, score, method)
      unmatched_prev, unmatched_curr : lists of indices

    Uses unified 'centroid' key which is 3D (mm) when aligned data is
    available, or 2D (pixels) otherwise.  MAX_CENTROID_DIST is in mm
    for aligned scans (where 1mm ≈ ~1px at typical scanner resolution)
    so the same threshold works reasonably for both cases.
    """
    n_prev, n_curr = len(prev_regions), len(curr_regions)
    matches = []
    used_prev, used_curr = set(), set()

    # Pass 1 -- IoU (2D polygon overlap -- still useful as a shape filter)
    iou_pairs = []
    for i, pr in enumerate(prev_regions):
        for j, cr in enumerate(curr_regions):
            score = iou(pr["mask"], cr["mask"])
            if score >= MIN_IOU:
                iou_pairs.append((score, i, j))
    iou_pairs.sort(reverse=True)
    for score, i, j in iou_pairs:
        if i in used_prev or j in used_curr:
            continue
        matches.append((i, j, round(score, 4), "iou"))
        used_prev.add(i)
        used_curr.add(j)

    # Pass 2 -- centroid distance fallback (3D mm if aligned, 2D px otherwise)
    dist_pairs = []
    for i, pr in enumerate(prev_regions):
        if i in used_prev:
            continue
        for j, cr in enumerate(curr_regions):
            if j in used_curr:
                continue
            d = centroid_dist(pr["centroid"], cr["centroid"])
            if d <= MAX_CENTROID_DIST:
                dist_pairs.append((d, i, j))
    dist_pairs.sort()
    for d, i, j in dist_pairs:
        if i in used_prev or j in used_curr:
            continue
        matches.append((i, j, round(d, 2), "centroid_dist"))
        used_prev.add(i)
        used_curr.add(j)

    unmatched_prev = [i for i in range(n_prev) if i not in used_prev]
    unmatched_curr = [j for j in range(n_curr) if j not in used_curr]
    return matches, unmatched_prev, unmatched_curr


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK A FULL VARIANT SERIES
# ─────────────────────────────────────────────────────────────────────────────

def strip_masks(region: dict) -> dict:
    return {k: v for k, v in region.items() if k != "mask"}


def track_variant_series(scans: list) -> dict:
    """scans: list of load_scan_regions() outputs, sorted chronologically."""
    tracks = {}      # track_id -> list of entries
    next_track_id = 1
    match_log = []

    # active[i] = track_id currently holding "last seen" region i of the previous scan
    active_track_ids = []
    prev_regions = []

    for scan_idx, scan in enumerate(scans):
        elapsed = elapsed_days(scan["timepoint"])
        curr_regions = scan["regions"]

        if scan_idx == 0:
            # seed one track per region in the first scan
            for r in curr_regions:
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = [{
                    "scan": scan["scan"], "timepoint": scan["timepoint"],
                    "elapsed_days": elapsed, "status": "first_seen",
                    **strip_masks(r),
                }]
            active_track_ids = list(tracks.keys())
            prev_regions = curr_regions
            continue

        matches, unmatched_prev, unmatched_curr = match_regions(prev_regions, curr_regions)

        # log every decision for QC
        for i, j, score, method in matches:
            match_log.append({
                "from_scan": scans[scan_idx - 1]["scan"], "to_scan": scan["scan"],
                "prev_region_id": prev_regions[i]["region_id"],
                "curr_region_id": curr_regions[j]["region_id"],
                "method": method, "score": score, "result": "matched",
            })
        for i in unmatched_prev:
            match_log.append({
                "from_scan": scans[scan_idx - 1]["scan"], "to_scan": scan["scan"],
                "prev_region_id": prev_regions[i]["region_id"],
                "curr_region_id": None, "method": None, "score": None,
                "result": "not_detected_this_scan",
            })
        for j in unmatched_curr:
            match_log.append({
                "from_scan": scans[scan_idx - 1]["scan"], "to_scan": scan["scan"],
                "prev_region_id": None,
                "curr_region_id": curr_regions[j]["region_id"],
                "method": None, "score": None, "result": "new_region",
            })

        new_active_track_ids = [None] * len(curr_regions)

        for i, j, score, method in matches:
            tid = active_track_ids[i]
            new_active_track_ids[j] = tid
            baseline = tracks[tid][0]
            prev     = tracks[tid][-1]
            curr     = curr_regions[j]

            # prefer mm² for pct_change (real units), fall back to pixels
            baseline_area = baseline.get("area_mm2") or baseline.get("area_pixels")
            prev_area     = prev.get("area_mm2")     or prev.get("area_pixels")
            curr_area     = curr.get("area_mm2")     or curr.get("area_pixels")
            area_unit     = "mm2" if curr.get("area_mm2") else "pixels"

            tracks[tid].append({
                "scan": scan["scan"], "timepoint": scan["timepoint"],
                "elapsed_days": elapsed, "status": "tracked",
                "match_method": method, "match_score": score,
                "area_unit": area_unit,
                "pct_change_from_baseline": round(
                    100.0 * (curr_area - baseline_area) / baseline_area, 2
                ) if baseline_area else None,
                "pct_change_from_previous": round(
                    100.0 * (curr_area - prev_area) / prev_area, 2
                ) if prev_area else None,
                **strip_masks(curr),
            })

        # regions present now with no match to any previous track -> new tracks
        for j in unmatched_curr:
            tid = next_track_id
            next_track_id += 1
            curr = curr_regions[j]
            tracks[tid] = [{
                "scan": scan["scan"], "timepoint": scan["timepoint"],
                "elapsed_days": elapsed, "status": "first_seen",
                **strip_masks(curr),
            }]
            new_active_track_ids[j] = tid

        # previous tracks not found this scan: keep the track_id "available"
        # for re-matching in a future scan, but record the gap
        for i in unmatched_prev:
            tid = active_track_ids[i]
            tracks[tid].append({
                "scan": scan["scan"], "timepoint": scan["timepoint"],
                "elapsed_days": elapsed, "status": "not_detected_this_scan",
            })

        active_track_ids = new_active_track_ids
        prev_regions = curr_regions

    return {"tracks": tracks, "match_log": match_log}


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVER + DRIVE
# ─────────────────────────────────────────────────────────────────────────────

def discover_polygon_files(dataset_dir: Path, patient: str):
    pat_dir = dataset_dir / patient
    return sorted(pat_dir.glob("*/*/*_burn_polygons.json"))


def group_by_variant(files):
    parsed = []
    for f in files:
        m = SCAN_RE.match(f.stem.replace("_burn_polygons", ""))
        if not m:
            continue
        parsed.append((m.group("variant").upper(), m.group("timepoint").upper(), f))
    parsed.sort(key=lambda x: x[0])
    groups = {}
    for variant, items in groupby(parsed, key=lambda x: x[0]):
        items = list(items)
        items.sort(key=lambda x: elapsed_days(x[1]))
        groups[variant] = [f for _, _, f in items]
    return groups


def run_for_patient(dataset_dir: Path, patient: str, variant_filter: str = None, out_dir: Path = None):
    files = discover_polygon_files(dataset_dir, patient)
    if not files:
        print(f"  No *_burn_polygons.json found for {patient}")
        return

    groups = group_by_variant(files)
    if variant_filter:
        groups = {k: v for k, v in groups.items() if k == variant_filter.upper()}

    # save tracking JSONs into <patient>/tracking/
    if out_dir is None:
        out_dir = dataset_dir / patient / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant, file_list in groups.items():
        scans = [load_scan_regions(f) for f in file_list]
        result = track_variant_series(scans)
        out_path = out_dir / f"{patient}_{variant}_tracking.json"
        with open(out_path, "w") as out_f:
            json.dump(result, out_f, indent=2)
        n_tracks = len(result["tracks"])
        n_flags  = sum(1 for m in result["match_log"] if m["result"] != "matched")
        print(f"  ✓ {patient}_{variant}: {len(scans)} scans, {n_tracks} track(s) "
              f"→ {out_path}  ({n_flags} log entries need review)")


def main():
    p = argparse.ArgumentParser(description="Step 3 — track burn regions over time per variant")
    p.add_argument("--dataset", required=True, help="Dataset root")
    p.add_argument("--patient", required=True, help="Patient ID, e.g. PAT01")
    p.add_argument("--variant", default=None, help="Limit to one variant letter, e.g. A")
    args = p.parse_args()
    run_for_patient(Path(args.dataset), args.patient, args.variant)


if __name__ == "__main__":
    main()