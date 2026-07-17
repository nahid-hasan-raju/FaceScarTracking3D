#!/usr/bin/env python3
"""
step2_detect_landmarks.py
===========================
Detects ALL MediaPipe Face Mesh landmarks (~478 points, full mesh) on each
scan's TIF texture, then converts every one of them into REAL 3D coordinates
using the exact same Cyberware range-file grid math as
step1_convert_all_to_3d.py. Everything is stored -- which subset to actually
use for the alignment fit (step 3) is decided later, not hardcoded here.

A small named subset (STABLE_LANDMARK_NAMES below) is called out for
convenience and used for the debug overlay, since drawing all ~478 points
would be unreadable clutter. These particular points sit on bone/cartilage
and don't change shape with burn swelling or healing -- good default
candidates for the alignment fit -- but the full mesh is available in the
JSON if you decide you need a different or larger set later.

WHY 2D FIRST, THEN PROJECT TO 3D:
  MediaPipe only works on 2D images. Running it on the TIF (before any 3D
  alignment) avoids depending on a snapshot of the aligned mesh (which was
  unreliable with Open3D). We then look up the REAL scanner-measured depth
  for each landmark from the range file grid -- not MediaPipe's own (much
  less accurate) estimated depth.

STRUCTURE (matches step1_convert_all_to_3d.py):
    PAT01/D00/PAT01_D00_A/
        PAT01_D00_A                     <- range file
        PAT01_D00_A.tif
        PAT01_D00_A.ply
        landmarks/                       <- written here
            PAT01_D00_A_landmarks.json
            PAT01_D00_A_landmarks_debug.png   (only with --debug)

OUTPUT JSON per scan:
    {
      "scan": "PAT01_D00_A",
      "image_size": [cw, ch],
      "stable_landmark_names": {"nose_tip": 1, "chin": 152, ...},
      "landmarks_2d_px":       {"0": [x, y], "1": [x, y], ... "477": [x, y]},
      "landmarks_3d_mm":       {"0": [x, y, z], ...},
      "landmark_depth_samples":{"0": n, ...}   -- depth readings averaged per point
    }

USAGE:
    python step2_detect_landmarks.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
    python step2_detect_landmarks.py --dataset "D:/..." --patient PAT01
    python step2_detect_landmarks.py --dataset "D:/..." --scan-id PAT01_D00_A
    python step2_detect_landmarks.py --dataset "D:/..." --overwrite
    python step2_detect_landmarks.py --dataset "D:/..." --debug   # saves landmark overlay PNG

REQUIREMENTS:
    pip install mediapipe opencv-python numpy pillow
    (the FaceLandmarker model file is auto-downloaded on first run, ~4MB,
    saved next to this script as face_landmarker.task)
"""

import sys, math, re, argparse, json
import numpy as np
from pathlib import Path

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4
SCANNER_RADIUS_MM = 9  * 25.4

SCAN_RE = re.compile(r"^(PAT\d+)_([DM]\d+)_([A-Z][A-Z0-9]?)$")

# ─────────────────────────────────────────────────────────────────────────────
#  Stable (non-burn-prone) landmark names -> MediaPipe index.
#  These are just LABELS for a few indices worth naming for convenience --
#  we still detect and store ALL ~478 mesh points below. Which subset to
#  actually use for the rigid alignment fit is decided later (step 3), not
#  hardcoded here. Sit on bone/cartilage -> don't deform with swelling or
#  scarring elsewhere on cheeks/forehead/jaw.
# ─────────────────────────────────────────────────────────────────────────────
STABLE_LANDMARK_NAMES = {
    "nose_tip":        1,
    "nose_bridge":     6,
    "left_eye_inner":  362,
    "left_eye_outer":  263,
    "right_eye_inner": 133,
    "right_eye_outer": 33,
    "mouth_left":      291,
    "mouth_right":     61,
    "chin":            152,
    "forehead":        10,
}


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVERY  (identical to step1_convert_all_to_3d.py)
# ─────────────────────────────────────────────────────────────────────────────

def discover_scans(dataset_dir, patient_filter=None, timepoint_filter=None, scan_filter=None):
    scans = []
    for scan_dir in sorted(dataset_dir.glob("PAT*/[DM]*/PAT*_[DM]*_*")):
        if not scan_dir.is_dir():
            continue
        m = SCAN_RE.match(scan_dir.name)
        if not m:
            continue
        patient_id, timepoint_id, variant = m.group(1), m.group(2), m.group(3)
        if patient_filter   and patient_id   != patient_filter.upper():   continue
        if timepoint_filter and timepoint_id != timepoint_filter.upper():  continue
        if scan_filter      and variant      != scan_filter.upper():       continue

        range_path = scan_dir / scan_dir.name
        tif_path   = scan_dir / f"{scan_dir.name}.tif"
        if not range_path.exists():
            print(f"  ⚠  range file missing: {range_path}"); continue
        if not tif_path.exists():
            print(f"  ⚠  tif missing: {tif_path}"); continue

        scans.append((patient_id, timepoint_id, scan_dir.name, range_path, tif_path, scan_dir))
    return sorted(scans, key=lambda x: (x[0], x[1], x[2]))


# ─────────────────────────────────────────────────────────────────────────────
#  CYBERWARE GRID  (same parsing as step1_convert_all_to_3d.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_header(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"Cyberware"):
        raise ValueError(f"Not a Cyberware file: {filepath}")
    idx = raw.find(b"DATA=\n")
    if idx == -1:
        raise ValueError("DATA= marker not found.")
    header_end = idx + len(b"DATA=\n")
    params = {}
    for line in raw[:header_end].decode("ascii", errors="replace").split("\n"):
        if "=" in line and not line.startswith("DATA"):
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params, header_end, raw


def load_range_grid(range_path):
    """Returns radius_mm[NLG,NLT], valid_mask, NLG, NLT, theta_step, z_scale_mm."""
    params, header_end, raw = parse_header(range_path)
    NLG    = int(params["NLG"])
    NLT    = int(params["NLT"])
    RSHIFT = int(params["RSHIFT"])
    LGINCR = int(params["LGINCR"])

    r_scale_mm = LGINCR / 32768.0
    z_scale_mm = SCANNER_HEIGHT_MM / NLT
    theta_step = (2.0 * math.pi) / NLG

    data = (np.frombuffer(raw[header_end:header_end + NLG * NLT * 2], dtype=">u2")
              .reshape(NLG, NLT).astype(np.float32))

    valid_mask = (data != INVALID_SENTINEL) & (data > 0)
    radius_mm  = np.where(valid_mask, (data / (2 ** RSHIFT)) * r_scale_mm, np.nan)
    valid_mask = (~np.isnan(radius_mm) & (radius_mm > 0) & (radius_mm <= SCANNER_RADIUS_MM))
    return radius_mm, valid_mask, NLG, NLT, theta_step, z_scale_mm


# ─────────────────────────────────────────────────────────────────────────────
#  PIXEL  <->  GRID   (exact inverse of the tif_row/tif_col mapping in step1)
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_grid_index(px, py, cw, ch, NLG, NLT):
    """
    step1 forward mapping (for reference):
        tif_row = ch-1 - (col * ch/NLT)      # col = z index   (0..NLT-1)
        tif_col = row * cw/NLG               # row = theta index (0..NLG-1)
    This is the inverse: pixel (px, py) on the TIF -> (row, col) grid index.
    """
    col = (ch - 1 - py) * NLT / ch
    row = px * NLG / cw
    row = int(round(row)) % NLG          # theta axis wraps around (360°)
    col = int(round(col))
    col = max(0, min(NLT - 1, col))       # z axis does not wrap
    return row, col


def nearest_valid(radius_mm, valid_mask, row, col, NLG, NLT, max_radius=6):
    """If the exact grid cell has no valid range reading, search outward in a
    small window (handles the rare case a landmark lands on a scanner dropout
    pixel). Theta axis wraps, Z axis is clipped."""
    if valid_mask[row, col]:
        return row, col, float(radius_mm[row, col])
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr = (row + dr) % NLG
                cc = col + dc
                if 0 <= cc < NLT and valid_mask[rr, cc]:
                    return rr, cc, float(radius_mm[rr, cc])
    return None


def grid_to_xyz(row, col, radius, theta_step, z_scale_mm):
    theta = row * theta_step
    z = col * z_scale_mm
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    return [float(x), float(y), float(z)]


# Pixel-neighborhood radius (TIF pixels) to average over per landmark.
# This is NOT about MediaPipe accuracy (its 2D point is already sub-pixel
# precise) -- it's about the range SCANNER having local depth noise/dropouts.
# Averaging a small neighborhood of independently-measured depths around the
# landmark pixel smooths that out. Set to 0 to disable and use a single point.
PIXEL_AVG_RADIUS = 2


def landmark_to_xyz_averaged(px, py, cw, ch, radius_mm, valid_mask, NLG, NLT,
                              theta_step, z_scale_mm, radius=PIXEL_AVG_RADIUS):
    """Average the 3D position over a small pixel window around (px, py) to
    smooth out scanner depth noise. Falls back to a single-point lookup if
    nothing in the window is valid."""
    xyz_samples = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            row, col = pixel_to_grid_index(px + dx, py + dy, cw, ch, NLG, NLT)
            if 0 <= row < NLG and 0 <= col < NLT and valid_mask[row, col]:
                xyz_samples.append(
                    grid_to_xyz(row, col, float(radius_mm[row, col]), theta_step, z_scale_mm)
                )
    if xyz_samples:
        arr = np.array(xyz_samples)
        return arr.mean(axis=0).tolist(), len(xyz_samples)

    # Nothing valid in the window -- fall back to nearest-valid search
    row, col = pixel_to_grid_index(px, py, cw, ch, NLG, NLT)
    found = nearest_valid(radius_mm, valid_mask, row, col, NLG, NLT)
    if found is None:
        return None, 0
    rr, cc, r = found
    return grid_to_xyz(rr, cc, r, theta_step, z_scale_mm), 1


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE  2D DETECTION  (new Tasks API -- the old mp.solutions.face_mesh
#  API was deprecated by Google in 2023 and is broken/removed on recent
#  MediaPipe PyPI releases. This uses the currently-supported FaceLandmarker
#  task instead. Same 468-point topology, so LANDMARK_INDICES is unchanged.)
# ─────────────────────────────────────────────────────────────────────────────

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"\n  Downloading MediaPipe face_landmarker model (one-time, ~4MB)...")
        import urllib.request
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, MODEL_PATH)
        print(f"  Saved to {MODEL_PATH}\n")
    return MODEL_PATH


def create_face_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    model_path = ensure_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_landmarks_2d(tif_path, detector):
    import cv2
    import mediapipe as mp
    img = cv2.imread(str(tif_path))          # BGR
    if img is None:
        raise RuntimeError(f"Could not read TIF: {tif_path}")
    ch, cw = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None, cw, ch

    lm = result.face_landmarks[0]   # first detected face
    # ALL mesh points (typically 478: 468 face + 10 iris), keyed by index.
    # Which subset to actually use is decided later, not here.
    pts2d_all = {i: (l.x * (cw - 1), l.y * (ch - 1)) for i, l in enumerate(lm)}
    return pts2d_all, cw, ch


def save_debug_overlay(tif_path, pts2d_all, out_path):
    """Draws every detected mesh point as a small dot (no labels -- with ~478
    points, text would be unreadable clutter). The named stable subset is
    drawn slightly larger/brighter so it's still easy to spot among the mesh."""
    import cv2
    img = cv2.imread(str(tif_path))
    stable_idx = set(STABLE_LANDMARK_NAMES.values())

    for idx, (px, py) in pts2d_all.items():
        if idx in stable_idx:
            continue  # drawn separately below, on top
        cv2.circle(img, (int(px), int(py)), 1, (0, 255, 0), -1)

    for idx in stable_idx:
        if idx not in pts2d_all:
            continue
        px, py = pts2d_all[idx]
        cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

    cv2.imwrite(str(out_path), img)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(scan_name, range_path, tif_path, out_json, detector, debug=False):
    pts2d_all, cw, ch = detect_landmarks_2d(tif_path, detector)
    if pts2d_all is None:
        return {"scan": scan_name, "status": "failed", "error": "no face detected"}

    radius_mm, valid_mask, NLG, NLT, theta_step, z_scale_mm = load_range_grid(range_path)

    landmarks_3d = {}
    landmark_confidence = {}
    n_missing = 0
    for idx, (px, py) in pts2d_all.items():
        xyz, n_samples = landmark_to_xyz_averaged(
            px, py, cw, ch, radius_mm, valid_mask, NLG, NLT, theta_step, z_scale_mm
        )
        if xyz is None:
            n_missing += 1
            continue
        landmarks_3d[str(idx)] = xyz
        landmark_confidence[str(idx)] = n_samples  # depth samples averaged (lower = noisier)

    # Flag if any of the named stable points specifically failed -- those
    # matter more than a random mesh point failing.
    missing_stable = [name for name, idx in STABLE_LANDMARK_NAMES.items()
                       if str(idx) not in landmarks_3d]
    if missing_stable:
        print(f"(missing stable: {missing_stable}) ", end="")

    payload = {
        "scan": scan_name,
        "image_size": [cw, ch],
        "stable_landmark_names": STABLE_LANDMARK_NAMES,  # name -> index, for convenience
        "landmarks_2d_px": {str(i): [round(v[0], 1), round(v[1], 1)] for i, v in pts2d_all.items()},
        "landmarks_3d_mm": landmarks_3d,                 # index (str) -> [x, y, z]
        "landmark_depth_samples": landmark_confidence,   # index (str) -> n depth readings averaged
    }
    out_json.write_text(json.dumps(payload, indent=2))

    if debug:
        save_debug_overlay(tif_path, pts2d_all, out_json.with_name(out_json.stem + "_debug.png"))

    return {"scan": scan_name, "status": "ok",
            "n_landmarks": len(landmarks_3d), "n_total": len(pts2d_all),
            "n_missing": n_missing, "missing_stable": missing_stable}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True)
    p.add_argument("--patient",   default=None, help="e.g. PAT01 -> all scans for this patient")
    p.add_argument("--timepoint", default=None, help="e.g. D00 -> combine with --patient")
    p.add_argument("--scan",      default=None, help="variant only, e.g. A -> combine with --patient/--timepoint")
    p.add_argument("--scan-id",   default=None,
                   help="exact single scan folder name, e.g. PAT01_D00_A -> run just this one scan")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug",     action="store_true", help="save landmark overlay PNG")
    args = p.parse_args()

    # --scan-id is a convenience that overrides patient/timepoint/scan filters
    if args.scan_id:
        m = SCAN_RE.match(args.scan_id.upper())
        if not m:
            print(f"--scan-id '{args.scan_id}' doesn't match PATxx_Dxx_A format")
            sys.exit(1)
        args.patient, args.timepoint, args.scan = m.group(1), m.group(2), m.group(3)

    try:
        import mediapipe  # noqa: F401 -- just checking it's installed
    except ImportError:
        print("MediaPipe not installed. Run: pip install mediapipe")
        sys.exit(1)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}"); sys.exit(1)

    scans = discover_scans(dataset_dir, args.patient, args.timepoint, args.scan)
    if not scans:
        print("No scans found."); sys.exit(1)

    print(f"\n{'='*60}\n  Landmark detection  ({len(scans)} scan(s))\n{'='*60}\n")

    report = []
    prev_pat, prev_tp = None, None
    detector = create_face_landmarker()
    try:
        for patient_id, timepoint_id, scan_name, range_path, tif_path, scan_dir in scans:
            if patient_id != prev_pat:
                print(f"\n{'─'*60}\n  {patient_id}"); prev_pat, prev_tp = patient_id, None
            if timepoint_id != prev_tp:
                print(f"    [{timepoint_id}]"); prev_tp = timepoint_id

            out_dir = scan_dir / "landmarks"
            out_dir.mkdir(exist_ok=True)
            out_json = out_dir / f"{scan_name}_landmarks.json"
            if out_json.exists() and not args.overwrite:
                print(f"      ↷  {scan_name}  already done — skip")
                report.append({"scan": scan_name, "status": "skipped"})
                continue

            print(f"      →  {scan_name}", end="  ", flush=True)
            try:
                status = process_scan(scan_name, range_path, tif_path, out_json,
                                       detector, debug=args.debug)
                if status["status"] == "ok":
                    warn = f"  ⚠ missing stable: {status['missing_stable']}" if status["missing_stable"] else ""
                    print(f"✓  {status['n_landmarks']}/{status['n_total']} mesh points{warn}")
                else:
                    print(f"✗  {status.get('error')}")
                report.append(status)
            except Exception as e:
                print(f"✗  {e}")
                report.append({"scan": scan_name, "status": "failed", "error": str(e)})
    finally:
        detector.close()

    ok     = sum(1 for r in report if r["status"] == "ok")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  Landmarked : {ok}")
    print(f"  Skipped    : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed     : {len(failed)}")
        for r in failed: print(f"    ✗  {r['scan']} — {r['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()