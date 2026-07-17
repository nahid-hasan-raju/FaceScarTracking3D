#!/usr/bin/env python3
"""
step2b_detect_geometric_landmarks.py
======================================
Detects a small set of anatomical landmarks DIRECTLY from the 3D range grid
-- no texture, no color, no MediaPipe. This is a fallback/complement to
step2_detect_landmarks.py specifically for scans where burn discoloration or
medical equipment confuses the texture-based detector.

WHY THIS WORKS ON BURN SCANS WHERE MEDIAPIPE FAILS:
  MediaPipe finds faces by recognizing typical skin-tone/contrast patterns in
  the 2D texture. Severe burn discoloration or occlusion breaks that. This
  detector instead uses pure 3D SHAPE (how far the surface protrudes at each
  angle around the head) -- burns change skin COLOR, not the underlying bone
  structure, so protrusion-based landmarks are largely unaffected by them.

WHAT IT CAN FIND (and why):
  - nose_tip : the single most protruding point on the entire head at any
               given height -- a sharp, narrow radius peak.
  - ear_a / ear_b : smaller, wider protrusion peaks roughly 50-130 degrees
               either side of the nose. Ears are on the side of the head,
               away from the face proper, so they're usually spared even
               when the face itself is heavily burned.
  - crown : a point near the top of the head, straight up from the nose
               direction. Approximate (not the literal vertex), but a
               stable, far-from-the-face anchor point.

WHAT IT CANNOT FIND:
  Eye corners, mouth corners -- these aren't geometric extrema, they need
  texture/color to locate precisely. Use step2_detect_landmarks.py (MediaPipe)
  for those; this script is a complement, not a full replacement.

THIS IS HEURISTIC AND UNVALIDATED ON YOUR REAL DATA YET.
  Run with --debug first and inspect the radius-profile plots before trusting
  the output for alignment. The exact nose/ear angular offsets can vary with
  how the patient was positioned in the scanner.

STRUCTURE (matches step1/step2):
    PAT01/D00/PAT01_D00_A/
        PAT01_D00_A                          <- range file
        landmarks/
            PAT01_D00_A_geometric.json       <- written here
            PAT01_D00_A_geometric_debug.png  <- only with --debug

USAGE:
    python step2b_detect_geometric_landmarks.py --dataset "D:/..." --scan-id PAT04_M24_A --debug
    python step2b_detect_geometric_landmarks.py --dataset "D:/..." --patient PAT04
    python step2b_detect_geometric_landmarks.py --dataset "D:/..."

REQUIREMENTS:
    pip install numpy matplotlib scipy
"""

import sys, math, re, argparse, json
import numpy as np
from pathlib import Path

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4
SCANNER_RADIUS_MM = 9  * 25.4

SCAN_RE = re.compile(r"^(PAT\d+)_([DM]\d+)_([A-Z][A-Z0-9]?)$")


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVERY  (identical to step1/step2)
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
        if not range_path.exists():
            print(f"  ⚠  range file missing: {range_path}"); continue

        scans.append((patient_id, timepoint_id, scan_dir.name, range_path, scan_dir))
    return sorted(scans, key=lambda x: (x[0], x[1], x[2]))


# ─────────────────────────────────────────────────────────────────────────────
#  CYBERWARE GRID  (same parsing as step1/step2)
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


def grid_to_xyz(row, col, radius, theta_step, z_scale_mm):
    theta = row * theta_step
    z = col * z_scale_mm
    return [float(radius * math.cos(theta)), float(radius * math.sin(theta)), float(z)]


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRIC FEATURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def radius_profile_at(radius_mm, valid_mask, NLG, NLT, col, band=2):
    """Average radius across theta at a given height (col), smoothed over a
    few neighboring columns for noise reduction. NaN where too little valid
    data exists at that angle."""
    c0, c1 = max(0, col - band), min(NLT, col + band + 1)
    sub_r = radius_mm[:, c0:c1]
    sub_v = valid_mask[:, c0:c1]
    counts = sub_v.sum(axis=1)
    sums = np.where(sub_v, sub_r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return profile


def peak_prominence(profile, min_valid_frac=0.5):
    """Highest point minus local baseline (median). Returns (-inf, None) if
    too much of this ring is missing data to trust it."""
    valid = ~np.isnan(profile)
    if valid.mean() < min_valid_frac:
        return -np.inf, None
    baseline = np.nanmedian(profile)
    idx = int(np.nanargmax(profile))
    return float(profile[idx] - baseline), idx


def local_validity_fraction(valid_mask, row, col, NLG, NLT, radius_idx=15):
    """Fraction of valid pixels in a window around (row, col). Low values
    mean the candidate sits right next to a large data void -- the classic
    signature of a scanner dropout (very common at the nose tip on this
    scanner type, due to steep surface angle / specular reflection there)
    rather than a genuinely absent feature."""
    r0, r1 = row - radius_idx, row + radius_idx + 1
    c0, c1 = max(0, col - radius_idx), min(NLT, col + radius_idx + 1)
    rows = np.arange(r0, r1) % NLG
    window = valid_mask[rows][:, c0:c1]
    return float(window.mean())


def find_nose_height(radius_mm, valid_mask, NLG, NLT, col_step=3, band=2,
                      skip_top_frac=0.08, skip_bottom_frac=0.35):
    """Scans candidate heights and picks the one with the single most
    prominent, narrow protrusion peak -- that's the nose. Skips the very top
    (often hair / no-data) and the lower third (often below the chin).

    NOTE: this is a BLIND global search with no prior on where the nose
    should be. If the true nose has a data dropout, this can lock onto an
    unrelated feature elsewhere on the head (e.g. the jaw) since it's simply
    the next-biggest bump anywhere. See find_nose_height_hinted() for a
    version that searches only near a known rough location instead."""
    c_min = int(NLT * skip_top_frac)
    c_max = int(NLT * (1 - skip_bottom_frac))
    best = (-np.inf, None, None)  # prominence, col, theta_idx
    for col in range(c_min, c_max, col_step):
        profile = radius_profile_at(radius_mm, valid_mask, NLG, NLT, col, band)
        prom, idx = peak_prominence(profile)
        if prom > best[0]:
            best = (prom, col, idx)
    return best  # (prominence, col_nose, theta_idx_nose)


def find_nose_height_hinted(radius_mm, valid_mask, NLG, NLT, theta_hint_idx, col_hint,
                             theta_window_deg=35, col_window=25, band=2):
    """Like find_nose_height, but constrained to a small region around a
    known rough location (theta_hint_idx, col_hint) -- e.g. from a 2D/texture
    detector. This is the fix for the 'wanders off to the jaw' failure: with
    a prior, the search physically cannot consider points far from where the
    nose should be, so a dropout at the true nose location causes a
    lower-confidence nearby answer instead of a wrong answer far away."""
    deg_per_idx = 360.0 / NLG
    theta_window_idx = int(theta_window_deg / deg_per_idx)
    c_min = max(0, col_hint - col_window)
    c_max = min(NLT, col_hint + col_window + 1)

    best = (-np.inf, None, None)
    for col in range(c_min, c_max, max(1, band)):
        profile = radius_profile_at(radius_mm, valid_mask, NLG, NLT, col, band)
        # restrict to the angular window around the hint (circular-safe)
        rotated = np.roll(profile, -theta_hint_idx)
        window = rotated[:theta_window_idx + 1]
        window = np.concatenate([window, rotated[-theta_window_idx:]]) if theta_window_idx > 0 else window
        if np.all(np.isnan(window)):
            continue
        local_idx = int(np.nanargmax(window))
        # map local_idx (within concatenated window) back to a global theta index
        if local_idx <= theta_window_idx:
            global_idx = (theta_hint_idx + local_idx) % NLG
        else:
            global_idx = (theta_hint_idx - (len(window) - local_idx)) % NLG
        baseline = np.nanmedian(profile)
        prom = float(window[local_idx] - baseline)
        if prom > best[0]:
            best = (prom, col, global_idx)
    return best


def find_ear_peaks(radius_mm, valid_mask, NLG, NLT, col_nose, theta_idx_nose,
                    band=2, search_deg=(30, 150), min_prominence_mm=3.0):
    """Given the nose height/angle, finds genuine local-maximum peaks (via
    scipy's proper peak-finding, using prominence to reject points that are
    merely on the descending flank of the main nose hump -- important when
    burn swelling merges nose+cheek into one wide protrusion) within an
    angular window on each side of the nose."""
    from scipy.signal import find_peaks

    profile = radius_profile_at(radius_mm, valid_mask, NLG, NLT, col_nose, band)
    deg_per_idx = 360.0 / NLG
    lo_idx = int(search_deg[0] / deg_per_idx)
    hi_idx = int(search_deg[1] / deg_per_idx)

    # Rotate so the nose sits at index 0 -- turns the circular search into two
    # simple contiguous slices (no wraparound headaches).
    rotated = np.roll(profile, -theta_idx_nose)

    def best_peak_in_slice(segment, is_right_side):
        valid_vals = segment[~np.isnan(segment)]
        if len(valid_vals) < 3:
            return None
        fill_value = float(np.min(valid_vals)) - 5.0
        filled = np.where(np.isnan(segment), fill_value, segment)
        peaks, props = find_peaks(filled, prominence=min_prominence_mm)
        if len(peaks) == 0:
            return None
        best = peaks[np.argmax(props["prominences"])]
        offset = lo_idx + best
        global_idx = (theta_idx_nose + (offset if is_right_side else -offset)) % NLG
        return global_idx

    right_segment = rotated[lo_idx:hi_idx + 1]
    left_segment  = rotated[NLG - hi_idx: NLG - lo_idx + 1][::-1]

    ear_a = best_peak_in_slice(right_segment, is_right_side=True)
    ear_b = best_peak_in_slice(left_segment, is_right_side=False)
    return ear_a, ear_b, profile


def find_crown(radius_mm, valid_mask, NLG, NLT, theta_idx_nose,
               skip_top_frac=0.08):
    """Approximate crown: go up the nose-direction sagittal line to near the
    top of valid data (skipping the very top, often noisy/hair-covered)."""
    c_target = int(NLT * (1 - skip_top_frac))
    # search downward from the target row until we hit valid data at the
    # nose angle (hair etc. may make the exact top row invalid)
    for col in range(c_target, 0, -1):
        if valid_mask[theta_idx_nose, col]:
            return col
    return None


def detect_geometric_landmarks(range_path, dropout_warn_thresh=0.7):
    radius_mm, valid_mask, NLG, NLT, theta_step, z_scale_mm = load_range_grid(range_path)

    prominence, col_nose, theta_idx_nose = find_nose_height(radius_mm, valid_mask, NLG, NLT)
    if col_nose is None:
        return None, {"error": "no reliable nose peak found (too much missing data)"}

    ear_a_idx, ear_b_idx, profile_at_nose = find_ear_peaks(
        radius_mm, valid_mask, NLG, NLT, col_nose, theta_idx_nose
    )
    col_crown = find_crown(radius_mm, valid_mask, NLG, NLT, theta_idx_nose)

    landmarks = {}
    confidence = {}

    def add_landmark(name, row, col):
        landmarks[name] = grid_to_xyz(row, col, float(radius_mm[row, col]), theta_step, z_scale_mm)
        confidence[name] = local_validity_fraction(valid_mask, row, col, NLG, NLT)

    add_landmark("nose_tip", theta_idx_nose, col_nose)
    if ear_a_idx is not None:
        add_landmark("ear_a", ear_a_idx, col_nose)
    if ear_b_idx is not None:
        add_landmark("ear_b", ear_b_idx, col_nose)
    if col_crown is not None:
        add_landmark("crown", theta_idx_nose, col_crown)

    suspect = [name for name, frac in confidence.items() if frac < dropout_warn_thresh]

    meta = {
        "nose_prominence_mm": prominence,
        "col_nose": col_nose,
        "theta_idx_nose": theta_idx_nose,
        "NLG": NLG, "NLT": NLT,
        "local_validity_fraction": confidence,  # per-landmark: fraction of nearby data that's valid
        "suspect_landmarks": suspect,            # sits next to a large data void -- verify before trusting
    }
    return landmarks, meta


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG PLOT
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_plot(range_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radius_mm, valid_mask, NLG, NLT, theta_step, z_scale_mm = load_range_grid(range_path)
    prominence, col_nose, theta_idx_nose = find_nose_height(radius_mm, valid_mask, NLG, NLT)
    if col_nose is None:
        return False

    ear_a_idx, ear_b_idx, profile = find_ear_peaks(radius_mm, valid_mask, NLG, NLT, col_nose, theta_idx_nose)
    degrees = np.arange(NLG) * (360.0 / NLG)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(degrees, profile, color="steelblue", lw=1)
    ax.axvline(degrees[theta_idx_nose], color="red", lw=1.5, label="nose_tip")
    if ear_a_idx is not None:
        ax.axvline(degrees[ear_a_idx], color="green", lw=1.5, label="ear_a")
    if ear_b_idx is not None:
        ax.axvline(degrees[ear_b_idx], color="orange", lw=1.5, label="ear_b")
    ax.set_xlabel("theta (degrees around head)")
    ax.set_ylabel("radius (mm)")
    ax.set_title(f"Radius profile at nose height (col={col_nose}, prominence={prominence:.1f}mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  3D RENDER  (actual point cloud + landmark markers, not just an abstract plot)
# ─────────────────────────────────────────────────────────────────────────────

def read_ply_binary(ply_path):
    """Reads the binary_little_endian PLY written by step1_convert_all_to_3d.py
    (x,y,z floats + r,g,b uchars). Returns (pts[N,3] float, colors[N,3] uint8)."""
    with open(ply_path, "rb") as f:
        raw = f.read()
    header_end = raw.find(b"end_header\n") + len(b"end_header\n")
    header_text = raw[:header_end].decode("ascii", errors="replace")
    n_vertices = None
    for line in header_text.split("\n"):
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
    if n_vertices is None:
        raise ValueError(f"Could not find vertex count in PLY header: {ply_path}")

    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data = np.frombuffer(raw[header_end:header_end + n_vertices * dtype.itemsize], dtype=dtype)
    pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    colors = np.column_stack([data["r"], data["g"], data["b"]]).astype(np.uint8)
    return pts, colors


def _angular_diff(a, b):
    """Smallest signed difference between two angles (radians), wrapped to [-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def save_debug_render_3d(ply_path, landmarks, theta_idx_nose, NLG, out_path,
                          front_window_deg=100, point_size=1.5):
    """Renders the actual point cloud from a viewpoint facing the detected
    nose direction, with landmark markers overlaid -- so you can see directly
    on the face whether nose_tip/ear_a/ear_b/crown landed on real anatomy,
    rather than inferring it from an abstract theta-radius plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, colors = read_ply_binary(ply_path)
    nose_theta = theta_idx_nose * (2 * np.pi / NLG)

    point_theta = np.arctan2(pts[:, 1], pts[:, 0])
    diff = _angular_diff(point_theta, nose_theta)
    front_mask = np.abs(diff) < np.radians(front_window_deg)
    pts_f = pts[front_mask]
    colors_f = colors[front_mask]

    view_dir   = np.array([np.cos(nose_theta), np.sin(nose_theta), 0.0])
    right_axis = np.array([-np.sin(nose_theta), np.cos(nose_theta), 0.0])

    depth = pts_f @ view_dir
    right = pts_f @ right_axis
    up    = pts_f[:, 2]

    order = np.argsort(depth)  # painter's algorithm: far points first, near points drawn on top
    right, up, colors_plot = right[order], up[order], colors_f[order] / 255.0

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(right, up, c=colors_plot, s=point_size, linewidths=0)

    marker_colors = {"nose_tip": "red", "ear_a": "lime", "ear_b": "orange", "crown": "cyan"}
    for name, xyz in landmarks.items():
        p = np.array(xyz)
        r = p @ right_axis
        u = p[2]
        ax.scatter([r], [u], c=marker_colors.get(name, "yellow"), s=80,
                   edgecolors="black", linewidths=1.2, zorder=10, label=name)

    ax.set_aspect("equal")
    ax.invert_yaxis() if False else None  # up is already correct (Z increases upward)
    ax.set_xlabel("right (mm)")
    ax.set_ylabel("up / height (mm)")
    ax.set_title("Front-facing render (nose-direction viewpoint) with detected landmarks")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True)
    p.add_argument("--patient",   default=None)
    p.add_argument("--timepoint", default=None)
    p.add_argument("--scan",      default=None, help="variant only, e.g. A")
    p.add_argument("--scan-id",   default=None, help="exact scan name, e.g. PAT01_D00_A")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug",     action="store_true", help="save radius-profile plot")
    args = p.parse_args()

    if args.scan_id:
        m = SCAN_RE.match(args.scan_id.upper())
        if not m:
            print(f"--scan-id '{args.scan_id}' doesn't match PATxx_Dxx_A format")
            sys.exit(1)
        args.patient, args.timepoint, args.scan = m.group(1), m.group(2), m.group(3)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}"); sys.exit(1)

    scans = discover_scans(dataset_dir, args.patient, args.timepoint, args.scan)
    if not scans:
        print("No scans found."); sys.exit(1)

    print(f"\n{'='*60}\n  Geometric landmark detection  ({len(scans)} scan(s))\n{'='*60}\n")

    report = []
    prev_pat, prev_tp = None, None
    for patient_id, timepoint_id, scan_name, range_path, scan_dir in scans:
        if patient_id != prev_pat:
            print(f"\n{'─'*60}\n  {patient_id}"); prev_pat, prev_tp = patient_id, None
        if timepoint_id != prev_tp:
            print(f"    [{timepoint_id}]"); prev_tp = timepoint_id

        out_dir = scan_dir / "landmarks"
        out_dir.mkdir(exist_ok=True)
        out_json = out_dir / f"{scan_name}_geometric.json"
        if out_json.exists() and not args.overwrite:
            print(f"      ↷  {scan_name}  already done — skip")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"      →  {scan_name}", end="  ", flush=True)
        try:
            landmarks, meta = detect_geometric_landmarks(range_path)
            if landmarks is None:
                print(f"✗  {meta['error']}")
                report.append({"scan": scan_name, "status": "failed", "error": meta["error"]})
                continue

            payload = {"scan": scan_name, "landmarks_3d_mm": landmarks, "meta": meta}
            out_json.write_text(json.dumps(payload, indent=2))

            if args.debug:
                save_debug_plot(range_path, out_dir / f"{scan_name}_geometric_debug.png")
                ply_path = range_path.parent / f"{scan_name}.ply"
                if ply_path.exists():
                    save_debug_render_3d(ply_path, landmarks, meta["theta_idx_nose"], meta["NLG"],
                                         out_dir / f"{scan_name}_geometric_3d.png")
                else:
                    print("(no .ply found -- run step1 first for the 3D render) ", end="")

            print(f"✓  {len(landmarks)} points  (nose prominence {meta['nose_prominence_mm']:.1f}mm)", end="")
            if meta["suspect_landmarks"]:
                print(f"  ⚠ possible dropout near: {meta['suspect_landmarks']}")
            else:
                print()
            report.append({"scan": scan_name, "status": "ok", "n_landmarks": len(landmarks),
                           "suspect": meta["suspect_landmarks"]})
        except Exception as e:
            print(f"✗  {e}")
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok     = sum(1 for r in report if r["status"] == "ok")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  Detected : {ok}")
    print(f"  Skipped  : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed   : {len(failed)}")
        for r in failed: print(f"    ✗  {r['scan']} — {r['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
