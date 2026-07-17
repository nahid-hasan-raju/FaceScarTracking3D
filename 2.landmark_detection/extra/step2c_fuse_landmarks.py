#!/usr/bin/env python3
"""
step2c_fuse_landmarks.py
==========================
Combines 2D (MediaPipe, step2) and 3D (geometric) detection for the standard
case: reads MediaPipe's nose_tip position and uses it to anchor a precise 3D
geometric search nearby, instead of trusting either method blind.

  1. Read MediaPipe's nose_tip 3D position from step2's output (same
     coordinate system as the range grid -- no extra registration needed).
  2. Use it as a search hint: look for the true protrusion peak only near
     that location, not the whole head. Fixes the failure mode where a
     scanner depth dropout at the nose causes a blind search to lock onto an
     unrelated bump elsewhere (e.g. the jaw) with no warning.
  3. Compare MediaPipe's point vs. the refined point -- large disagreement
     is flagged, not silently resolved.
  4. Detect ears/crown relative to the (now well-anchored) nose direction.
  5. Sanity-check the result against plausible human proportions.

If nose_tip can't be found at all (no MediaPipe hint AND no 3D signal nearby),
the scan is reported as FAILED rather than guessed at -- these rare outliers
(severe occlusion, total scanner dropout) are meant for manual review, not
further automated guessing.

STRUCTURE (matches step1/step2):
    PAT01/D00/PAT01_D00_A/
        PAT01_D00_A                       <- range file
        PAT01_D00_A.ply
        landmarks/
            PAT01_D00_A_landmarks.json    <- step2 (MediaPipe) output, read here
            PAT01_D00_A_final.json        <- written here

USAGE:
    python step2c_fuse_landmarks.py --dataset "D:/..." --scan-id PAT01_D00_A --debug
    python step2c_fuse_landmarks.py --dataset "D:/..." --patient PAT01
    python step2c_fuse_landmarks.py --dataset "D:/..."

REQUIREMENTS:
    pip install numpy matplotlib scipy
    (needs step2's _landmarks.json to already exist -- run step2 first)
"""

import sys, math, re, argparse, json
import numpy as np
from pathlib import Path

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4
SCANNER_RADIUS_MM = 9  * 25.4
SCAN_RE = re.compile(r"^(PAT\d+)_([DM]\d+)_([A-Z][A-Z0-9]?)$")

PLAUSIBLE_EAR_TO_EAR_MM    = (100, 220)
PLAUSIBLE_NOSE_TO_CROWN_MM = (60, 220)
DISAGREEMENT_FLAG_MM       = 15.0
SUSPECT_VALIDITY_THRESHOLD = 0.7


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVERY
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
#  CYBERWARE GRID
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


def xyz_to_grid_hint(xyz, theta_step, z_scale_mm, NLG):
    x, y, z = xyz
    theta = math.atan2(y, x) % (2 * math.pi)
    theta_idx = int(round(theta / theta_step)) % NLG
    col = int(round(z / z_scale_mm))
    return theta_idx, col


def dist3d(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRIC DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def radius_profile_at(radius_mm, valid_mask, NLG, NLT, col, band=2):
    c0, c1 = max(0, col - band), min(NLT, col + band + 1)
    sub_r = radius_mm[:, c0:c1]
    sub_v = valid_mask[:, c0:c1]
    counts = sub_v.sum(axis=1)
    sums = np.where(sub_v, sub_r, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return profile


def peak_prominence(profile, min_valid_frac=0.5):
    valid = ~np.isnan(profile)
    if valid.mean() < min_valid_frac:
        return -np.inf, None
    baseline = np.nanmedian(profile)
    idx = int(np.nanargmax(profile))
    return float(profile[idx] - baseline), idx


def local_validity_fraction(valid_mask, row, col, NLG, NLT, radius_idx=15):
    r0, r1 = row - radius_idx, row + radius_idx + 1
    c0, c1 = max(0, col - radius_idx), min(NLT, col + radius_idx + 1)
    rows = np.arange(r0, r1) % NLG
    window = valid_mask[rows][:, c0:c1]
    return float(window.mean())


def find_nose_height_blind(radius_mm, valid_mask, NLG, NLT, col_step=3, band=2,
                            skip_top_frac=0.08, skip_bottom_frac=0.35):
    c_min = int(NLT * skip_top_frac)
    c_max = int(NLT * (1 - skip_bottom_frac))
    best = (-np.inf, None, None)
    for col in range(c_min, c_max, col_step):
        profile = radius_profile_at(radius_mm, valid_mask, NLG, NLT, col, band)
        prom, idx = peak_prominence(profile)
        if prom > best[0]:
            best = (prom, col, idx)
    return best


def find_nose_height_hinted(radius_mm, valid_mask, NLG, NLT, theta_hint_idx, col_hint,
                             theta_window_deg=35, col_window=25, band=2):
    """Constrained search -- looks only near a known rough location, so it
    cannot wander off to an unrelated feature elsewhere on the head."""
    deg_per_idx = 360.0 / NLG
    theta_window_idx = max(1, int(theta_window_deg / deg_per_idx))
    c_min = max(0, col_hint - col_window)
    c_max = min(NLT, col_hint + col_window + 1)

    best = (-np.inf, None, None)
    for col in range(c_min, c_max, max(1, band)):
        profile = radius_profile_at(radius_mm, valid_mask, NLG, NLT, col, band)
        rotated = np.roll(profile, -theta_hint_idx)
        window = np.concatenate([rotated[:theta_window_idx + 1], rotated[-theta_window_idx:]])
        if np.all(np.isnan(window)):
            continue
        local_idx = int(np.nanargmax(window))
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
    from scipy.signal import find_peaks
    profile = radius_profile_at(radius_mm, valid_mask, NLG, NLT, col_nose, band)
    deg_per_idx = 360.0 / NLG
    lo_idx = int(search_deg[0] / deg_per_idx)
    hi_idx = int(search_deg[1] / deg_per_idx)
    rotated = np.roll(profile, -theta_idx_nose)

    def best_peak_in_slice(segment, is_right_side):
        valid_vals = segment[~np.isnan(segment)]
        if len(valid_vals) < 3:
            return None
        fill_value = float(np.min(valid_vals)) - 5.0  # modest fill, not an extreme
        filled = np.where(np.isnan(segment), fill_value, segment)  # sentinel that would
        peaks, props = find_peaks(filled, prominence=min_prominence_mm)  # fake a bogus prominence
        if len(peaks) == 0:
            return None
        best = peaks[np.argmax(props["prominences"])]
        offset = lo_idx + best
        return (theta_idx_nose + (offset if is_right_side else -offset)) % NLG

    right_segment = rotated[lo_idx:hi_idx + 1]
    left_segment  = rotated[NLG - hi_idx: NLG - lo_idx + 1][::-1]
    ear_a = best_peak_in_slice(right_segment, is_right_side=True)
    ear_b = best_peak_in_slice(left_segment, is_right_side=False)
    return ear_a, ear_b


def find_crown(radius_mm, valid_mask, NLG, NLT, theta_idx_nose, skip_top_frac=0.08):
    c_target = int(NLT * (1 - skip_top_frac))
    for col in range(c_target, 0, -1):
        if valid_mask[theta_idx_nose, col]:
            return col
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  3D DEBUG RENDER
# ─────────────────────────────────────────────────────────────────────────────

def read_ply_binary(ply_path):
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
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def save_debug_render_3d(ply_path, landmarks, theta_idx_nose, NLG, out_path,
                          front_window_deg=100, point_size=1.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, colors = read_ply_binary(ply_path)
    nose_theta = theta_idx_nose * (2 * np.pi / NLG)
    point_theta = np.arctan2(pts[:, 1], pts[:, 0])
    front_mask = np.abs(_angular_diff(point_theta, nose_theta)) < np.radians(front_window_deg)
    pts_f, colors_f = pts[front_mask], colors[front_mask]

    view_dir   = np.array([np.cos(nose_theta), np.sin(nose_theta), 0.0])
    right_axis = np.array([-np.sin(nose_theta), np.cos(nose_theta), 0.0])
    depth = pts_f @ view_dir
    right = pts_f @ right_axis
    up    = pts_f[:, 2]
    order = np.argsort(depth)
    right, up, colors_plot = right[order], up[order], colors_f[order] / 255.0

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(right, up, c=colors_plot, s=point_size, linewidths=0)
    marker_colors = {"nose_tip": "red", "ear_a": "lime", "ear_b": "orange", "crown": "cyan"}
    for name, xyz in landmarks.items():
        p = np.array(xyz)
        ax.scatter([p @ right_axis], [p[2]], c=marker_colors.get(name, "yellow"), s=80,
                   edgecolors="black", linewidths=1.2, zorder=10, label=name)
    ax.set_aspect("equal")
    ax.set_xlabel("right (mm)"); ax.set_ylabel("up / height (mm)")
    ax.set_title("Fused (2D+3D) landmarks")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  FUSION  (standard case only -- outliers are reported as failed, not guessed)
# ─────────────────────────────────────────────────────────────────────────────

def load_mediapipe_nose(landmarks_json_path):
    if not landmarks_json_path.exists():
        return None, "no_step2_output"
    data = json.loads(landmarks_json_path.read_text())
    stable = data.get("stable_landmark_names", {})
    nose_idx = stable.get("nose_tip")
    if nose_idx is None:
        return None, "no_nose_tip_in_stable_set"
    xyz = data.get("landmarks_3d_mm", {}).get(str(nose_idx))
    if xyz is None:
        return None, "mediapipe_nose_had_no_valid_depth"
    return xyz, "ok"


def fuse_scan(range_path, mediapipe_json_path):
    radius_mm, valid_mask, NLG, NLT, theta_step, z_scale_mm = load_range_grid(range_path)
    mp_nose_xyz, mp_status = load_mediapipe_nose(mediapipe_json_path)

    provenance = {}
    if mp_nose_xyz is not None:
        theta_hint, col_hint = xyz_to_grid_hint(mp_nose_xyz, theta_step, z_scale_mm, NLG)
        prom, col_nose, theta_idx_nose = find_nose_height_hinted(
            radius_mm, valid_mask, NLG, NLT, theta_hint, col_hint
        )
        if col_nose is None:
            prom, col_nose, theta_idx_nose = find_nose_height_blind(radius_mm, valid_mask, NLG, NLT)
            provenance["nose_tip"] = "blind_fallback_after_hint_failed"
        else:
            provenance["nose_tip"] = "hinted_by_2d"
    else:
        prom, col_nose, theta_idx_nose = find_nose_height_blind(radius_mm, valid_mask, NLG, NLT)
        provenance["nose_tip"] = f"blind_no_2d_hint ({mp_status})"

    if col_nose is None:
        return None, {"error": f"no reliable nose location found ({provenance['nose_tip']}) "
                                "-- likely an outlier scan, needs manual review"}

    landmarks, confidence = {}, {}

    def add(name, row, col, prov):
        landmarks[name] = grid_to_xyz(row, col, float(radius_mm[row, col]), theta_step, z_scale_mm)
        confidence[name] = local_validity_fraction(valid_mask, row, col, NLG, NLT)
        provenance[name] = prov

    add("nose_tip", theta_idx_nose, col_nose, provenance["nose_tip"])

    agreement_mm = dist3d(mp_nose_xyz, landmarks["nose_tip"]) if mp_nose_xyz is not None else None

    ear_a_idx, ear_b_idx = find_ear_peaks(radius_mm, valid_mask, NLG, NLT, col_nose, theta_idx_nose)
    if ear_a_idx is not None:
        add("ear_a", ear_a_idx, col_nose, "geometric")
    if ear_b_idx is not None:
        add("ear_b", ear_b_idx, col_nose, "geometric")

    col_crown = find_crown(radius_mm, valid_mask, NLG, NLT, theta_idx_nose)
    if col_crown is not None:
        add("crown", theta_idx_nose, col_crown, "geometric")

    plausibility_flags = []
    if "ear_a" in landmarks and "ear_b" in landmarks:
        d = dist3d(landmarks["ear_a"], landmarks["ear_b"])
        if not (PLAUSIBLE_EAR_TO_EAR_MM[0] <= d <= PLAUSIBLE_EAR_TO_EAR_MM[1]):
            plausibility_flags.append(f"ear_to_ear_distance_implausible ({d:.0f}mm)")
    if "crown" in landmarks:
        d = dist3d(landmarks["nose_tip"], landmarks["crown"])
        if not (PLAUSIBLE_NOSE_TO_CROWN_MM[0] <= d <= PLAUSIBLE_NOSE_TO_CROWN_MM[1]):
            plausibility_flags.append(f"nose_to_crown_distance_implausible ({d:.0f}mm)")

    suspect = [name for name, frac in confidence.items() if frac < SUSPECT_VALIDITY_THRESHOLD]
    if agreement_mm is not None and agreement_mm > DISAGREEMENT_FLAG_MM:
        suspect.append("nose_tip_2d_3d_disagreement")

    meta = {
        "nose_prominence_mm": prom, "col_nose": col_nose, "theta_idx_nose": theta_idx_nose,
        "NLG": NLG, "NLT": NLT,
        "mediapipe_nose_xyz": mp_nose_xyz, "mediapipe_status": mp_status,
        "nose_2d_3d_agreement_mm": agreement_mm,
        "local_validity_fraction": confidence,
        "provenance": provenance,
        "plausibility_flags": plausibility_flags,
        "suspect_landmarks": suspect,
    }
    return landmarks, meta


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True)
    p.add_argument("--patient",   default=None)
    p.add_argument("--timepoint", default=None)
    p.add_argument("--scan",      default=None)
    p.add_argument("--scan-id",   default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug",     action="store_true", help="save 3D render")
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

    print(f"\n{'='*60}\n  Fused (2D+3D) landmark detection  ({len(scans)} scan(s))\n{'='*60}\n")

    report = []
    prev_pat, prev_tp = None, None
    for patient_id, timepoint_id, scan_name, range_path, scan_dir in scans:
        if patient_id != prev_pat:
            print(f"\n{'─'*60}\n  {patient_id}"); prev_pat, prev_tp = patient_id, None
        if timepoint_id != prev_tp:
            print(f"    [{timepoint_id}]"); prev_tp = timepoint_id

        out_dir = scan_dir / "landmarks"
        out_dir.mkdir(exist_ok=True)
        out_json = out_dir / f"{scan_name}_final.json"
        mediapipe_json_path = out_dir / f"{scan_name}_landmarks.json"

        if out_json.exists() and not args.overwrite:
            print(f"      ↷  {scan_name}  already done — skip")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"      →  {scan_name}", end="  ", flush=True)
        try:
            landmarks, meta = fuse_scan(range_path, mediapipe_json_path)
            if landmarks is None:
                print(f"✗  {meta['error']}")
                report.append({"scan": scan_name, "status": "failed", "error": meta["error"]})
                continue

            payload = {"scan": scan_name, "landmarks_3d_mm": landmarks, "meta": meta}
            out_json.write_text(json.dumps(payload, indent=2))

            if args.debug:
                ply_path = range_path.parent / f"{scan_name}.ply"
                if ply_path.exists():
                    save_debug_render_3d(ply_path, landmarks, meta["theta_idx_nose"], meta["NLG"],
                                         out_dir / f"{scan_name}_final_3d.png")

            tag = ""
            if meta["nose_2d_3d_agreement_mm"] is not None:
                tag += f"  2D/3D: {meta['nose_2d_3d_agreement_mm']:.1f}mm"
            if meta["suspect_landmarks"]:
                tag += f"  ⚠ {meta['suspect_landmarks']}"
            if meta["plausibility_flags"]:
                tag += f"  ⚠ {meta['plausibility_flags']}"
            print(f"✓  {len(landmarks)} points{tag}")
            report.append({"scan": scan_name, "status": "ok", "suspect": meta["suspect_landmarks"]})
        except Exception as e:
            print(f"✗  {e}")
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok     = sum(1 for r in report if r["status"] == "ok")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  Fused    : {ok}")
    print(f"  Skipped  : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed   : {len(failed)}  (outliers -- needs manual review, not auto-guessed)")
        for r in failed: print(f"    ✗  {r['scan']} — {r['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()