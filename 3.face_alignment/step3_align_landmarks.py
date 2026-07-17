#!/usr/bin/env python3
"""
step3_align_landmarks.py
==========================
Aligns every scan of a patient into the same coordinate frame, using the
stable landmarks step2 already detected -- no manual axis guessing, no
per-scan fine-tuning.

HOW:
  1. Pick a reference scan per patient (default: earliest timepoint, variant A
     if available -- override with --reference).
  2. For every other scan of that patient, find the landmarks it shares with
     the reference (nose, eye corners, forehead -- NOT mouth/chin, since
     those shift with expression and would bias the fit).
  3. Compute the rigid transform (rotation + translation, no scaling -- same
     physical head, same scanner units) that best maps this scan's landmarks
     onto the reference's, via the Kabsch/SVD method.
  4. Apply that transform to the scan's full point cloud -> {scan}_aligned.ply
  5. Report the fit residual (how well the landmarks actually lined up after
     alignment) so a bad fit is visible, not silently trusted.

WHY THESE LANDMARKS:
  A rigid fit only needs 3 non-collinear points to be fully determined. Using
  more only helps if they're equally trustworthy. Nose/eyes/forehead sit on
  bone and don't move with expression or burn swelling elsewhere; mouth
  corners and chin do (mouth open vs closed, jaw drop) and are excluded by
  default.

STRUCTURE (matches step1/step2):
    PAT01/D00/PAT01_D00_A/
        PAT01_D00_A.ply
        landmarks/
            PAT01_D00_A_landmarks.json     <- step2 output, read here
        PAT01_D00_A_aligned.ply            <- written here
        PAT01_D00_A_aligned_preview.png    <- only with --debug

USAGE:
    python step3_align_landmarks.py --dataset "D:/..." --patient PAT01
    python step3_align_landmarks.py --dataset "D:/..." --patient PAT01 --reference PAT01_D00_A
    python step3_align_landmarks.py --dataset "D:/..." --scan-id PAT01_D14_A --debug
    python step3_align_landmarks.py --dataset "D:/..."

REQUIREMENTS:
    pip install numpy matplotlib
    (needs step2's _landmarks.json to already exist for each scan)
"""

import sys, re, argparse, json
import numpy as np
from pathlib import Path

SCAN_RE = re.compile(r"^(PAT\d+)_([DM]\d+)_([A-Z][A-Z0-9]?)$")

# Landmarks used for the rigid fit -- bone-anchored, expression-independent.
# Mouth corners and chin are deliberately excluded (see module docstring).
FIT_LANDMARK_NAMES = [
    "nose_tip", "nose_bridge",
    "left_eye_inner", "left_eye_outer",
    "right_eye_inner", "right_eye_outer",
    "forehead",
]
# Variant D captures more neck than face (per your protocol), so eyes/forehead
# are often not visible. Isolated fallback used ONLY for variant D -- A/B/C
# behavior is completely unchanged. Falls back to mouth/chin as extra
# candidates only when the core set alone isn't enough points to fit.
FIT_LANDMARK_NAMES_D_FALLBACK = FIT_LANDMARK_NAMES + ["mouth_left", "mouth_right", "chin"]

MIN_POINTS_FOR_FIT = 3
HIGH_RESIDUAL_WARN_MM = 5.0


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

        ply_path = scan_dir / f"{scan_dir.name}.ply"
        landmarks_path = scan_dir / "landmarks" / f"{scan_dir.name}_landmarks.json"
        if not ply_path.exists():
            print(f"  ⚠  ply missing: {ply_path}"); continue
        if not landmarks_path.exists():
            print(f"  ⚠  landmarks missing (run step2 first): {landmarks_path}"); continue

        scans.append({
            "patient": patient_id, "timepoint": timepoint_id, "variant": variant,
            "scan_name": scan_dir.name, "scan_dir": scan_dir,
            "ply_path": ply_path, "landmarks_path": landmarks_path,
        })
    return sorted(scans, key=lambda s: (s["patient"], s["timepoint"], s["variant"]))


def timepoint_sort_key(tp):
    """D00 < D14 < M02 < M24 etc -- D(ays) before M(onths), then numeric."""
    prefix, num = tp[0], int(tp[1:])
    return (0 if prefix == "D" else 1, num)


def pick_reference(scans_for_patient, reference_override=None):
    if reference_override:
        for s in scans_for_patient:
            if s["scan_name"] == reference_override:
                return s
        raise ValueError(f"--reference {reference_override} not found for this patient")
    # earliest timepoint, prefer variant A
    ordered = sorted(scans_for_patient, key=lambda s: (timepoint_sort_key(s["timepoint"]), s["variant"] != "A", s["variant"]))
    return ordered[0]


# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_fit_landmarks(landmarks_path, candidate_names=FIT_LANDMARK_NAMES):
    """Returns {name: xyz} for whichever of candidate_names actually have
    valid 3D positions in this scan's step2 output. Default candidate list is
    unchanged (FIT_LANDMARK_NAMES); pass FIT_LANDMARK_NAMES_D_FALLBACK for
    variant D scans that may lack eyes/forehead."""
    data = json.loads(landmarks_path.read_text())
    stable = data.get("stable_landmark_names", {})
    all_3d = data.get("landmarks_3d_mm", {})
    out = {}
    for name in candidate_names:
        idx = stable.get(name)
        if idx is None:
            continue
        xyz = all_3d.get(str(idx))
        if xyz is not None:
            out[name] = np.array(xyz, dtype=np.float64)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  RIGID FIT  (Kabsch / SVD method)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rigid_transform(P, Q):
    """Finds rotation R and translation t minimizing sum ||R@P_i + t - Q_i||^2.
    P, Q are Nx3 corresponding point sets (this scan -> reference)."""
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    P_c, Q_c = P - cP, Q - cQ
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])  # handles reflection case
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


def apply_transform(pts, R, t):
    return pts @ R.T + t


def fit_residual_mm(P, Q, R, t):
    P_aligned = apply_transform(P, R, t)
    dists = np.linalg.norm(P_aligned - Q, axis=1)
    return float(np.sqrt(np.mean(dists ** 2))), float(np.max(dists))


def save_transform_json(out_path, R, t, reference_scan, rms_mm, max_mm, common_landmarks):
    """Saves the exact rigid transform applied to this scan, so any later
    step (e.g. burn segmentation) can reapply the SAME alignment to a fresh
    point cloud built from the raw range file, without recomputing it or
    falling back to a different alignment method."""
    payload = {
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "reference_scan": reference_scan,
        "rms_residual_mm": rms_mm,
        "max_residual_mm": max_mm,
        "landmarks_used": common_landmarks,
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
#  PLY I/O  (pure numpy, matches step1's writer format)
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


def write_ply_binary(pts, colors, out_path):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    packed = np.zeros(len(pts), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")
    ])
    packed["x"], packed["y"], packed["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    packed["r"], packed["g"], packed["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG PREVIEW
# ─────────────────────────────────────────────────────────────────────────────

def save_preview(pts, colors, out_path, img_w=800, img_h=1000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(pts[:, 2])
    pts, colors = pts[order], colors[order]
    x, y = pts[:, 0], pts[:, 1]
    pad = 0.02
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    ix = (((x - x_min) / (x_max - x_min) * (1 - 2 * pad) + pad) * (img_w - 1)).astype(int).clip(0, img_w - 1)
    iy = (((y - y_min) / (y_max - y_min) * (1 - 2 * pad) + pad) * (img_h - 1)).astype(int).clip(0, img_h - 1)

    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100))
    ax.scatter(ix, img_h - iy, c=colors / 255.0, s=1, linewidths=0)
    ax.set_xlim(0, img_w); ax.set_ylim(0, img_h)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)


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
    p.add_argument("--reference", default=None, help="exact scan name to use as reference for its pose/variant (default per variant: earliest timepoint)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug",     action="store_true", help="save aligned preview PNG")
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

    all_scans = discover_scans(dataset_dir, args.patient, args.timepoint, args.scan)
    if not all_scans:
        print("No scans found (with both .ply and landmarks present)."); sys.exit(1)
    # Process in (patient, variant, timepoint) order so each pose's scans
    # group together -- each variant (A/B/C/D) gets its OWN reference, since
    # they're different poses that shouldn't be cross-registered.
    all_scans.sort(key=lambda s: (s["patient"], s["variant"], timepoint_sort_key(s["timepoint"])))

    # Reference is chosen per (patient, variant), from ALL that patient's
    # scans of that variant (not just the filtered subset), so --scan-id
    # still aligns against the right reference even for a single scan.
    all_patient_scans = discover_scans(dataset_dir, args.patient)
    by_patient_variant = {}
    for s in all_patient_scans:
        by_patient_variant.setdefault((s["patient"], s["variant"]), []).append(s)

    print(f"\n{'='*60}\n  Landmark-based alignment  ({len(all_scans)} scan(s) to process)\n{'='*60}\n")

    report = []
    reference_cache = {}  # (patient, variant) -> (reference_dict, ref_landmarks, candidate_names)
    prev_pat, prev_variant = None, None
    for s in all_scans:
        group_key = (s["patient"], s["variant"])
        if s["patient"] != prev_pat:
            print(f"\n{'─'*60}\n  {s['patient']}"); prev_pat, prev_variant = s["patient"], None
        if s["variant"] != prev_variant:
            prev_variant = s["variant"]
            print(f"    [pose {s['variant']}]")
            if group_key not in reference_cache:
                # Isolated D-pose handling: use the extended candidate list
                # (adds mouth/chin) only for variant D, since it often lacks
                # eyes/forehead. A/B/C are completely unaffected.
                candidate_names = FIT_LANDMARK_NAMES_D_FALLBACK if s["variant"] == "D" else FIT_LANDMARK_NAMES
                try:
                    ref_override = args.reference if (args.reference and
                        any(sc["scan_name"] == args.reference for sc in by_patient_variant[group_key])) else None
                    reference = pick_reference(by_patient_variant[group_key], ref_override)
                    ref_landmarks = load_fit_landmarks(reference["landmarks_path"], candidate_names)
                    reference_cache[group_key] = (reference, ref_landmarks, candidate_names)
                    print(f"      reference: {reference['scan_name']}  ({len(ref_landmarks)} usable landmarks)")
                except Exception as e:
                    print(f"      ✗ could not establish reference: {e}")
                    reference_cache[group_key] = (None, {}, candidate_names)

        reference, ref_landmarks, candidate_names = reference_cache[group_key]

        scan_name = s["scan_name"]
        out_ply = s["scan_dir"] / f"{scan_name}_aligned.ply"
        if out_ply.exists() and not args.overwrite:
            print(f"      ↷  {scan_name}  already done — skip")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        if reference is None:
            report.append({"scan": scan_name, "status": "failed", "error": "no reference for this pose group"})
            continue

        print(f"      →  {scan_name}", end="  ", flush=True)
        try:
            if scan_name == reference["scan_name"]:
                # Reference aligns to itself -- identity transform, still
                # written out so every scan (including reference) has a
                # consistent *_aligned.ply for downstream steps.
                pts, colors = read_ply_binary(s["ply_path"])
                write_ply_binary(pts, colors, out_ply)
                save_transform_json(
                    s["scan_dir"] / f"{scan_name}_alignment.json",
                    np.eye(3), np.zeros(3), reference["scan_name"], 0.0, 0.0,
                    sorted(ref_landmarks.keys())
                )
                if args.debug:
                    save_preview(pts, colors, s["scan_dir"] / f"{scan_name}_aligned_preview.png")
                print("✓  (reference scan, identity transform)")
                report.append({"scan": scan_name, "status": "ok", "rms_mm": 0.0})
                continue

            this_landmarks = load_fit_landmarks(s["landmarks_path"], candidate_names)
            common = sorted(set(this_landmarks) & set(ref_landmarks))
            if len(common) < MIN_POINTS_FOR_FIT:
                print(f"✗  only {len(common)} shared landmarks with reference (need >= {MIN_POINTS_FOR_FIT})")
                report.append({"scan": scan_name, "status": "failed",
                               "error": f"only {len(common)} shared landmarks"})
                continue

            P = np.array([this_landmarks[n] for n in common])
            Q = np.array([ref_landmarks[n] for n in common])
            R, t = compute_rigid_transform(P, Q)
            rms_mm, max_mm = fit_residual_mm(P, Q, R, t)

            pts, colors = read_ply_binary(s["ply_path"])
            pts_aligned = apply_transform(pts, R, t)
            write_ply_binary(pts_aligned, colors, out_ply)
            save_transform_json(
                s["scan_dir"] / f"{scan_name}_alignment.json",
                R, t, reference["scan_name"], rms_mm, max_mm, common
            )

            if args.debug:
                save_preview(pts_aligned, colors, s["scan_dir"] / f"{scan_name}_aligned_preview.png")

            warn = f"  ⚠ high residual" if rms_mm > HIGH_RESIDUAL_WARN_MM else ""
            print(f"✓  {len(common)} pts  RMS={rms_mm:.1f}mm  max={max_mm:.1f}mm{warn}")
            report.append({"scan": scan_name, "status": "ok", "rms_mm": rms_mm,
                           "n_points": len(common), "high_residual": rms_mm > HIGH_RESIDUAL_WARN_MM})
        except Exception as e:
            print(f"✗  {e}")
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok       = [r for r in report if r["status"] == "ok"]
    skip     = sum(1 for r in report if r["status"] == "skipped")
    failed   = [r for r in report if r["status"] == "failed"]
    flagged  = [r for r in ok if r.get("high_residual")]

    print(f"\n{'='*60}")
    print(f"  Aligned  : {len(ok)}")
    if flagged:
        print(f"  High residual (>{HIGH_RESIDUAL_WARN_MM}mm, check landmarks): {len(flagged)}")
        for r in flagged: print(f"    ⚠  {r['scan']} — RMS {r['rms_mm']:.1f}mm")
    print(f"  Skipped  : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed   : {len(failed)}")
        for r in failed: print(f"    ✗  {r['scan']} — {r['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()