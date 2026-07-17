#!/usr/bin/env python3
"""
visualize_alignment.py
========================
Visual QC for step3's alignment: overlays every timepoint of a given patient
+ pose (variant) in a single view, each timepoint in a distinct color.

WHY OVERLAY, NOT SIDE-BY-SIDE:
  If alignment worked, stable anatomy (nose, eyes, forehead, hairline) should
  sit almost exactly on top of itself across timepoints -- you'd see mostly
  one blended color where they overlap. Only the burn region (and anything
  that genuinely changed) should show visibly separated colors. This is a
  much more direct check than eyeballing separate single-scan renders side by
  side and trying to remember if the nose was in the same spot.

READS:
    PAT01/D00/PAT01_D00_A/PAT01_D00_A_aligned.ply   (per timepoint, from step3)
    PAT01/D00/PAT01_D00_A/landmarks/PAT01_D00_A_landmarks.json  (for view direction)

WRITES:
    {dataset}/alignment_qc/PAT01_A_overlay.png   (one per patient+variant)

USAGE:
    python visualize_alignment.py --dataset "D:/..." --patient PAT01
    python visualize_alignment.py --dataset "D:/..." --patient PAT01 --variant A
    python visualize_alignment.py --dataset "D:/..."   # every patient/variant found

REQUIREMENTS:
    pip install numpy matplotlib
    (needs step3's *_aligned.ply to already exist)
"""

import sys, math, re, argparse, json
import numpy as np
from pathlib import Path

SCAN_RE = re.compile(r"^(PAT\d+)_([DM]\d+)_([A-Z][A-Z0-9]?)$")

# Distinct, easily-told-apart colors per timepoint (cycles if more needed)
OVERLAY_COLORS = [
    (0.85, 0.10, 0.10),  # red        -- earliest / reference
    (0.10, 0.45, 0.85),  # blue
    (0.10, 0.75, 0.25),  # green
    (0.90, 0.60, 0.05),  # orange
    (0.55, 0.15, 0.75),  # purple
    (0.10, 0.75, 0.75),  # teal
]


def timepoint_sort_key(tp):
    prefix, num = tp[0], int(tp[1:])
    return (0 if prefix == "D" else 1, num)


# ─────────────────────────────────────────────────────────────────────────────
#  DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_aligned_scans(dataset_dir, patient_filter=None, variant_filter=None):
    """Finds every *_aligned.ply, grouped by (patient, variant)."""
    groups = {}
    for scan_dir in sorted(dataset_dir.glob("PAT*/[DM]*/PAT*_[DM]*_*")):
        if not scan_dir.is_dir():
            continue
        m = SCAN_RE.match(scan_dir.name)
        if not m:
            continue
        patient_id, timepoint_id, variant = m.group(1), m.group(2), m.group(3)
        if patient_filter and patient_id != patient_filter.upper():
            continue
        if variant_filter and variant != variant_filter.upper():
            continue

        aligned_path = scan_dir / f"{scan_dir.name}_aligned.ply"
        landmarks_path = scan_dir / "landmarks" / f"{scan_dir.name}_landmarks.json"
        if not aligned_path.exists():
            continue  # not every scan needs to have been aligned successfully

        groups.setdefault((patient_id, variant), []).append({
            "timepoint": timepoint_id, "scan_name": scan_dir.name,
            "aligned_path": aligned_path, "landmarks_path": landmarks_path,
        })

    for key in groups:
        groups[key].sort(key=lambda s: timepoint_sort_key(s["timepoint"]))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
#  PLY I/O
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
    return pts


# ─────────────────────────────────────────────────────────────────────────────
#  VIEW DIRECTION  (from the reference scan's nose, so every group gets a
#  consistent front-facing projection axis)
# ─────────────────────────────────────────────────────────────────────────────

def get_nose_theta(landmarks_path):
    if not landmarks_path.exists():
        return None
    data = json.loads(landmarks_path.read_text())
    stable = data.get("stable_landmark_names", {})
    idx = stable.get("nose_tip")
    if idx is None:
        return None
    xyz = data.get("landmarks_3d_mm", {}).get(str(idx))
    if xyz is None:
        return None
    x, y, _ = xyz
    return math.atan2(y, x)


# ─────────────────────────────────────────────────────────────────────────────
#  OVERLAY RENDER
# ─────────────────────────────────────────────────────────────────────────────

def save_overlay(group_scans, nose_theta, out_path, front_window_deg=100,
                  point_size=1.0, alpha=0.35, subsample=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view_dir   = np.array([math.cos(nose_theta), math.sin(nose_theta), 0.0])
    right_axis = np.array([-math.sin(nose_theta), math.cos(nose_theta), 0.0])

    fig, ax = plt.subplots(figsize=(7, 8))
    for i, s in enumerate(group_scans):
        pts = read_ply_binary(s["aligned_path"])[::subsample]  # subsample for speed/clarity
        point_theta = np.arctan2(pts[:, 1], pts[:, 0])
        diff = (point_theta - nose_theta + np.pi) % (2 * np.pi) - np.pi
        pts = pts[np.abs(diff) < np.radians(front_window_deg)]

        right = pts @ right_axis
        up    = pts[:, 2]
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        ax.scatter(right, up, s=point_size, c=[color], alpha=alpha, linewidths=0,
                   label=f"{s['timepoint']} ({s['scan_name']})")

    ax.set_aspect("equal")
    ax.set_xlabel("right (mm)"); ax.set_ylabel("up / height (mm)")
    ax.set_title(f"Alignment overlay -- {len(group_scans)} timepoint(s)\n"
                 f"Good alignment = mostly blended color; separated colors = misalignment or real change")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--patient", default=None)
    p.add_argument("--variant", default=None, help="e.g. A -- limit to one pose")
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}"); sys.exit(1)

    groups = discover_aligned_scans(dataset_dir, args.patient, args.variant)
    if not groups:
        print("No aligned scans found (run step3 first)."); sys.exit(1)

    out_dir = dataset_dir / "alignment_qc"
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}\n  Alignment overlay QC  ({len(groups)} patient/pose group(s))\n{'='*60}\n")

    for (patient, variant), scans in sorted(groups.items()):
        if len(scans) < 2:
            print(f"  ↷  {patient} pose {variant} -- only 1 timepoint, nothing to compare against")
            continue

        nose_theta = None
        for s in scans:
            nose_theta = get_nose_theta(s["landmarks_path"])
            if nose_theta is not None:
                break
        if nose_theta is None:
            print(f"  ✗  {patient} pose {variant} -- no landmarks available for view direction")
            continue

        out_path = out_dir / f"{patient}_{variant}_overlay.png"
        try:
            save_overlay(scans, nose_theta, out_path)
            print(f"  ✓  {patient} pose {variant}  ({len(scans)} timepoints: "
                  f"{', '.join(s['timepoint'] for s in scans)}) -> {out_path.name}")
        except Exception as e:
            print(f"  ✗  {patient} pose {variant} -- {e}")

    print(f"\n{'='*60}")
    print(f"  Saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
