#!/usr/bin/env python3
"""
check_alignment.py
==================
Visual sanity check for 3D landmark alignment.

Projects two scans' point clouds onto a frontal 2D view (looking along Y axis)
and overlays them in different colours.

  Blue  = reference scan
  Red   = other scan BEFORE alignment
  Green = other scan AFTER  alignment

If alignment is working: green should sit on top of blue in the face region.

NOTE: A, B, C, D variants are different postures — only compare within the
same variant (A vs A, B vs B, etc.). The batch mode handles this automatically.

USAGE:
  # Single pair
  python check_alignment.py --ref ".../PAT01/D00/PAT01_D00_A" --other ".../PAT01/D14/PAT01_D14_A" --outdir ".../alignment_checks"

  # Batch — all scans for a patient, grouped by variant
  python check_alignment.py --dataset "D:/.../" --patient PAT01 --outdir ".../PAT01/alignment_checks"
"""

import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

import tifffile
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "3.burn_segmentation_3d_pipeline"))
from utils.cyberware import range_to_3d

SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _variant_of(scan_dir: Path):
    m = SCAN_RE.match(scan_dir.name)
    return m.group("variant").upper() if m else None


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


def load_pointcloud(scan_dir: Path):
    name = scan_dir.name
    tif_path   = scan_dir / f"{name}.tif"
    range_path = scan_dir / name
    if not tif_path.exists():
        raise FileNotFoundError(f"TIF not found: {tif_path}")
    if not range_path.exists():
        raise FileNotFoundError(f"Range file not found: {range_path}")
    img_rgb = load_tiff_rgb(tif_path)
    pts, colors, *_ = range_to_3d(range_path, img_rgb)
    return pts


def load_alignment(scan_dir: Path):
    align_path = scan_dir / f"{scan_dir.name}_alignment.json"
    if not align_path.exists():
        return None, None, "no_file"
    data   = json.load(open(align_path))
    status = data.get("status", "unknown")
    if status == "failed" or data.get("R") is None:
        return None, None, status
    return np.array(data["R"]), np.array(data["t"]), status


def apply_transform(pts, R, t):
    if R is None or t is None:
        return pts
    return (np.array(R) @ pts.T).T + np.array(t)


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER
# ─────────────────────────────────────────────────────────────────────────────

def project_frontal(pts, canvas_size, margin, x_range, z_range):
    x, z = pts[:, 0], pts[:, 2]
    span  = max(x_range[1]-x_range[0], z_range[1]-z_range[0], 1)
    scale = (canvas_size - 2*margin) / span
    col = ((x - x_range[0]) * scale + margin).astype(int)
    row = (canvas_size - margin - (z - z_range[0]) * scale).astype(int)
    valid = (col >= 0) & (col < canvas_size) & (row >= 0) & (row < canvas_size)
    return col[valid], row[valid]


def render_overlay(ref_pts, other_pts_raw, other_pts_aligned,
                   ref_name, other_name, variant, status,
                   canvas_size=700, margin=40, subsample=6):
    ref_s = ref_pts[::subsample]
    raw_s = other_pts_raw[::subsample]
    aln_s = other_pts_aligned[::subsample]

    all_pts = np.vstack([ref_s, aln_s])
    x_range = (all_pts[:,0].min(), all_pts[:,0].max())
    z_range = (all_pts[:,2].min(), all_pts[:,2].max())

    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 245

    def draw(pts, color):
        cols, rows = project_frontal(pts, canvas_size, margin, x_range, z_range)
        for c, r in zip(cols, rows):
            cv2.circle(canvas, (c, r), 1, color, -1)

    draw(raw_s, (180, 60,  60))
    draw(aln_s, (40,  160, 60))
    draw(ref_s, (50,  80,  200))

    font = cv2.FONT_HERSHEY_SIMPLEX
    legends = [
        ((50,80,200),  f"Reference: {ref_name}"),
        ((40,160,60),  f"Aligned:   {other_name}  [{status}]"),
        ((180,60,60),  f"Raw:       {other_name}"),
    ]
    for idx, (color, text) in enumerate(legends):
        y = canvas_size - 20 - (len(legends)-1-idx)*22
        cv2.rectangle(canvas, (16, y-12), (28, y+2), color[::-1], -1)
        cv2.putText(canvas, text, (34, y), font, 0.42, (40,40,40), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Alignment check [{variant}]: {ref_name} vs {other_name}",
                (margin, 24), font, 0.50, (30,30,30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "blue=reference  green=aligned  red=raw",
                (margin, 42), font, 0.40, (100,100,100), 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  PAIR CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_pair(ref_dir: Path, other_dir: Path, out_dir: Path):
    variant = _variant_of(ref_dir) or "?"
    if _variant_of(other_dir) != _variant_of(ref_dir):
        print(f"     ⚠ Warning: comparing different variants "
              f"({_variant_of(ref_dir)} vs {_variant_of(other_dir)}) — "
              f"results may not be meaningful")

    print(f"  Loading {ref_dir.name} ...", end=" ", flush=True)
    ref_pts = load_pointcloud(ref_dir)
    R_ref, t_ref, _ = load_alignment(ref_dir)
    ref_pts = apply_transform(ref_pts, R_ref, t_ref)
    print("done")

    print(f"  Loading {other_dir.name} ...", end=" ", flush=True)
    other_pts = load_pointcloud(other_dir)
    R, t, status = load_alignment(other_dir)
    other_aligned = apply_transform(other_pts, R, t)
    print("done")

    print(f"  Rendering ...", end=" ", flush=True)
    canvas = render_overlay(ref_pts, other_pts, other_aligned,
                            ref_dir.name, other_dir.name, variant, status)
    out_path = out_dir / f"{ref_dir.name}_vs_{other_dir.name}_check.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"saved → {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH — groups scans by variant, uses each variant's D00 as reference
# ─────────────────────────────────────────────────────────────────────────────

def batch_check(dataset_dir: Path, patient: str, out_dir: Path):
    pat_dir   = dataset_dir / patient
    scan_dirs = sorted([p for p in pat_dir.glob("*/*") if p.is_dir()])
    out_dir.mkdir(parents=True, exist_ok=True)

    # group by variant
    by_variant = {}
    for sd in scan_dirs:
        v = _variant_of(sd)
        if v:
            by_variant.setdefault(v, []).append(sd)

    print(f"  Variants found: {sorted(by_variant.keys())}\n")

    for variant, scans in sorted(by_variant.items()):
        # find the D00 scan for this variant to use as reference
        ref = next((s for s in scans
                    if re.match(r".*_D00_", s.name, re.IGNORECASE)), None)
        if ref is None:
            print(f"  [{variant}] No D00 scan found — skipping variant")
            continue

        print(f"  ── Variant {variant}  (reference: {ref.name})")
        others = [s for s in scans if s != ref]
        for other in others:
            print(f"     {other.name}")
            try:
                check_pair(ref, other, out_dir)
            except FileNotFoundError as e:
                print(f"     skipped: {e}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visual alignment checker")
    p.add_argument("--ref",     default=None, help="Reference scan dir (single-pair mode)")
    p.add_argument("--other",   default=None, help="Other scan dir (single-pair mode)")
    p.add_argument("--dataset", default=None)
    p.add_argument("--patient", default=None)
    p.add_argument("--outdir",  default=None,
                   help="Output folder for PNGs. "
                        "Default: next to script (single) or "
                        "<dataset>/<patient>/alignment_checks/ (batch)")
    args = p.parse_args()

    if args.ref and args.other:
        out_dir = Path(args.outdir) if args.outdir else Path(__file__).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        check_pair(Path(args.ref), Path(args.other), out_dir)

    elif args.dataset and args.patient:
        out_dir = (Path(args.outdir) if args.outdir
                   else Path(args.dataset) / args.patient / "alignment_checks")
        batch_check(Path(args.dataset), args.patient, out_dir)

    else:
        p.error(
            "Provide either:\n"
            "  --ref <dir> --other <dir>           (single pair)\n"
            "  --dataset <dir> --patient PAT01     (batch, auto-groups by variant)"
        )


if __name__ == "__main__":
    main()