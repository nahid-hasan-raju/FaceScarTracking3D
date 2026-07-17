#!/usr/bin/env python3
"""
convert_to_3d_v2.py
===================
Converts Cyberware range files to PLY. Stores .ply in the same scan subfolder.

STRUCTURE:
    PAT01/D00/PAT01_D00_A/
        PAT01_D00_A        <- range file
        PAT01_D00_A.tif
        PAT01_D00_A.lnd
        PAT01_D00_A.ply    <- written here

USAGE:
    python convert_to_3d_v2.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
    python convert_to_3d_v2.py --dataset "D:/..." --patient PAT01
    python convert_to_3d_v2.py --dataset "D:/..." --patient PAT01 --timepoint D00
    python convert_to_3d_v2.py --dataset "D:/..." --patient PAT01 --timepoint D00 --scan A
    python convert_to_3d_v2.py --dataset "D:/..." --overwrite

REQUIREMENTS:
    pip install numpy pillow
"""

import sys, math, re, argparse
import numpy as np
from PIL import Image
from pathlib import Path

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4
SCANNER_RADIUS_MM = 9  * 25.4

SCAN_RE = re.compile(r"^(PAT\d+)_([DM]\d+)_([A-Z][A-Z0-9]?)$")


# ── Discovery ─────────────────────────────────────────────────────────────────

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
            print(f"  ⚠  range file missing: {range_path}")
            continue
        if not tif_path.exists():
            print(f"  ⚠  tif missing: {tif_path}")
            continue

        scans.append((patient_id, timepoint_id, scan_dir.name, range_path, tif_path, scan_dir))

    return sorted(scans, key=lambda x: (x[0], x[1], x[2]))


# ── Cyberware parser ──────────────────────────────────────────────────────────

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


def load_scan(range_path, tif_path):
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

    if not valid_mask.any():
        raise RuntimeError("No valid range points found.")

    Z_grid, THETA = np.meshgrid(np.arange(NLT) * z_scale_mm, np.arange(NLG) * theta_step)
    X = np.where(valid_mask, radius_mm * np.cos(THETA), np.nan)
    Y = np.where(valid_mask, radius_mm * np.sin(THETA), np.nan)

    rows, cols = np.where(valid_mask)
    pts = np.column_stack([X[valid_mask], Y[valid_mask], Z_grid[valid_mask]])

    color  = np.array(Image.open(tif_path).convert("RGB"))
    ch, cw = color.shape[:2]
    tif_row = (ch - 1 - (cols * ch / NLT).astype(int).clip(0, ch - 1))
    tif_col = (rows * cw / NLG).astype(int).clip(0, cw - 1)
    colors  = color[tif_row, tif_col].astype(np.uint8)

    return pts, colors, int(valid_mask.sum())


# ── Writer ────────────────────────────────────────────────────────────────────

def write_ply(pts, colors, output_path):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    packed = np.zeros(len(pts), dtype=[
        ("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")
    ])
    packed["x"], packed["y"], packed["z"] = pts[:,0], pts[:,1], pts[:,2]
    packed["r"], packed["g"], packed["b"] = colors[:,0], colors[:,1], colors[:,2]
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())
    return output_path.stat().st_size


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True)
    p.add_argument("--patient",    default=None)
    p.add_argument("--timepoint",  default=None)
    p.add_argument("--scan",       default=None)
    p.add_argument("--overwrite",  action="store_true")
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset_dir}")
    print(f"  Format   : PLY (stored in each scan subfolder)")
    print(f"{'='*60}")

    scans = discover_scans(dataset_dir, args.patient, args.timepoint, args.scan)
    if not scans:
        print("No scans found."); sys.exit(1)
    print(f"\n  Found {len(scans)} scan(s)\n")

    report = []
    prev_pat, prev_tp = None, None

    for patient_id, timepoint_id, scan_name, range_path, tif_path, scan_dir in scans:
        if patient_id != prev_pat:
            print(f"\n{'─'*60}\n  {patient_id}")
            prev_pat, prev_tp = patient_id, None
        if timepoint_id != prev_tp:
            print(f"    [{timepoint_id}]")
            prev_tp = timepoint_id

        ply_out = scan_dir / f"{scan_name}.ply"   # same folder as .tif

        if ply_out.exists() and not args.overwrite:
            print(f"      ↷  {scan_name}  already done ({ply_out.stat().st_size/1e6:.1f}MB) — skip")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"      →  {scan_name}", end="  ", flush=True)
        try:
            pts, colors, n_valid = load_scan(range_path, tif_path)
            sz = write_ply(pts, colors, ply_out)
            print(f"✓  {n_valid:,} pts  |  {sz/1e6:.1f}MB")
            report.append({"scan": scan_name, "status": "ok", "pts": n_valid})
        except Exception as e:
            print(f"✗  {e}")
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok     = sum(1 for r in report if r["status"] == "ok")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*60}")
    print(f"  Converted : {ok}")
    print(f"  Skipped   : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for r in failed: print(f"    ✗  {r['scan']} — {r['error']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()