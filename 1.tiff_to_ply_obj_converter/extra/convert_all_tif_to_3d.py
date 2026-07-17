#!/usr/bin/env python3
"""
convert_to_3d.py
================
Batch converts all Cyberware range files to both PLY and OBJ formats.

READS FROM:
  Dataset/
    pat1/
        pat1day0C/
            pat1day0C        <- Cyberware range file (no extension)
            pat1day0C.tif    <- color texture (must be upright/straight)
        pat1day28A/
            pat1day28A
            pat1day28A.tif

WRITES TO:
  Dataset/3d_scans/
    ply/
        pat1/
            pat1day0C.ply
            pat1day28A.ply
            pat1day28C2.ply
        pat2/
            pat2day0A.ply
    obj/
        pat1/
            pat1day0C.obj
            pat1day28A.obj
        pat2/
            pat2day0A.obj

USAGE:
  python convert_to_3d.py
  python convert_to_3d.py --patient pat1
  python convert_to_3d.py --patient pat1 --format ply
  python convert_to_3d.py --patient pat1 --format obj
  python convert_to_3d.py --dataset "D:/NahidW/Dataset"
  python convert_to_3d.py --overwrite

REQUIREMENTS:
  pip install numpy pillow
"""

import sys, math, os, re, argparse
import numpy as np
from PIL import Image
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
OUTPUT_DIR  = DATASET_DIR / "3d_scans"

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4   # 457.2 mm
SCANNER_RADIUS_MM = 9  * 25.4   # 228.6 mm


# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


def discover_scans(dataset_dir: Path, patient_filter=None) -> list:
    """
    Find all scan folders: Dataset/<patient>/<scan>/<scan> + <scan>.tif
    Returns sorted list of (patient_id, scan_name, range_path, tif_path).
    """
    scans = []
    for scan_folder in sorted(dataset_dir.glob("*/*/")):
        scan_name = scan_folder.name
        m = SCAN_RE.match(scan_name)
        if not m:
            continue
        patient_id = m.group("patient").lower()
        if patient_filter and patient_id != patient_filter.lower():
            continue
        range_path = scan_folder / scan_name
        tif_path   = scan_folder / f"{scan_name}.tif"
        if not range_path.exists():
            print(f"  ⚠ Range file missing: {range_path}")
            continue
        if not tif_path.exists():
            print(f"  ⚠ TIF missing: {tif_path}")
            continue
        scans.append((patient_id, scan_name, range_path, tif_path))
    scans.sort(key=lambda x: (x[0], x[1]))
    return scans


# ─────────────────────────────────────────────────────────────────────────────
#  CYBERWARE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_header(filepath: Path):
    with open(filepath, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"Cyberware"):
        raise ValueError(f"Not a Cyberware range file: {filepath}")
    idx = raw.find(b"DATA=\n")
    if idx == -1:
        raise ValueError("Could not find DATA= marker.")
    header_end = idx + len(b"DATA=\n")
    params = {}
    for line in raw[:header_end].decode("ascii", errors="replace").split("\n"):
        if "=" in line and not line.startswith("DATA"):
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params, header_end, raw


def load_scan(range_path: Path, tif_path: Path):
    """
    Parse Cyberware range file and paired TIF.
    Returns (pts, colors, n_valid) where:
      pts    = (N, 3) float32 XYZ in mm
      colors = (N, 3) uint8  RGB
    """
    params, header_end, raw = parse_header(range_path)

    NLG    = int(params["NLG"])     # angular steps  (theta axis)
    NLT    = int(params["NLT"])     # height steps   (Z axis)
    RSHIFT = int(params["RSHIFT"])
    LGINCR = int(params["LGINCR"])

    r_scale_mm = LGINCR / 32768.0
    z_scale_mm = SCANNER_HEIGHT_MM / NLT
    theta_step = (2.0 * math.pi) / NLG

    # ── Range data ────────────────────────────────────────────────────────────
    data = (np.frombuffer(raw[header_end:header_end + NLG * NLT * 2], dtype=">u2")
              .reshape(NLG, NLT)
              .astype(np.float32))

    valid_mask = (data != INVALID_SENTINEL) & (data > 0)
    radius_mm  = np.where(valid_mask, (data / (2 ** RSHIFT)) * r_scale_mm, np.nan)
    valid_mask = (~np.isnan(radius_mm) &
                  (radius_mm > 0) &
                  (radius_mm <= SCANNER_RADIUS_MM))

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise RuntimeError("No valid range points found.")

    # ── Cylindrical → Cartesian ───────────────────────────────────────────────
    Z_grid, THETA = np.meshgrid(
        np.arange(NLT) * z_scale_mm,
        np.arange(NLG) * theta_step
    )
    X = np.where(valid_mask, radius_mm * np.cos(THETA), np.nan)
    Y = np.where(valid_mask, radius_mm * np.sin(THETA), np.nan)

    rows, cols = np.where(valid_mask)   # rows = angular (0..NLG-1)
                                         # cols = height  (0..NLT-1)
    pts = np.column_stack([X[valid_mask],
                            Y[valid_mask],
                            Z_grid[valid_mask]])

    # ── Color mapping ─────────────────────────────────────────────────────────
    # TIF shape = (NLT, NLG) = (height rows, angular cols)
    # Range rows = angular index → TIF column axis
    # Range cols = height index  → TIF row axis (flipped: scanner is bottom-up,
    #                                            TIF is top-down)
    color      = np.array(Image.open(tif_path).convert("RGB"))
    ch, cw     = color.shape[:2]   # ch = NLT, cw = NLG

    tif_row    = (ch - 1 - (cols * ch / NLT).astype(int).clip(0, ch - 1))
    tif_col    = (rows * cw / NLG).astype(int).clip(0, cw - 1)
    colors     = color[tif_row, tif_col].astype(np.uint8)

    return pts, colors, n_valid


# ─────────────────────────────────────────────────────────────────────────────
#  WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_ply(pts: np.ndarray, colors: np.ndarray, output_path: Path):
    """Write binary PLY with XYZ + RGB per vertex."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    packed = np.zeros(len(pts), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
    ])
    packed["x"], packed["y"], packed["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    packed["r"], packed["g"], packed["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())
    return output_path.stat().st_size


def write_obj(pts: np.ndarray, colors: np.ndarray, output_path: Path):
    """Write OBJ with vertex colors (v x y z r g b)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Cyberware scan — {output_path.stem}\n")
        f.write(f"# {len(pts):,} points\n\n")
        rgb = colors.astype(np.float32) / 255.0
        for i in range(len(pts)):
            f.write(f"v {pts[i,0]:.4f} {pts[i,1]:.4f} {pts[i,2]:.4f} "
                    f"{rgb[i,0]:.4f} {rgb[i,1]:.4f} {rgb[i,2]:.4f}\n")
    return output_path.stat().st_size


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch convert Cyberware range files to PLY and/or OBJ"
    )
    p.add_argument("--patient",  type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--format",   type=str, default="both",
                   choices=["ply", "obj", "both"],
                   help="Output format (default: both)")
    p.add_argument("--dataset",  type=str, default=None,
                   help="Override dataset directory")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing files (default: skip)")
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        OUTPUT_DIR  = DATASET_DIR / "3d_scans"

    do_ply = args.format in ("ply", "both")
    do_obj = args.format in ("obj", "both")

    ply_dir = OUTPUT_DIR / "ply"
    obj_dir = OUTPUT_DIR / "obj"

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Cyberware → 3D Converter")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Formats : {'PLY + OBJ' if do_ply and do_obj else args.format.upper()}")
    print(f"{'='*55}")

    scans = discover_scans(DATASET_DIR, patient_filter=args.patient)
    if not scans:
        print("\n  No scan folders found.")
        print("  Expected: Dataset/pat1/pat1day0C/pat1day0C  (range file)")
        sys.exit(1)

    print(f"\n  Found {len(scans)} scan(s)\n")

    report   = []
    prev_pat = None

    for patient_id, scan_name, range_path, tif_path in scans:

        if patient_id != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id

        # Check if outputs already exist
        ply_out = ply_dir / patient_id / f"{scan_name}.ply"
        obj_out = obj_dir / patient_id / f"{scan_name}.obj"

        ply_exists = ply_out.exists()
        obj_exists = obj_out.exists()

        need_ply = do_ply and (not ply_exists or args.overwrite)
        need_obj = do_obj and (not obj_exists or args.overwrite)

        if not need_ply and not need_obj:
            sizes = []
            if ply_exists: sizes.append(f"PLY {ply_out.stat().st_size/1e6:.1f}MB")
            if obj_exists: sizes.append(f"OBJ {obj_out.stat().st_size/1e6:.1f}MB")
            print(f"  ↷ {scan_name}  already exists ({', '.join(sizes)}) — skipping")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"  → {scan_name}", end="  ", flush=True)

        try:
            pts, colors, n_valid = load_scan(range_path, tif_path)

            results = []
            if need_ply:
                sz = write_ply(pts, colors, ply_out)
                results.append(f"PLY {sz/1e6:.1f}MB")
            if need_obj:
                sz = write_obj(pts, colors, obj_out)
                results.append(f"OBJ {sz/1e6:.1f}MB")

            print(f"✓  {n_valid:,} pts  |  {' + '.join(results)}")
            report.append({"scan": scan_name, "status": "success",
                           "points": n_valid})

        except Exception as e:
            print(f"✗  {e}")
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Converted : {ok}")
    print(f"  Skipped   : {skip}  (use --overwrite to redo)")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['error']}")
    print(f"\n  Output    : {OUTPUT_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()





# python convert_to_3d.py                                      # all patients, PLY + OBJ (default)
# python convert_to_3d.py --format ply                         # PLY only
# python convert_to_3d.py --format obj                         # OBJ only
# python convert_to_3d.py --patient pat1                       # one patient, PLY + OBJ
# python convert_to_3d.py --patient pat1 --format ply          # one patient, PLY only
# python convert_to_3d.py --patient pat1 --format obj          # one patient, OBJ only
# python convert_to_3d.py --overwrite                          # redo all existing files
# python convert_to_3d.py --dataset "D:/NahidW/Dataset"        # manual dataset path
# python convert_to_3d.py --dataset "D:/NahidW/Dataset" --patient pat1 --format obj --overwrite