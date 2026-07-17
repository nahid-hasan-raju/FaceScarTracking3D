#!/usr/bin/env python3
"""
convert_all_to_ply.py
=====================
Batch converts all Cyberware range files in the Dataset to colored PLY files.

READS FROM:
  Dataset/
    pat1/
        pat1day0C/
            pat1day0C        <- range file (no extension)
            pat1day0C.tif    <- color texture
        pat1day28A/
            pat1day28A
            pat1day28A.tif

WRITES TO:
  Dataset/ply_files/
    pat1/
        pat1day0C.ply
        pat1day28A.ply
        pat1day28C2.ply
    pat2/
        pat2day0A.ply

USAGE:
  python convert_all_to_ply.py
  python convert_all_to_ply.py --patient pat1
  python convert_all_to_ply.py --dataset "D:/NahidW/Dataset"

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
OUTPUT_DIR  = DATASET_DIR / "ply_files"

# ─────────────────────────────────────────────────────────────────────────────
#  CYBERWARE CONVERSION (from original script)
# ─────────────────────────────────────────────────────────────────────────────

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4
SCANNER_RADIUS_MM = 9  * 25.4


def parse_header(filepath):
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


def cyberware_to_ply(range_path: Path, color_path: Path, output_path: Path):
    params, header_end, raw = parse_header(range_path)
    NLG    = int(params["NLG"])
    NLT    = int(params["NLT"])
    RSHIFT = int(params["RSHIFT"])
    LGINCR = int(params["LGINCR"])

    N_THETA    = NLG
    N_Z        = NLT
    r_scale_mm = LGINCR / 32768.0
    z_scale_mm = SCANNER_HEIGHT_MM / N_Z
    theta_step = (2.0 * math.pi) / N_THETA

    data = (np.frombuffer(raw[header_end:header_end + NLG*NLT*2], dtype=">u2")
              .reshape(NLG, NLT)
              .astype(np.float32))

    valid_mask = (data != INVALID_SENTINEL) & (data > 0)
    radius_mm  = np.where(valid_mask, (data / (2**RSHIFT)) * r_scale_mm, np.nan)
    valid_mask = ~np.isnan(radius_mm) & (radius_mm > 0) & (radius_mm <= SCANNER_RADIUS_MM)

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise RuntimeError("No valid range points found.")

    Z_grid, THETA = np.meshgrid(
        np.arange(N_Z)     * z_scale_mm,
        np.arange(N_THETA) * theta_step
    )
    X = np.where(valid_mask, radius_mm * np.cos(THETA), np.nan)
    Y = np.where(valid_mask, radius_mm * np.sin(THETA), np.nan)
    Z = Z_grid

    color  = np.array(Image.open(color_path).convert("RGB"))
    ch, cw = color.shape[:2]   # ch = NLT (height), cw = NLG (angular/width)

    rows, cols = np.where(valid_mask)  # rows = angular (0..NLG-1), cols = height (0..NLT-1)
    pts        = np.column_stack([X[valid_mask], Y[valid_mask], Z[valid_mask]])

    # TIF shape = (NLT, NLG) = (height rows, angular cols)
    # Range rows = angular index -> TIF col axis
    # Range cols = height index  -> TIF row axis, but FLIPPED
    #   (scanner stores bottom-to-top, TIF is top-to-bottom)
    tif_row = (ch - 1 - (cols * ch / N_Z).astype(int).clip(0, ch - 1))
    tif_col = (rows * cw / N_THETA).astype(int).clip(0, cw - 1)
    colors   = color[tif_row, tif_col].astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    packed = np.zeros(len(pts), dtype=[
        ("x","<f4"),("y","<f4"),("z","<f4"),
        ("r","u1"), ("g","u1"), ("b","u1")
    ])
    packed["x"],packed["y"],packed["z"] = pts[:,0],pts[:,1],pts[:,2]
    packed["r"],packed["g"],packed["b"] = colors[:,0],colors[:,1],colors[:,2]

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())

    return n_valid, os.path.getsize(output_path)


# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


def discover_scans(dataset_dir: Path, patient_filter=None) -> list:
    """
    Find all scan folders matching pat<N>day<D><V> inside patient subfolders.
    Returns list of (patient_id, scan_name, range_path, tif_path).
    """
    scans = []
    # Dataset/pat1/pat1day0C/
    for scan_folder in sorted(dataset_dir.glob("*/*/")):
        scan_name = scan_folder.name
        m = SCAN_RE.match(scan_name)
        if not m:
            continue

        patient_id = m.group("patient").lower()
        if patient_filter and patient_id != patient_filter.lower():
            continue

        range_path = scan_folder / scan_name          # no extension
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
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch convert Cyberware range files to PLY"
    )
    p.add_argument("--patient", type=str, default=None,
                   help="Process only this patient (e.g. pat1)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override dataset directory")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing PLY files (default: skip)")
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, OUTPUT_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        OUTPUT_DIR  = DATASET_DIR / "ply_files"

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Batch Cyberware → PLY Converter")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
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

        # Print patient header when patient changes
        if patient_id != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id

        out_path = OUTPUT_DIR / patient_id / f"{scan_name}.ply"

        # Skip if already exists and not overwriting
        if out_path.exists() and not args.overwrite:
            size_mb = out_path.stat().st_size / 1e6
            print(f"  ↷ {scan_name}.ply  already exists ({size_mb:.1f} MB) — skipping")
            report.append({"scan": scan_name, "status": "skipped"})
            continue

        print(f"  → {scan_name}", end="  ", flush=True)

        try:
            n_pts, size_bytes = cyberware_to_ply(range_path, tif_path, out_path)
            print(f"✓  {n_pts:,} pts  {size_bytes/1e6:.1f} MB")
            report.append({"scan": scan_name, "status": "success",
                           "points": n_pts, "output": str(out_path)})
        except Exception as e:
            print(f"✗  {e}")
            report.append({"scan": scan_name, "status": "failed", "error": str(e)})

    # Summary
    ok     = sum(1 for r in report if r["status"] == "success")
    skip   = sum(1 for r in report if r["status"] == "skipped")
    failed = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Converted : {ok}")
    print(f"  Skipped   : {skip}  (already exist, use --overwrite to redo)")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for r in failed:
            print(f"    {r['scan']} — {r['error']}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()