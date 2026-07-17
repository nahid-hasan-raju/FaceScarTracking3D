#!/usr/bin/env python3
"""
align_3d_front.py
=================
Align all 3D face scans to front-facing orientation.

FIXED ROTATIONS (same for all scans):
  Step 1: -90° around X axis
  Step 2: +90° around Y axis

FINE CORRECTION (per-scan, optional):
  Add to FINE_YAW_CONFIG after reviewing snapshots.
  Positive = rotate face left, Negative = rotate face right.

READS FROM:
  Dataset/3d_scans/ply/pat1/pat1day0C.ply   (--format ply or both)
  Dataset/3d_scans/obj/pat1/pat1day0C.obj   (--format obj or both)

WRITES TO:
  PLY input -> ply/pat1/pat1day0C_aligned.ply + _front.png
  OBJ input -> obj/pat1/pat1day0C_aligned.obj + _front.png

USAGE:
  python align_3d_front.py                        # PLY + OBJ (default)
  python align_3d_front.py --format ply           # PLY only
  python align_3d_front.py --format obj           # OBJ only
  python align_3d_front.py --patient pat1
  python align_3d_front.py --overwrite
  python align_3d_front.py --show
  python align_3d_front.py --fine-yaw 15.0
  python align_3d_front.py --format obj --patient pat1 --overwrite

REQUIREMENTS:
  pip install numpy open3d opencv-python
"""

import sys, re, argparse, time
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("D:/NahidW/Dataset")
SCANS_DIR   = DATASET_DIR / "3d_scans"
PLY_DIR     = SCANS_DIR / "ply"
OBJ_DIR     = SCANS_DIR / "obj"

IMG_W = 800
IMG_H = 1000
ZOOM  = 0.55

FINE_YAW_CONFIG = {
    # "pat1day0C":   -12.0,
    # "pat1day28A":   -3.0,
    # "pat1day28C2": -35.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(
    r"^(?P<patient>pat\d+)day(?P<day>\d+)(?P<variant>[A-Z][A-Z0-9]?)\.(ply|obj)$",
    re.IGNORECASE
)

def discover_scans(fmt: str, patient_filter=None) -> list:
    """
    FIX: Collect PLY and OBJ independently into separate lists,
    then combine. This prevents PLY entries from overwriting OBJ
    entries that share the same (patient, stem) key when fmt='both'.
    """
    results = []

    def _collect(base_dir, ext):
        found = []
        if not base_dir.exists():
            return found
        for f in sorted(base_dir.glob("*/*" + ext)):
            if "_aligned" in f.stem:
                continue
            m = SCAN_RE.match(f.name)
            if not m:
                continue
            pid = m.group("patient").lower()
            if patient_filter and pid != patient_filter.lower():
                continue
            found.append((pid, f.stem, f))
        return found

    # Collect each format independently — never deduplicate across formats
    if fmt in ("ply", "both"):
        results.extend(_collect(PLY_DIR, ".ply"))
    if fmt in ("obj", "both"):
        results.extend(_collect(OBJ_DIR, ".obj"))

    results.sort(key=lambda x: (x[0], x[1]))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  OBJ READ  (open3d cannot handle vertex-color OBJ)
# ─────────────────────────────────────────────────────────────────────────────

def _read_obj(path: Path) -> o3d.geometry.PointCloud:
    pts_list, col_list = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            p = line.split()
            pts_list.append([float(p[1]), float(p[2]), float(p[3])])
            if len(p) >= 7:
                rgb = [float(p[4]), float(p[5]), float(p[6])]
                col_list.append([v / 255.0 if v > 1.0 else v for v in rgb])
            else:
                col_list.append([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts_list))
    pcd.colors = o3d.utility.Vector3dVector(np.array(col_list))
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
#  ROTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_transform(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    def Rx(d):
        r = np.radians(d)
        return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    def Ry(d):
        r = np.radians(d)
        return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
    def Rz(d):
        r = np.radians(d)
        return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    T = np.eye(4)
    T[:3,:3] = R
    return T


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def render_snapshot(pcd, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    order       = np.argsort(pts[:, 2])
    pts         = pts[order]
    cols        = cols[order]
    cols_bright = np.clip(cols * 1.3, 0, 1)
    cols_uint8  = (cols_bright * 255).astype(np.uint8)

    x = pts[:, 0];  y = pts[:, 1]
    pad = 0.02
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ix = ((x - x_min) / (x_max - x_min) * (1-2*pad) + pad) * (IMG_W - 1)
    iy = ((y - y_min) / (y_max - y_min) * (1-2*pad) + pad) * (IMG_H - 1)
    ix = ix.astype(int).clip(0, IMG_W - 1)
    iy = iy.astype(int).clip(0, IMG_H - 1)

    import cv2 as _cv2
    img  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    N    = 3;  half = N // 2
    for i in range(len(pts)):
        y0 = max(0, iy[i]-half);  y1 = min(IMG_H, iy[i]+half+1)
        x0 = max(0, ix[i]-half);  x1 = min(IMG_W, ix[i]+half+1)
        img[y0:y1, x0:x1] = cols_uint8[i]

    img     = np.flipud(img)
    img_bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
    _cv2.imwrite(str(out_path), img_bgr)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE OBJ
# ─────────────────────────────────────────────────────────────────────────────

def save_obj(pcd, out_path: Path):
    pts    = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# " + out_path.stem + "\n# " + str(len(pts)) + " points\n\n")
        rgb = colors / 255.0
        for i in range(len(pts)):
            f.write("v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (
                pts[i,0], pts[i,1], pts[i,2],
                rgb[i,0], rgb[i,1], rgb[i,2]))


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def process_scan(patient_id, scan_name, input_path,
                 fine_yaw_override=None, overwrite=False, show=False):

    is_obj = input_path.suffix.lower() == ".obj"
    ext    = input_path.suffix

    # FIX: always mirror input_path.parent for output, same as PLY does.
    # Old code used hardcoded OBJ_DIR / patient_id which broke when
    # --dataset was passed and globals were reassigned mid-run.
    out_dir = input_path.parent

    out     = out_dir / f"{scan_name}_aligned{ext}"
    png_out = out_dir / f"{scan_name}_front.png"

    if out.exists() and png_out.exists() and not overwrite:
        print(f"  ↷ {scan_name}  already done — skipping")
        return {"scan": scan_name, "status": "skipped"}

    print(f"  → {scan_name}  [{ext.upper()}]")

    try:
        pcd = _read_obj(input_path) if is_obj else o3d.io.read_point_cloud(str(input_path))
        print(f"    Points: {np.asarray(pcd.points).shape[0]:,}")

        # Fixed rotations
        pcd.transform(make_transform(rx=-90.0, ry=90.0))
        print(f"    Applied: Rx=-90°, Ry=+90°")

        # Fine correction
        fine = fine_yaw_override
        if fine is None:
            fine = FINE_YAW_CONFIG.get(scan_name, 0.0)
        if fine != 0.0:
            pcd.transform(make_transform(ry=fine))
            print(f"    Fine Ry: {fine:+.1f}°")

        # Save
        out.parent.mkdir(parents=True, exist_ok=True)
        if is_obj:
            save_obj(pcd, out)
        else:
            o3d.io.write_point_cloud(str(out), pcd)
        render_snapshot(pcd, png_out)
        print(f"    ✓ {out.name}")
        print(f"    ✓ {png_out.name}")

        if show:
            print(f"    Opening viewer — close to continue...")
            o3d.visualization.draw_geometries(
                [pcd], window_name=scan_name, width=900, height=700)

        return {"scan": scan_name, "status": "success", "fine_ry": fine}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"scan": scan_name, "status": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient",   type=str,   default=None)
    p.add_argument("--dataset",   type=str,   default=None)
    p.add_argument("--format",    type=str,   default="both",
                   choices=["ply", "obj", "both"],
                   help="Input format to process (default: both)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show",      action="store_true")
    p.add_argument("--fine-yaw",  type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global DATASET_DIR, SCANS_DIR, PLY_DIR, OBJ_DIR
    if args.dataset:
        DATASET_DIR = Path(args.dataset)
        SCANS_DIR   = DATASET_DIR / "3d_scans"
        PLY_DIR     = SCANS_DIR / "ply"
        OBJ_DIR     = SCANS_DIR / "obj"

    if not PLY_DIR.exists() and not OBJ_DIR.exists():
        print(f"No input folders found. Run convert_to_3d.py first.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  3D Face Alignment")
    print(f"  Fixed: Rx=-90°, Ry=+90°")
    print(f"  Format: {args.format.upper()}")
    print(f"{'='*55}\n")

    scans = discover_scans(args.format, patient_filter=args.patient)
    if not scans:
        print("No scan files found.")
        sys.exit(1)

    print(f"  Found {len(scans)} scan(s)\n")

    report   = []
    prev_pat = None
    for patient_id, scan_name, input_path in scans:
        if patient_id != prev_pat:
            print(f"{'─'*55}")
            print(f"  Patient: {patient_id.upper()}")
            prev_pat = patient_id
        status = process_scan(patient_id, scan_name, input_path,
                               fine_yaw_override=args.fine_yaw,
                               overwrite=args.overwrite,
                               show=args.show)
        report.append(status)

    ok   = sum(1 for r in report if r["status"] == "success")
    skip = sum(1 for r in report if r["status"] == "skipped")
    fail = [r for r in report if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  DONE  {ok} aligned, {skip} skipped")
    if fail:
        for r in fail:
            print(f"  ✗ {r['scan']} — {r['error']}")
    print(f"\n  Review _front.png snapshots.")
    print(f"  Add fine corrections to FINE_YAW_CONFIG if needed.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()


# python align_3d_front.py                                    # PLY + OBJ (default)
# python align_3d_front.py --format ply                       # PLY only
# python align_3d_front.py --format obj                       # OBJ only
# python align_3d_front.py --patient pat1                     # one patient
# python align_3d_front.py --overwrite                        # redo existing
# python align_3d_front.py --show                             # open 3D viewer
# python align_3d_front.py --fine-yaw 15.0                    # yaw correction
# python align_3d_front.py --format obj --patient pat1 --overwrite