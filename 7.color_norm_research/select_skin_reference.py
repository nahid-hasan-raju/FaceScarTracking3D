#!/usr/bin/env python3
"""
select_skin_reference.py
=========================
Interactive tool — run ONCE per patient.

Opens the D00 TIF in a window. User clicks to draw a polygon
over a clean, normal skin area (no burn, no hair, no shadow).
That region's LAB colour statistics become the reference for
normalizing all later scans of this patient.

Controls:
  Left click  : add polygon vertex
  Enter       : confirm selection
  R           : reset / start over
  Z           : undo last point
  Q           : quit without saving

OUTPUT:
  <patient_dir>/color_normalization/
    <patient>_skin_reference.json   reference region + LAB stats
    <patient>_skin_reference.png    verification image (selected region highlighted)

USAGE:
  python select_skin_reference.py --dataset "D:/.../" --patient PAT01
  python select_skin_reference.py --tif "D:/.../PAT01/D00/PAT01_D00_A/PAT01_D00_A.tif"
"""

import json
import argparse
import numpy as np
import cv2
import tifffile
from pathlib import Path


def load_tif(path: Path) -> np.ndarray:
    raw = tifffile.imread(str(path))
    if raw.ndim == 3 and raw.shape[0] < 10:
        raw = raw[0]
    raw = raw.astype(np.float32)
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        raw = (raw - lo) / (hi - lo) * 255.0
    return raw.astype(np.uint8)[..., :3].copy()


def find_d00_tif(pat_dir: Path) -> Path | None:
    for p in pat_dir.glob("*/PAT*_D00_A/PAT*_D00_A.tif"):
        return p
    for p in pat_dir.glob("D00/*/PAT*_D00_A.tif"):
        return p
    for p in pat_dir.glob("*/*D00*.tif"):
        return p
    return None


def run_selector(img_rgb: np.ndarray, window_title: str = "Select normal skin region"):
    """
    Interactive polygon selector.
    Returns list of (x, y) points or None if cancelled.
    """
    # scale display if image is large
    H, W = img_rgb.shape[:2]
    max_dim = 800
    scale = min(max_dim / W, max_dim / H, 1.0)
    dW, dH = int(W * scale), int(H * scale)
    display_base = cv2.resize(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (dW, dH)
    )

    points = []         # in display coords
    confirmed = False

    def draw(pts):
        disp = display_base.copy()
        overlay = disp.copy()

        # draw filled polygon if 3+ points
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 200, 80))
            cv2.addWeighted(overlay, 0.25, disp, 0.75, 0, disp)

        # draw edges
        if len(pts) >= 2:
            cv2.polylines(disp, [np.array(pts, np.int32)], False, (0, 230, 80), 2)
        if len(pts) >= 3:
            cv2.line(disp, pts[-1], pts[0], (0, 230, 80), 1)  # closing line

        # draw vertices
        for pt in pts:
            cv2.circle(disp, pt, 5, (0, 230, 80), -1)
            cv2.circle(disp, pt, 5, (255, 255, 255), 1)

        # instructions
        instructions = [
            "LEFT CLICK: add point",
            "Z: undo last point   R: reset",
            "ENTER: confirm   Q: quit",
        ]
        y0 = 20
        for line in instructions:
            cv2.putText(disp, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(disp, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
            y0 += 20

        if len(pts) >= 3:
            cv2.putText(disp, f"  {len(pts)} points — press ENTER to confirm",
                        (10, dH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 80), 1)
        return disp

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.imshow(window_title, draw(points))

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, dW, dH)
    cv2.setMouseCallback(window_title, mouse_cb)
    cv2.imshow(window_title, draw(points))

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 10:   # Enter
            if len(points) >= 3:
                confirmed = True
                break
            print("  Need at least 3 points — keep clicking")
        elif key == ord('r') or key == ord('R'):
            points.clear()
            cv2.imshow(window_title, draw(points))
        elif key == ord('z') or key == ord('Z'):
            if points:
                points.pop()
                cv2.imshow(window_title, draw(points))
        elif key == ord('q') or key == ord('Q') or key == 27:
            break

    cv2.destroyAllWindows()
    if not confirmed or len(points) < 3:
        return None

    # convert back to original image coords
    orig_pts = [(int(x / scale), int(y / scale)) for x, y in points]
    return orig_pts


def extract_region_stats(img_rgb: np.ndarray, polygon_pts: list) -> dict:
    """Extract LAB stats from within the selected polygon."""
    H, W = img_rgb.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.array(polygon_pts, np.int32)], 1)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(float)
    region_pixels = mask.astype(bool)
    n = int(region_pixels.sum())
    if n == 0:
        raise ValueError("Selected region contains no pixels")

    stats = {"n_pixels": n, "polygon": polygon_pts}
    for ci, name in enumerate(["L_star", "a_star", "b_star"]):
        vals = lab[..., ci][region_pixels]
        stats[name] = {
            "mean": round(float(vals.mean()), 3),
            "std":  round(float(vals.std()),  3),
        }

    print(f"  Reference region: {n} pixels")
    print(f"    L*={stats['L_star']['mean']:.1f}±{stats['L_star']['std']:.1f}"
          f"  a*={stats['a_star']['mean']:.1f}±{stats['a_star']['std']:.1f}"
          f"  b*={stats['b_star']['mean']:.1f}±{stats['b_star']['std']:.1f}")
    return stats


def save_verification(img_rgb: np.ndarray, polygon_pts: list, out_path: Path):
    """Save a PNG highlighting the selected reference region."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    overlay = bgr.copy()
    pts = np.array(polygon_pts, np.int32)
    cv2.fillPoly(overlay, [pts], (0, 200, 80))
    cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0, bgr)
    cv2.polylines(bgr, [pts], True, (0, 230, 80), 2)
    for pt in polygon_pts:
        cv2.circle(bgr, pt, 5, (0, 230, 80), -1)
    cv2.putText(bgr, "Reference normal skin region",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 80), 2)
    cv2.imwrite(str(out_path), bgr)


def main():
    p = argparse.ArgumentParser(
        description="Select reference normal skin region from D00 scan")
    p.add_argument("--dataset", default=None)
    p.add_argument("--patient", default=None)
    p.add_argument("--tif",     default=None,
                   help="Direct path to D00 TIF (alternative to --dataset + --patient)")
    args = p.parse_args()

    if args.tif:
        tif_path = Path(args.tif)
        pat_dir  = tif_path.parent.parent.parent   # .../PAT01/D00/PAT01_D00_A/
        patient  = tif_path.parent.parent.parent.name
    elif args.dataset and args.patient:
        pat_dir  = Path(args.dataset) / args.patient
        patient  = args.patient
        tif_path = find_d00_tif(pat_dir)
        if tif_path is None:
            raise FileNotFoundError(f"No D00 TIF found under {pat_dir}")
    else:
        p.error("Provide --dataset + --patient, or --tif")

    print(f"Patient : {patient}")
    print(f"D00 TIF : {tif_path}")
    print()
    print("  Draw a polygon over a clean normal skin area.")
    print("  Choose a region with NO burn, NO shadow, NO hair.")
    print()

    img = load_tif(tif_path)
    polygon_pts = run_selector(img, f"{patient} — Select normal skin reference region")

    if polygon_pts is None:
        print("  Cancelled — nothing saved.")
        return

    stats = extract_region_stats(img, polygon_pts)

    out_dir = pat_dir / "color_normalization"
    out_dir.mkdir(exist_ok=True)
    ref_path = out_dir / f"{patient}_skin_reference.json"
    png_path = out_dir / f"{patient}_skin_reference.png"

    json_data = {
        "patient":    patient,
        "d00_tif":    str(tif_path),
        "polygon":    polygon_pts,
        "lab_stats":  {
            "L_star": stats["L_star"],
            "a_star": stats["a_star"],
            "b_star": stats["b_star"],
        },
        "n_pixels":   stats["n_pixels"],
    }
    ref_path.write_text(json.dumps(json_data, indent=2))
    save_verification(img, polygon_pts, png_path)

    print(f"\n  ✓ Reference saved : {ref_path}")
    print(f"  ✓ Verification PNG: {png_path}")
    print(f"\n  Now run normalize_color.py --dataset ... --patient {patient}")


if __name__ == "__main__":
    main()
