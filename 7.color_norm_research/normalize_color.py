#!/usr/bin/env python3
"""
normalize_color.py
==================
Batch colour normalization using the user-selected reference region.
Requires select_skin_reference.py to have been run first.

METHOD: Reinhard LAB transfer
  Reference colour: from user-selected normal skin polygon in D00
  Applied to: auto-detected skin pixels only (not hair, background, eyes)
  Per variant: D00_A reference → all A scans, D00_B → all B, etc.

OUTPUT:
  <scan_dir>/<scan>_normalized.tif     drop-in replacement for input TIF
  <patient_dir>/color_normalization/<patient>_<variant>_norm_report.txt

USAGE:
  python normalize_color.py --dataset "D:/.../" --patient PAT01
  python normalize_color.py --dataset "D:/.../" --patient PAT01 --variant A
"""

import re
import json
import argparse
import numpy as np
import cv2
import tifffile
from pathlib import Path

SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


def load_tif(path: Path) -> np.ndarray:
    raw = tifffile.imread(str(path))
    if raw.ndim == 3 and raw.shape[0] < 10:
        raw = raw[0]
    raw = raw.astype(np.float32)
    lo, hi = raw.min(), raw.max()
    if hi > lo:
        raw = (raw - lo) / (hi - lo) * 255.0
    return raw.astype(np.uint8)[..., :3].copy()


def load_burn_mask(scan_dir: Path) -> np.ndarray | None:
    p = scan_dir / f"{scan_dir.name}_burn_polygons.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    h, w = data["image_size"]["height"], data["image_size"]["width"]
    mask = np.zeros((h, w), bool)
    for r in data.get("regions", []):
        cnt = np.array(r["polygon"], np.int32).reshape(-1, 1, 2)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [cnt], 1)
        mask |= m.astype(bool)
    return mask


def face_skin_mask(img: np.ndarray) -> np.ndarray:
    """Detect face skin pixels — excludes hair, background, specular highlights."""
    ycr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycr[..., 0], ycr[..., 1], ycr[..., 2]
    return (
        (Y > 30) & (Y < 230) &
        (Cr > 133) & (Cr < 177) &
        (Cb > 77)  & (Cb < 127)
    )


def normalize_scan(src_rgb: np.ndarray,
                   ref_lab_stats: dict,
                   src_burn: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """
    Apply Reinhard LAB correction to src_rgb.

    ref_lab_stats: {L_star:{mean,std}, a_star:{mean,std}, b_star:{mean,std}}
      — from the user-selected normal skin polygon in D00

    Statistics for the SOURCE are computed from:
      - auto-detected skin pixels
      - MINUS burn region (so the burn doesn't bias the correction)

    Correction applied to: all detected skin pixels only.
    Non-skin pixels: pixel-perfect unchanged.
    """
    src_skin = face_skin_mask(src_rgb)

    # exclude burn from statistics (but correction IS applied to burn pixels)
    if src_burn is not None and src_burn.sum() > 0:
        src_stat = src_skin & ~src_burn
    else:
        src_stat = src_skin

    if src_stat.sum() < 50:
        src_stat = src_skin   # fallback

    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(float)
    corrected = src_lab.copy()

    ref_channels = [ref_lab_stats["L_star"],
                    ref_lab_stats["a_star"],
                    ref_lab_stats["b_star"]]

    for c, ref in enumerate(ref_channels):
        vals = src_lab[..., c][src_stat]
        if len(vals) < 10:
            continue
        sm, ss = vals.mean(), vals.std() + 1e-6
        rm, rs = ref["mean"], ref["std"] + 1e-6
        # apply Reinhard shift+scale to skin pixels only
        corrected[..., c] = np.where(
            src_skin,
            (src_lab[..., c] - sm) * (rs / ss) + rm,
            src_lab[..., c]    # non-skin unchanged
        )

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)

    # stats
    ref_a = ref_lab_stats["a_star"]["mean"]
    src_skin_a_before = src_lab[..., 1][src_skin].mean() if src_skin.sum() > 0 else float('nan')
    src_skin_a_after  = corrected.astype(float)[..., 1][src_skin].mean() if src_skin.sum() > 0 else float('nan')

    burn_a_before = src_lab[..., 1][src_burn].mean() if src_burn is not None and src_burn.sum() > 0 else float('nan')
    burn_a_after  = corrected.astype(float)[..., 1][src_burn].mean() if src_burn is not None and src_burn.sum() > 0 else float('nan')

    stats = {
        "skin_px":           int(src_skin.sum()),
        "stat_px":           int(src_stat.sum()),
        "ref_skin_a":        round(ref_a, 2),
        "skin_a_before":     round(float(src_skin_a_before), 2),
        "skin_a_after":      round(float(src_skin_a_after), 2),
        "skin_delta_before": round(abs(float(src_skin_a_before) - ref_a), 2),
        "skin_delta_after":  round(abs(float(src_skin_a_after)  - ref_a), 2),
        "burn_a_before":     round(float(burn_a_before), 2),
        "burn_a_after":      round(float(burn_a_after), 2),
        "has_burn":          src_burn is not None and src_burn.sum() > 0,
    }
    return corrected_rgb, stats


def tp_days(tp):
    m = re.match(r'([DM])(\d+)', tp.upper())
    return int(m.group(2)) if m and m.group(1) == 'D' else int(m.group(2)) * 30 if m else 9999


def process_variant(pat_dir: Path, patient: str, variant: str,
                    all_scan_dirs: list, ref_lab_stats: dict) -> list:
    v_scans = sorted(
        [d for d in all_scan_dirs
         if SCAN_RE.match(d.name)
         and SCAN_RE.match(d.name).group("variant").upper() == variant],
        key=lambda d: tp_days(SCAN_RE.match(d.name).group("timepoint"))
    )
    # skip D00 (it IS the reference)
    to_process = [d for d in v_scans
                  if SCAN_RE.match(d.name).group("timepoint").upper() != "D00"]

    if not to_process:
        print(f"  [{variant}] No scans to normalise (only D00 found)")
        return []

    print(f"\n  Variant {variant} — {len(to_process)} scan(s)")
    all_stats = []

    for src_dir in to_process:
        name = src_dir.name
        tif_path = src_dir / f"{name}.tif"
        if not tif_path.exists():
            print(f"    ✗ {name}: TIF not found"); continue

        src_rgb  = load_tif(tif_path)
        src_burn = load_burn_mask(src_dir)
        corrected, stats = normalize_scan(src_rgb, ref_lab_stats, src_burn)

        out_path = src_dir / f"{name}_normalized.tif"
        tifffile.imwrite(str(out_path), corrected)

        burn_str = (f"burn_a*: {stats['burn_a_before']:.1f}→{stats['burn_a_after']:.1f}"
                    if stats["has_burn"] else "no burn polygon")
        print(f"    ✓ {name}  "
              f"skin_Δa*: {stats['skin_delta_before']:.1f}→{stats['skin_delta_after']:.1f}  "
              f"{burn_str}")

        stats["scan"] = name
        all_stats.append(stats)

    return all_stats


def write_report(pat_dir: Path, patient: str, variant: str,
                 ref_stats: dict, all_stats: list):
    report_dir = pat_dir / "color_normalization"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / f"{patient}_{variant}_norm_report.txt"

    ref_a = ref_stats["a_star"]["mean"]
    lines = [
        f"Colour Normalization Report",
        f"Patient   : {patient}",
        f"Variant   : {variant}",
        f"Method    : Reinhard LAB (user-selected normal skin reference)",
        f"Reference : D00_{variant} — user-selected skin region",
        f"  L*={ref_stats['L_star']['mean']:.1f}±{ref_stats['L_star']['std']:.1f}"
        f"  a*={ref_stats['a_star']['mean']:.1f}±{ref_stats['a_star']['std']:.1f}"
        f"  b*={ref_stats['b_star']['mean']:.1f}±{ref_stats['b_star']['std']:.1f}",
        "=" * 70, "",
        f"{'Scan':<22} {'Δa* before':>10} {'Δa* after':>10} "
        f"{'burn_a* D00':>12} {'burn_a* raw':>12} {'burn_a* norm':>12}",
        "-" * 80,
    ]

    for s in all_stats:
        b_ref   = f"{s['burn_a_before']:.1f}"  if s["has_burn"] else "—"
        b_after = f"{s['burn_a_after']:.1f}"   if s["has_burn"] else "—"
        lines.append(
            f"{s['scan']:<22} "
            f"{s['skin_delta_before']:>10.1f} "
            f"{s['skin_delta_after']:>10.1f} "
            f"{ref_a:>12.1f} "
            f"{b_ref:>12} "
            f"{b_after:>12}"
        )

    # burn trend
    burn_scans = [s for s in all_stats if s["has_burn"]]
    if burn_scans:
        lines += ["", "",
                  "BURN a* TREND  (reference D00 a*={:.1f})".format(ref_a),
                  "  Lower = less red = more healed", ""]
        for s in burn_scans:
            delta = s["burn_a_after"] - ref_a
            sign  = "+" if delta >= 0 else ""
            bar   = "#" * max(0, int(abs(delta) / 0.5))
            lines.append(
                f"  {s['scan']:<22}  "
                f"normalised burn a*={s['burn_a_after']:.1f}  "
                f"({sign}{delta:.1f} vs D00)  {bar}"
            )

    lines += [
        "", "",
        "OUTPUT FILES:",
        "  Each scan folder contains: <scan>_normalized.tif",
        "  Use this file as input to the segmentation model (step2_segment_3d.py)",
        "  instead of the original TIF — everything else in the pipeline unchanged.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True)
    p.add_argument("--patient",  required=True)
    p.add_argument("--variant",  default=None)
    args = p.parse_args()

    pat_dir = Path(args.dataset) / args.patient
    ref_path = pat_dir / "color_normalization" / f"{args.patient}_skin_reference.json"

    if not ref_path.exists():
        print(f"  Reference region not found: {ref_path}")
        print(f"  Run select_skin_reference.py --dataset {args.dataset} --patient {args.patient} first.")
        return

    ref_data     = json.loads(ref_path.read_text())
    ref_lab_stats = ref_data["lab_stats"]
    print(f"  Reference region loaded: {ref_data['n_pixels']} pixels")
    print(f"  a*={ref_lab_stats['a_star']['mean']:.1f}  (skin redness reference)")

    all_scan_dirs = sorted([p for p in pat_dir.glob("*/*")
                             if p.is_dir() and SCAN_RE.match(p.name)])
    variants = sorted(set(
        SCAN_RE.match(d.name).group("variant").upper()
        for d in all_scan_dirs if SCAN_RE.match(d.name)
    ))
    if args.variant:
        variants = [v for v in variants if v == args.variant.upper()]

    print(f"  Patient: {args.patient}   Variants: {variants}\n")

    for variant in variants:
        stats = process_variant(pat_dir, args.patient, variant,
                                all_scan_dirs, ref_lab_stats)
        if stats:
            write_report(pat_dir, args.patient, variant, ref_lab_stats, stats)


if __name__ == "__main__":
    main()