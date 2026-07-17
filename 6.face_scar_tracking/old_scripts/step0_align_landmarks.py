#!/usr/bin/env python3
"""
step0_align_landmarks.py
==========================
Step 0 of the tracking pipeline: 3D landmark-based alignment.

KEY DESIGN: each camera variant (A, B, C, D, ...) captures a different
posture/angle and has different landmark dots visible. Variants MUST be
aligned separately against their own per-variant reference template.

  D00_A template  ->  aligns  D14_A, D28_A, M02_A ... M24_A
  D00_B template  ->  aligns  D14_B, D28_B, M02_B ... M24_B
  D00_C template  ->  aligns  D14_C ...
  D00_D template  ->  aligns  D14_D ...

MODES:

  build-template  -- run ONCE per variant, on that variant's D00 scan.
                     You note the landmark dot pixel coords once, this
                     lifts them to 3D and saves the reference template.

  align-scan      -- align one scan against its variant's template.

  align-batch     -- align every scan in a patient folder. Auto-discovers
                     all per-variant templates and routes each scan to
                     the right one. Scans whose variant has no template
                     are skipped with a clear message.

USAGE:
  # Build template for each variant (run 4 times, once per variant)
  python step0_align_landmarks.py build-template --scandir ".../PAT01/D00/PAT01_D00_A"
  python step0_align_landmarks.py build-template --scandir ".../PAT01/D00/PAT01_D00_B"
  python step0_align_landmarks.py build-template --scandir ".../PAT01/D00/PAT01_D00_C"
  python step0_align_landmarks.py build-template --scandir ".../PAT01/D00/PAT01_D00_D"
  # (each D00_<V> scan needs its own <scan>_landmark_targets.json first)

  # Align all scans in one batch (auto-routes by variant)
  python step0_align_landmarks.py align-batch --dataset "D:/.../" --patient PAT01

  # Align a single scan (auto-finds the right template by variant)
  python step0_align_landmarks.py align-scan --scandir ".../PAT01/M06/PAT01_M06_A"

OUTPUT:
  Templates   -> <patient_root>/PAT01_D00_<V>_landmark_template.json
  Alignments  -> <scan_dir>/<scan>_alignment.json  (one per scan)
    {
      "scan": "...", "variant": "A",
      "template_used": "PAT01_D00_A_landmark_template.json",
      "status": "ok" | "low_confidence" | "failed",
      "n_inliers": 3..7, "mean_residual_mm": ..., "rotation_deg": ...,
      "matched_landmarks": [...], "R": [[...]], "t": [...]
    }
"""

import sys
import re
import json
import argparse
import numpy as np
import tifffile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "3.burn_segmentation_3d_pipeline"))
from utils.cyberware import range_to_3d
from utils.landmarks import (
    detect_candidates_2d, lift_candidates_to_3d, match_to_template, build_template
)

SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _variant_of(scan_dir: Path) -> str | None:
    m = SCAN_RE.match(scan_dir.name)
    return m.group("variant").upper() if m else None


def _discover_templates(patient_dir: Path) -> dict:
    """Returns {variant: template_path} — checks landmark_templates/ first, then patient root."""
    templates = {}
    search_dirs = [patient_dir / "landmark_templates", patient_dir]
    for d in search_dirs:
        for t in sorted(d.glob("*_landmark_template.json")):
            stem = t.stem.replace("_landmark_template", "")
            m = SCAN_RE.match(stem)
            if m:
                v = m.group("variant").upper()
                if v not in templates:   # prefer landmark_templates/ over root
                    templates[v] = t
    return templates


def _resolve_template_for_variant(template_arg, patient_dir: Path, variant: str) -> Path:
    """Find the template for a specific variant."""
    if template_arg:
        return Path(template_arg)
    templates = _discover_templates(patient_dir)
    if variant in templates:
        return templates[variant]
    raise FileNotFoundError(
        f"No template found for variant '{variant}' in {patient_dir}.\n"
        f"  Run: python step0_align_landmarks.py build-template "
        f"--scandir <path_to_D00_{variant}_scan>"
    )


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


def load_scan_3d(scan_dir: Path):
    name = scan_dir.name
    tif_path   = scan_dir / f"{name}.tif"
    range_path = scan_dir / name
    if not tif_path.exists():
        raise FileNotFoundError(f"TIF not found: {tif_path}")
    if not range_path.exists():
        raise FileNotFoundError(f"Range file not found: {range_path}")
    img_rgb = load_tiff_rgb(tif_path)
    pts, colors, tif_row, tif_col, valid_mask = range_to_3d(range_path, img_rgb)
    return name, img_rgb, pts, tif_row, tif_col


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 1 -- build-template  (once per variant)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build_template(args):
    scan_dir    = Path(args.scandir)
    patient_dir = scan_dir.parent.parent
    variant     = _variant_of(scan_dir) or "?"

    name, img_rgb, pts, tif_row, tif_col = load_scan_3d(scan_dir)

    # targets lookup priority:
    #   1. explicit --targets argument
    #   2. patient_root/landmark_targets/PAT01_<V>_landmark_targets.json
    #   3. patient_root/PAT01_<V>_landmark_targets.json  (legacy)
    #   4. scan folder/<scan>_landmark_targets.json       (legacy)
    patient_id   = patient_dir.name
    tgt_dir      = patient_dir / "landmark_targets"
    targets_candidates = [
        Path(args.targets) if args.targets else None,
        tgt_dir      / f"{patient_id}_{variant}_landmark_targets.json",
        patient_dir  / f"{patient_id}_{variant}_landmark_targets.json",
        scan_dir     / f"{name}_landmark_targets.json",
    ]
    targets_path = next((p for p in targets_candidates if p and p.exists()), None)
    if targets_path is None:
        preferred = tgt_dir / f"{patient_id}_{variant}_landmark_targets.json"
        raise FileNotFoundError(
            f"Targets file not found. Create:\n"
            f"  {preferred}\n\n"
            f"  Contents: pixel [x, y] coords for each visible landmark dot, e.g.:\n"
            f'  {{"left_forehead":[x,y], "right_forehead":[x,y], '
            f'"between_eyebrows":[x,y],\n'
            f'   "nose_tip":[x,y], "left_ear":[x,y], '
            f'"right_ear":[x,y], "chin":[x,y]}}\n'
            f"  Only include landmarks actually visible in variant {variant}'s framing."
        )

    targets  = {k: tuple(v) for k, v in json.load(open(targets_path)).items()}
    template = build_template(img_rgb, pts, tif_row, tif_col, targets)
    missing  = set(targets) - set(template)
    if missing:
        print(f"  ⚠ Could not lift to 3D: {sorted(missing)}")

    # save template to landmark_templates/ subfolder
    if args.out:
        out_path = Path(args.out)
    else:
        tpl_dir  = patient_dir / "landmark_templates"
        tpl_dir.mkdir(parents=True, exist_ok=True)
        out_path = tpl_dir / f"{name}_landmark_template.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(template, open(out_path, "w"), indent=2)

    print(f"  ✓ Targets:  {targets_path}")
    print(f"  ✓ Template: {out_path}  (variant={variant}, {len(template)} landmarks)")
    for k, v in template.items():
        print(f"      {k:20s} XYZ={[round(x,1) for x in v]}")


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 2 -- align-scan  (single scan)
# ─────────────────────────────────────────────────────────────────────────────

def align_one_scan(scan_dir: Path, template: dict, template_name: str,
                   tol_mm: float, min_inliers: int,
                   max_rotation_deg: float, max_tilt_deg: float) -> dict:
    name, img_rgb, pts, tif_row, tif_col = load_scan_3d(scan_dir)
    variant = _variant_of(scan_dir) or "?"

    candidates    = detect_candidates_2d(img_rgb)
    candidates_3d = lift_candidates_to_3d(candidates, pts, tif_row, tif_col)
    result        = match_to_template(candidates_3d, template, tol_mm=tol_mm,
                                      min_inliers=min_inliers,
                                      max_rotation_deg=max_rotation_deg,
                                      max_tilt_deg=max_tilt_deg)
    if result is None:
        return {
            "scan": name, "variant": variant,
            "template_used": template_name,
            "status": "failed", "n_inliers": 0,
            "mean_residual_mm": None, "rotation_deg": None,
            "yaw_deg": None, "pitch_deg": None, "roll_deg": None,
            "matched_landmarks": [], "R": None, "t": None,
            "note": f"Fewer than {min_inliers} landmarks matched.",
        }
    status = "low_confidence" if result["n_inliers"] == 3 else "ok"
    return {
        "scan": name, "variant": variant,
        "template_used": template_name,
        "status": status,
        "n_inliers": result["n_inliers"],
        "mean_residual_mm": round(result["mean_residual_mm"], 3),
        "rotation_deg": round(result["rotation_deg"], 2),
        "yaw_deg":      round(result["yaw_deg"], 2),
        "pitch_deg":    round(result["pitch_deg"], 2),
        "roll_deg":     round(result["roll_deg"], 2),
        "matched_landmarks": sorted(result["matches"].keys()),
        "R": result["R"].tolist(), "t": result["t"].tolist(),
    }


def cmd_align_scan(args):
    scan_dir    = Path(args.scandir)
    variant     = _variant_of(scan_dir)
    patient_dir = scan_dir.parent.parent
    tpl_path    = _resolve_template_for_variant(args.template, patient_dir, variant)
    template    = json.load(open(tpl_path))

    out = align_one_scan(scan_dir, template, tpl_path.name,
                         args.tol_mm, args.min_inliers,
                         args.max_rotation_deg, args.max_tilt_deg)
    out_path = scan_dir / f"{scan_dir.name}_alignment.json"
    json.dump(out, open(out_path, "w"), indent=2)
    sym = {"ok": "✓", "low_confidence": "⚠", "failed": "✗"}[out["status"]]
    print(f"  {sym} {out['scan']} [{variant}]: {out['status']}  "
          f"(n_inliers={out['n_inliers']}, residual={out['mean_residual_mm']}mm, "
          f"yaw={out['yaw_deg']}° pitch={out['pitch_deg']}° roll={out['roll_deg']}°, "
          f"template={tpl_path.name})")


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 3 -- align-batch  (auto-routes each scan to its variant's template)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_align_batch(args):
    dataset_dir = Path(args.dataset)
    pat_dir     = dataset_dir / args.patient

    templates = _discover_templates(pat_dir)
    if not templates:
        print(f"  No landmark templates found in {pat_dir}")
        print(f"  Run build-template for each variant's D00 scan first:")
        for v in ["A", "B", "C", "D"]:
            print(f"    python step0_align_landmarks.py build-template "
                  f"--scandir <path>/D00/PAT_D00_{v}")
        return

    print(f"  Templates found for variants: {sorted(templates.keys())}")
    for v, t in sorted(templates.items()):
        print(f"    {v}: {t.name}")
    print()

    scan_dirs = sorted([p for p in pat_dir.glob("*/*") if p.is_dir()])
    print(f"  Found {len(scan_dirs)} scan folder(s)\n")

    ok, low_conf, failed, no_tpl = 0, 0, [], []

    for scan_dir in scan_dirs:
        variant = _variant_of(scan_dir)
        if not variant or variant not in templates:
            print(f"  – {scan_dir.name}: no template for variant '{variant}', skipping")
            no_tpl.append(scan_dir.name)
            continue

        tpl_path = templates[variant]
        template = json.load(open(tpl_path))
        try:
            out = align_one_scan(scan_dir, template, tpl_path.name,
                                 args.tol_mm, args.min_inliers,
                                 args.max_rotation_deg, args.max_tilt_deg)
            out_path = scan_dir / f"{scan_dir.name}_alignment.json"
            json.dump(out, open(out_path, "w"), indent=2)
            sym = {"ok": "✓", "low_confidence": "⚠", "failed": "✗"}[out["status"]]
            print(f"  {sym} {out['scan']} [{variant}]: {out['status']}  "
                  f"(n_inliers={out['n_inliers']}, residual={out['mean_residual_mm']}mm, "
                  f"yaw={out['yaw_deg']}° pitch={out['pitch_deg']}° roll={out['roll_deg']}°)")
            if out["status"] == "ok":          ok += 1
            elif out["status"] == "low_confidence": low_conf += 1
            else:                              failed.append(out["scan"])
        except FileNotFoundError as e:
            print(f"  ✗ {scan_dir.name}: skipped ({e})")
            failed.append(scan_dir.name)

    print(f"\n  DONE — {ok} ok, {low_conf} low_confidence, "
          f"{len(failed)} failed, {len(no_tpl)} skipped (no template)")
    if no_tpl:
        missing_variants = sorted(set(
            v for n in no_tpl if (v := _variant_of(Path(n))) is not None
        ))
        print(f"  Missing templates for variants: {missing_variants}")
        print(f"  Run build-template for their D00 scans to include them.")
    if failed:
        print("  Failed scans:")
        for f in failed:
            print(f"    ✗ {f}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p   = argparse.ArgumentParser(description="Step 0 — per-variant 3D landmark alignment")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bt = sub.add_parser("build-template",
                           help="Build reference template for one variant's D00 scan")
    p_bt.add_argument("--scandir",  required=True,
                      help="Path to D00_<V> scan folder, e.g. .../PAT01/D00/PAT01_D00_B")
    p_bt.add_argument("--targets",  default=None,
                      help="Targets JSON (default: <scandir>/<scan>_landmark_targets.json)")
    p_bt.add_argument("--out",      default=None,
                      help="Output path (default: <patient_root>/<scan>_landmark_template.json)")
    p_bt.set_defaults(func=cmd_build_template)

    p_as = sub.add_parser("align-scan", help="Align one scan (auto-finds its variant's template)")
    p_as.add_argument("--scandir",         required=True)
    p_as.add_argument("--template",        default=None,
                      help="Override template path (default: auto-found by variant)")
    p_as.add_argument("--tol-mm",           type=float, default=8.0)
    p_as.add_argument("--min-inliers",      type=int,   default=3)
    p_as.add_argument("--max-rotation-deg", type=float, default=90.0)
    p_as.add_argument("--max-tilt-deg",     type=float, default=20.0,
                      help="Max pitch/roll in degrees (Z-axis constraint). Default 20°.")
    p_as.set_defaults(func=cmd_align_scan)

    p_ab = sub.add_parser("align-batch",
                           help="Align every scan in a patient folder (routes by variant)")
    p_ab.add_argument("--dataset",         required=True)
    p_ab.add_argument("--patient",         required=True)
    p_ab.add_argument("--tol-mm",           type=float, default=8.0)
    p_ab.add_argument("--min-inliers",      type=int,   default=3)
    p_ab.add_argument("--max-rotation-deg", type=float, default=90.0)
    p_ab.add_argument("--max-tilt-deg",     type=float, default=20.0,
                      help="Max pitch/roll in degrees (Z-axis constraint). Default 20°.")
    p_ab.set_defaults(func=cmd_align_batch)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()