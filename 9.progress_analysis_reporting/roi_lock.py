#!/usr/bin/env python3
r"""
roi_lock.py
=============
Second, complementary way of measuring burn progress, alongside the existing
independent-per-scan-detection approach in analysis_common.py / step1-3.

WHY THIS EXISTS
---------------
The existing approach re-runs burn detection independently on every scan.
That's a good baseline view, but it inherits a specific failure mode: the
detector's segmentation quality varies day to day (lighting, healing-stage
contrast, etc.), so day-to-day area can go UP even when the wound is
visibly healing, just because that day's detection was noisier/more
fragmented than the previous one.

This module fixes the REGION being measured (from the Day-0 / earliest
scan) and only asks "how much of what's still detected on later scans
falls inside that same physical region" -- so a noisy detector can still
under- or over-segment on a given day, but it can no longer make the
tracked area jump around simply by detecting stray unrelated regions
elsewhere on the face.

HOW IT WORKS
------------
1.  Baseline reference: take the earliest scan for a given
    (patient, variant), and for each of its detected burn regions, build a
    padded bounding SPHERE in the already-registered 3D coordinate frame
    (`aligned_centroid_xyz`) -- center = region centroid, radius = the
    region's own real extent (max distance from centroid to any of its
    own polygon_3d points) + a fixed pad_mm margin. The union of these
    spheres is the "Day-0 ROI" for that patient/variant.

2.  Local alignment nudge: registration across scans is already close
    (global landmark-based mesh alignment from folder 1/6), but can be
    slightly off day to day. For each later scan, this module looks at
    that day's OWN independently-detected regions that fall near the
    baseline ROI, and uses their area-weighted centroid offset from the
    baseline centers as a small rigid correction (capped at max_shift_mm).
    This uses the day's detection only to help re-center the fixed ROI --
    it never changes the ROI's size/shape, and never contributes area on
    its own.

3.  Measurement: after nudging, every region in that day's detection is
    tested against the (shifted) baseline spheres. Regions that fall
    inside are summed into `roi_area_mm2` -- this is the tracked number.
    Regions that fall clearly outside (and are non-trivial in size) are
    logged as "external candidates", never added to the tracked area, and
    only escalated to a "confirmed_recurring" flag if a similarly-located
    external region shows up in more than one scan in a row (so a single
    stray false-positive detection can't trigger a false alarm).

WHAT THIS DOES NOT DO YET
--------------------------
Two more axes worth adding later (see the "Vancouver Scar Scale" style
discussion): a color/erythema score and a 3D relief/height score, computed
inside this same locked ROI. Both are left as `None` placeholders in the
output for now:
  - color_metrics needs pixel-level access to the (ideally color-normalized)
    TIF for that scan, cropped to the ROI polygon -- not just the polygon
    json this module reads.
  - relief_metrics needs the FULL facial mesh (not just the burn region
    points) to fit a local "normal skin" reference plane/surface to compare
    the wound's height against.
Both are straightforward to bolt on once those inputs are available --
the per-scan json schema below already has slots for them.

OUTPUT LOCATION (per patient) -- kept separate from the existing
independent-detection outputs so neither analysis overwrites the other:

  <patient>/analysis/roi_locked/
      <patient>_<variant>_roi_reference.json   -- the Day-0 ROI definition
      <scan>_roi_severity.json                 -- one per scan, per variant
      plots/<patient>_<variant>_roi_vs_independent.png
"""

import json
import math
import re
from pathlib import Path

from analysis_common import (
    SCAN_RE,
    elapsed_days,
    discover_scan_dirs,
    load_patient_area_series,
)

DEFAULT_PAD_MM = 8.0
DEFAULT_MAX_SHIFT_MM = 6.0
DEFAULT_MIN_FLAG_AREA_MM2 = 50.0
DEFAULT_CLUSTER_DIST_MM = 6.0


# ---------------------------------------------------------------- helpers --

def _dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _valid_point(p):
    """True if p looks like a usable [x, y, z] with real numbers -- guards
    against the null/partial entries seen in real polygon_3d data, where a
    boundary pixel's 3D back-projection failed but the point is still
    present in the list as None (or an incomplete/non-numeric entry)."""
    if not isinstance(p, (list, tuple)) or len(p) != 3:
        return False
    return all(isinstance(v, (int, float)) for v in p)


def _load_regions(scan_dir: Path, scan_name: str, need_polygon: bool = False):
    """Region list for one scan: centroid (3D, if available), area (mm2
    preferred, else pixels), and optionally the raw polygon_3d points (only
    needed once, when building the baseline reference). Returns None if no
    polygon file exists at all for this scan."""
    aligned_path = scan_dir / f"{scan_name}_burn_polygons_aligned.json"
    plain_path = scan_dir / f"{scan_name}_burn_polygons.json"
    path = aligned_path if aligned_path.exists() else (plain_path if plain_path.exists() else None)
    if path is None:
        return None

    data = json.loads(path.read_text())
    out = []
    for r in data.get("regions", []):
        centroid = r.get("aligned_centroid_xyz") or r.get("centroid")
        entry = {
            "region_id": r.get("region_id"),
            "area_mm2": r.get("area_mm2"),
            "area_pixels": r.get("area_pixels"),
            "centroid_xyz": tuple(centroid) if _valid_point(centroid) else None,
        }
        if need_polygon:
            raw_pts = r.get("polygon_3d") or []
            entry["polygon_3d"] = [p for p in raw_pts if _valid_point(p)]
            entry["n_polygon_3d_dropped"] = len(raw_pts) - len(entry["polygon_3d"])
        out.append(entry)
    return out


def find_baseline_candidates(dataset_dir: Path, patient: str, variant: str):
    """All scans for this patient/variant with at least one 3D-centroid
    region, sorted earliest-first. Returns a list of (scan_dir, scan_name,
    elapsed_days) -- more than one, because the very earliest candidate can
    still turn out to be unusable once we check for area_mm2 too (see
    build_baseline_reference)."""
    candidates = []
    for sd in discover_scan_dirs(dataset_dir, patient):
        m = SCAN_RE.match(sd.name)
        if not m or m.group("variant").upper() != variant:
            continue
        regions = _load_regions(sd, sd.name)
        if not regions or not any(r["centroid_xyz"] for r in regions):
            continue
        candidates.append((elapsed_days(m.group("timepoint").upper()), sd, sd.name))
    candidates.sort(key=lambda c: c[0])
    return candidates


# --------------------------------------------------------- reference build --

def build_baseline_reference(dataset_dir: Path, patient: str, variant: str,
                              pad_mm: float = DEFAULT_PAD_MM):
    """Builds the fixed Day-0 ROI (a set of padded bounding spheres, one per
    baseline region, in the already-registered 3D coordinate frame).

    Walks candidate scans earliest-first and uses the first one that
    actually yields at least one usable (3D-aligned, has area_mm2) region.
    This matters because a scan can have a region with a centroid but no
    computed area_mm2 (e.g. alignment partially failed for that one scan) --
    without this fallback, that single bad scan would silently kill the
    ROI-locked analysis for the entire patient/variant instead of just
    falling back to the next-earliest usable scan."""
    candidates = find_baseline_candidates(dataset_dir, patient, variant)
    skipped = []

    for e, scan_dir, scan_name in candidates:
        regions = _load_regions(scan_dir, scan_name, need_polygon=True)
        spheres = []
        baseline_total_area_mm2 = 0.0
        for r in regions:
            if r["centroid_xyz"] is None or r["area_mm2"] is None:
                continue
            baseline_total_area_mm2 += r["area_mm2"]
            pts = r.get("polygon_3d") or []
            if pts:
                max_r = max(_dist(r["centroid_xyz"], p) for p in pts)
            else:
                # fallback: treat the region as a circle of equivalent area
                max_r = math.sqrt(r["area_mm2"] / math.pi)
            spheres.append({
                "region_id": r["region_id"],
                "center_xyz": list(r["centroid_xyz"]),
                "radius_mm": round(max_r + pad_mm, 3),
                "area_mm2": r["area_mm2"],
            })

        if not spheres:
            skipped.append(scan_name)
            continue  # this scan had regions, but none usable in 3D -- try the next

        m = SCAN_RE.match(scan_name)
        return {
            "patient": patient,
            "variant": variant,
            "baseline_scan": scan_name,
            "baseline_timepoint": m.group("timepoint").upper(),
            "baseline_elapsed_days": e,
            "pad_mm": pad_mm,
            "baseline_total_area_mm2": round(baseline_total_area_mm2, 2),
            "n_baseline_regions": len(spheres),
            "spheres": spheres,
            "skipped_earlier_scans": skipped,
        }

    return None

def measure_scan_against_reference(reference: dict, scan_dir: Path, scan_name: str,
                                    elapsed: int,
                                    max_shift_mm: float = DEFAULT_MAX_SHIFT_MM,
                                    min_flag_area_mm2: float = DEFAULT_MIN_FLAG_AREA_MM2):
    """Applies the (capped) local shift correction, then classifies every
    detected region in this scan as inside the ROI (tracked), or a
    non-trivial external candidate (flagged, never tracked)."""
    regions = _load_regions(scan_dir, scan_name)
    if regions is None:
        return None

    spheres = reference["spheres"]
    usable = [r for r in regions if r["centroid_xyz"] is not None and r["area_mm2"] is not None]

    # Step A -- estimate a small local shift from this scan's own regions
    # that land near (baseline radius + search margin of) any sphere.
    near = []
    for r in usable:
        best = min(spheres, key=lambda s: _dist(s["center_xyz"], r["centroid_xyz"]))
        d = _dist(best["center_xyz"], r["centroid_xyz"])
        if d <= best["radius_mm"] + max_shift_mm:
            near.append((r, best))

    shift = [0.0, 0.0, 0.0]
    shift_capped = False
    if near:
        total_w = sum(r["area_mm2"] for r, _ in near) or 1.0
        for r, s in near:
            w = r["area_mm2"] / total_w
            for i in range(3):
                shift[i] += w * (r["centroid_xyz"][i] - s["center_xyz"][i])
        mag = math.sqrt(sum(v * v for v in shift))
        if mag > max_shift_mm and mag > 0:
            scale = max_shift_mm / mag
            shift = [v * scale for v in shift]
            shift_capped = True

    shifted = [
        {**s, "center_xyz": [s["center_xyz"][i] + shift[i] for i in range(3)]}
        for s in spheres
    ]

    # Step B -- classify every region against the shifted ROI.
    included, external = [], []
    for r in usable:
        best = min(shifted, key=lambda s: _dist(s["center_xyz"], r["centroid_xyz"]))
        d = _dist(best["center_xyz"], r["centroid_xyz"])
        if d <= best["radius_mm"]:
            included.append({**r, "matched_baseline_region_id": best["region_id"],
                              "distance_mm": round(d, 2)})
        elif r["area_mm2"] >= min_flag_area_mm2:
            external.append({**r, "distance_mm": round(d, 2)})

    roi_area_mm2 = sum(r["area_mm2"] for r in included)
    baseline_area = reference["baseline_total_area_mm2"]
    pct = (100.0 * roi_area_mm2 / baseline_area) if baseline_area else None

    m = SCAN_RE.match(scan_name)
    return {
        "scan": scan_name,
        "timepoint": m.group("timepoint").upper(),
        "elapsed_days": elapsed,
        "roi_area_mm2": round(roi_area_mm2, 2),
        "roi_pct_of_baseline": round(pct, 2) if pct is not None else None,
        "n_regions_included": len(included),
        "included_region_ids": [r["region_id"] for r in included],
        "local_shift_mm": {
            "vector": [round(v, 3) for v in shift],
            "magnitude": round(math.sqrt(sum(v * v for v in shift)), 3),
        },
        "shift_capped": shift_capped,
        "external_candidates": [
            {
                "region_id": r["region_id"],
                "area_mm2": r["area_mm2"],
                "centroid_xyz": list(r["centroid_xyz"]),
                "distance_from_nearest_roi_mm": r["distance_mm"],
            }
            for r in external
        ],
        "color_metrics": None,   # placeholder -- see module docstring
        "relief_metrics": None,  # placeholder -- see module docstring
    }


def _flag_external_persistence(per_scan: list, cluster_dist_mm: float = DEFAULT_CLUSTER_DIST_MM):
    """Marks each external candidate as 'confirmed_recurring' if a similarly
    located one also showed up in the immediately preceding scan, else
    'single_scan_only'. Done in place, in chronological order."""
    prev = []
    for s in per_scan:
        current = s["external_candidates"]
        for c in current:
            c["persistence"] = "confirmed_recurring" if any(
                _dist(c["centroid_xyz"], p["centroid_xyz"]) <= cluster_dist_mm for p in prev
            ) else "single_scan_only"
        prev = current


# ------------------------------------------------------------------ plots --

def plot_roi_vs_independent(dataset_dir: Path, patient: str, variant: str,
                             per_scan: list, out_path: Path):
    if len(per_scan) < 2:
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days_roi = [s["elapsed_days"] for s in per_scan]
    areas_roi = [s["roi_area_mm2"] for s in per_scan]

    area_series = load_patient_area_series(dataset_dir, patient).get(variant, [])
    days_ind = [e["elapsed_days"] for e in area_series]
    areas_ind = [e["total_area"] for e in area_series]

    fig, ax = plt.subplots(figsize=(8, 5))
    if days_ind:
        ax.plot(days_ind, areas_ind, marker="o", linewidth=1.5, linestyle="--",
                color="#999999", markersize=5, label="Independent per-scan detection (existing)")
    ax.plot(days_roi, areas_roi, marker="o", linewidth=2.2, color="#1e8449",
            markersize=6, label="ROI-locked (Day-0 fixed region)")

    for s in per_scan:
        if s["shift_capped"]:
            ax.annotate("shift capped\n(low confidence)", (s["elapsed_days"], s["roi_area_mm2"]),
                        textcoords="offset points", xytext=(0, -22), ha="center",
                        fontsize=7, color="#b03a2e")

    ax.set_title(f"{patient} — Variant {variant}: ROI-Locked vs Independent Detection",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Days since Day 0")
    ax.set_ylabel("Burn area (mm²)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------- entry --

def generate_roi_analysis_for_patient(dataset_dir: Path, patient: str,
                                       pad_mm: float = DEFAULT_PAD_MM,
                                       max_shift_mm: float = DEFAULT_MAX_SHIFT_MM,
                                       min_flag_area_mm2: float = DEFAULT_MIN_FLAG_AREA_MM2,
                                       cluster_dist_mm: float = DEFAULT_CLUSTER_DIST_MM):
    """Computes and writes the ROI-locked analysis for every variant this
    patient has, into <patient>/analysis/roi_locked/. Returns a summary dict
    for the caller to log ({variant: {reference, per_scan}})."""
    out_dir = dataset_dir / patient / "analysis" / "roi_locked"

    variants = set()
    for sd in discover_scan_dirs(dataset_dir, patient):
        m = SCAN_RE.match(sd.name)
        if m:
            variants.add(m.group("variant").upper())

    results = {}
    for variant in sorted(variants):
        try:
            reference = build_baseline_reference(dataset_dir, patient, variant, pad_mm=pad_mm)
            if reference is None:
                continue

            scans = []
            for sd in discover_scan_dirs(dataset_dir, patient):
                m = SCAN_RE.match(sd.name)
                if m and m.group("variant").upper() == variant:
                    scans.append((elapsed_days(m.group("timepoint").upper()), sd, sd.name))
            scans.sort(key=lambda t: t[0])

            per_scan = []
            for e, sd, name in scans:
                measured = measure_scan_against_reference(
                    reference, sd, name, e,
                    max_shift_mm=max_shift_mm, min_flag_area_mm2=min_flag_area_mm2,
                )
                if measured:
                    per_scan.append(measured)
        except Exception as exc:
            print(f"  ! {patient} variant {variant}: ROI-locked analysis skipped "
                  f"due to an error ({type(exc).__name__}: {exc})")
            continue

        _flag_external_persistence(per_scan, cluster_dist_mm)

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{patient}_{variant}_roi_reference.json").write_text(
            json.dumps(reference, indent=2))
        for s in per_scan:
            (out_dir / f"{s['scan']}_roi_severity.json").write_text(json.dumps(s, indent=2))

        plot_path = out_dir / "plots" / f"{patient}_{variant}_roi_vs_independent.png"
        plot_roi_vs_independent(dataset_dir, patient, variant, per_scan, plot_path)

        results[variant] = {"reference": reference, "per_scan": per_scan}

    return results


def read_roi_outputs(dataset_dir: Path, patient: str):
    """Reads back what generate_roi_analysis_for_patient() already wrote to
    disk, without recomputing anything -- used by the report step."""
    roi_dir = dataset_dir / patient / "analysis" / "roi_locked"
    if not roi_dir.exists():
        return {}
    out = {}
    for ref_path in sorted(roi_dir.glob(f"{patient}_*_roi_reference.json")):
        m = re.match(rf"^{re.escape(patient)}_([A-Z][A-Z0-9]?)_roi_reference$", ref_path.stem)
        if not m:
            continue
        variant = m.group(1)
        reference = json.loads(ref_path.read_text())
        per_scan = [json.loads(p.read_text())
                    for p in roi_dir.glob(f"{patient}_*_{variant}_roi_severity.json")]
        per_scan.sort(key=lambda s: s["elapsed_days"])
        out[variant] = {"reference": reference, "per_scan": per_scan}
    return out
