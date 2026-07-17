#!/usr/bin/env python3
r"""
analysis_common.py
====================
Shared helpers used by every script in this folder. Not meant to be run
directly. Kept dependency-free (stdlib + json only) so it can be imported
without pulling in cv2 / tifffile / the cyberware module that steps 1-3 need.

Data sources this reads (all produced earlier in the pipeline):
  - <scan>_burn_polygons_aligned.json   (folder 6, step 1)  -- preferred,
        has total_burn_area_mm2 (real units) and per-region area_mm2.
  - <scan>_burn_polygons.json           (folder 6, step 1 or folder 5)
        -- fallback if the _aligned file doesn't exist yet for a scan
        (e.g. alignment failed). Has area_pixels but not area_mm2.
  - <scan>_scan_features.json           (folder 6, step 2) -- optional,
        only used to fill in total_burn_area_pixels if even the plain
        _burn_polygons.json total is missing.
  - <patient>_<variant>_tracking.json   (folder 6, step 3, in
        <patient>/tracking/) -- per-region time series + match_log.

None of these are required to all exist for a patient; every loader here
skips missing pieces and reports what it found rather than raising.
"""

import json
import re
from pathlib import Path

SCAN_RE = re.compile(
    r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",
    re.IGNORECASE,
)


def elapsed_days(timepoint: str) -> int:
    """D00 -> 0, D14 -> 14, M02 -> 60 (approx, 30 days/month), etc.
    Same approximation used in step 3 -- fine for plotting/sorting."""
    m = re.match(r'([DM])(\d+)', timepoint.upper())
    if not m:
        return -1
    unit, n = m.group(1), int(m.group(2))
    return n if unit == 'D' else n * 30


def format_timepoint_label(timepoint: str) -> str:
    """D00 -> 'Day 0', D14 -> 'Day 14', M02 -> 'Month 2'."""
    m = re.match(r'([DM])(\d+)', timepoint.upper())
    if not m:
        return timepoint
    unit, n = m.group(1), int(m.group(2))
    return f"Day {n}" if unit == 'D' else f"Month {n}"


def discover_patients(dataset_dir: Path):
    """Every folder directly under the dataset root is treated as a patient."""
    if not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "reports"])


def discover_scan_dirs(dataset_dir: Path, patient: str):
    """<dataset>/<patient>/<timepoint>/<scan>/ -- every leaf scan folder."""
    pat_dir = dataset_dir / patient
    if not pat_dir.exists():
        return []
    return sorted([p for p in pat_dir.glob("*/*") if p.is_dir() and p.name != "tracking" and p.name != "analysis"])


def load_scan_total_area(scan_dir: Path):
    """
    Returns a dict describing one scan's total burn area, or None if no
    usable file is found. Prefers the aligned (mm²) output; falls back to
    plain polygons (px) if alignment hasn't been run for this scan; then to
    the step-2 feature file's pixel total as a last resort.

    {
      "scan": str, "patient": str, "timepoint": str, "variant": str,
      "elapsed_days": int, "total_area": float, "area_unit": "mm2"|"pixels",
      "alignment_status": str or None, "n_regions": int,
    }
    """
    name = scan_dir.name
    m = SCAN_RE.match(name)
    if not m:
        return None
    patient, timepoint, variant = m.group("patient").upper(), m.group("timepoint").upper(), m.group("variant").upper()

    aligned_path = scan_dir / f"{name}_burn_polygons_aligned.json"
    plain_path = scan_dir / f"{name}_burn_polygons.json"
    features_path = scan_dir / f"{name}_scan_features.json"

    if aligned_path.exists():
        data = json.loads(aligned_path.read_text())
        total = data.get("total_burn_area_mm2")
        if total is not None:
            return {
                "scan": name, "patient": patient, "timepoint": timepoint, "variant": variant,
                "elapsed_days": elapsed_days(timepoint), "total_area": float(total),
                "area_unit": "mm2", "alignment_status": data.get("alignment_status"),
                "n_regions": len(data.get("regions", [])),
            }
        # aligned file exists but total wasn't computed for some reason -- fall through

    if plain_path.exists():
        data = json.loads(plain_path.read_text())
        regions = data.get("regions", [])
        total_px = sum(r.get("area_pixels", 0) for r in regions)
        if regions:
            return {
                "scan": name, "patient": patient, "timepoint": timepoint, "variant": variant,
                "elapsed_days": elapsed_days(timepoint), "total_area": float(total_px),
                "area_unit": "pixels", "alignment_status": data.get("alignment_status", "no_alignment_file"),
                "n_regions": len(regions),
            }

    if features_path.exists():
        data = json.loads(features_path.read_text())
        total_px = data.get("total_burn_area_pixels")
        if total_px is not None:
            return {
                "scan": name, "patient": patient, "timepoint": timepoint, "variant": variant,
                "elapsed_days": elapsed_days(timepoint), "total_area": float(total_px),
                "area_unit": "pixels", "alignment_status": None,
                "n_regions": data.get("num_regions", 0),
            }

    return None


def load_patient_area_series(dataset_dir: Path, patient: str):
    """
    Returns {variant: [scan_area_dict, ...]} sorted by elapsed_days,
    for every variant found under this patient. Mixed units (mm2 vs
    pixels) within one variant's series are flagged via each entry's
    'area_unit' -- callers should check before assuming a single unit.
    """
    series = {}
    for sd in discover_scan_dirs(dataset_dir, patient):
        entry = load_scan_total_area(sd)
        if entry is None:
            continue
        series.setdefault(entry["variant"], []).append(entry)
    for variant in series:
        series[variant].sort(key=lambda e: e["elapsed_days"])
    return series


def load_tracking_json(dataset_dir: Path, patient: str, variant: str):
    """Returns the parsed <patient>_<variant>_tracking.json, or None if absent."""
    path = dataset_dir / patient / "tracking" / f"{patient}_{variant}_tracking.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def discover_patient_variants_with_tracking(dataset_dir: Path, patient: str):
    """Variant letters (e.g. ['A','B']) that have a tracking.json for this patient."""
    tdir = dataset_dir / patient / "tracking"
    if not tdir.exists():
        return []
    variants = []
    for f in sorted(tdir.glob(f"{patient}_*_tracking.json")):
        stem = f.stem  # e.g. PAT01_A_tracking
        m = re.match(rf"^{re.escape(patient)}_([A-Z][A-Z0-9]?)_tracking$", stem, re.IGNORECASE)
        if m:
            variants.append(m.group(1).upper())
    return variants


def track_area_timeseries(track_entries: list):
    """
    Given one track's list of per-scan entries (as stored in tracking.json's
    "tracks"), return (elapsed_days_list, area_list, area_unit) skipping any
    scan where this track was a gap ("not_detected_this_scan", no area on
    record). Prefers area_mm2 per entry, falls back to area_pixels.
    """
    days, areas = [], []
    unit = None
    for e in track_entries:
        area = e.get("area_mm2")
        this_unit = "mm2"
        if area is None:
            area = e.get("area_pixels")
            this_unit = "pixels"
        if area is None:
            continue  # gap entry, e.g. status == "not_detected_this_scan"
        days.append(e["elapsed_days"])
        areas.append(area)
        unit = unit or this_unit
    return days, areas, unit
