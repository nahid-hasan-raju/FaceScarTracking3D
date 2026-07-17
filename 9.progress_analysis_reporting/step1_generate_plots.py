#!/usr/bin/env python3
r"""
step1_generate_plots.py
=========================
Reads the outputs already produced by folder 6 (steps 1-3) and generates
PNG charts showing burn progress over time. Doesn't recompute anything --
pure visualization layer.

For each patient, and each camera-angle variant (A, B, C, ...) found for
that patient, produces two charts:

  1. <patient>_<variant>_total_area_over_time.png
       Total burn area (all regions summed) vs. time since Day 0.
       This is the headline "is it healing" chart -- sourced directly
       from step 1's per-scan total_burn_area_mm2 (or area_pixels
       fallback), so it doesn't depend on region-to-region tracking
       having gone well.

  2. <patient>_<variant>_region_trajectories.png
       One line per tracked region (from folder 6 step 3's
       tracking.json), showing how each individual burn region's area
       changed over time. More granular than (1), but only available
       once tracking.json exists for that variant, and a line breaks
       wherever a region has a "not_detected_this_scan" gap.

Output location:
  <dataset>/<patient>/analysis/plots/*.png

USAGE:
  # ALL patients
  python step1_generate_plots.py --dataset D:\NahidW\Dataset\face_burn_dataset

  # one patient
  python step1_generate_plots.py --dataset D:\NahidW\Dataset\face_burn_dataset --patient PAT01
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display available / needed -- file output only
import matplotlib.pyplot as plt

from analysis_common import (
    discover_patients,
    load_patient_area_series,
    load_tracking_json,
    discover_patient_variants_with_tracking,
    track_area_timeseries,
    format_timepoint_label,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 11,
})


def _unit_label(unit: str) -> str:
    return "Area (mm²)" if unit == "mm2" else "Area (pixels)"


def plot_total_area_over_time(entries: list, patient: str, variant: str, out_path: Path):
    """entries: sorted list of dicts from load_patient_area_series()[variant]."""
    if len(entries) < 2:
        return False

    days = [e["elapsed_days"] for e in entries]
    areas = [e["total_area"] for e in entries]
    units = {e["area_unit"] for e in entries}
    unit = entries[0]["area_unit"] if len(units) == 1 else "mixed"
    labels = [format_timepoint_label(e["timepoint"]) for e in entries]

    baseline = areas[0]
    latest = areas[-1]
    pct_change = (100.0 * (latest - baseline) / baseline) if baseline else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, areas, marker="o", linewidth=2, color="#c0392b", markersize=6)
    ax.fill_between(days, areas, color="#c0392b", alpha=0.08)

    for d, a, lbl in zip(days, areas, labels):
        ax.annotate(lbl, (d, a), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#555555")

    title = f"{patient} — Variant {variant}: Total Burn Area Over Time"
    if pct_change is not None:
        direction = "reduction" if pct_change < 0 else "increase"
        title += f"\n{abs(pct_change):.1f}% {direction} from baseline to latest scan"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Days since Day 0")
    ax.set_ylabel(_unit_label(unit) if unit != "mixed" else "Area (units vary — see report)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_region_trajectories(tracking_data: dict, patient: str, variant: str, out_path: Path):
    tracks = tracking_data.get("tracks", {})
    if not tracks:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    cmap = plt.get_cmap("tab10")

    for idx, (track_id, entries) in enumerate(sorted(tracks.items(), key=lambda kv: int(kv[0]))):
        days, areas, unit = track_area_timeseries(entries)
        if len(days) < 2:
            continue
        ax.plot(days, areas, marker="o", linewidth=1.8, markersize=5,
                 color=cmap(idx % 10), label=f"Region {track_id}")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_title(f"{patient} — Variant {variant}: Individual Region Trajectories",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Days since Day 0")
    ax.set_ylabel("Area (mm² if aligned, else pixels)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def generate_for_patient(dataset_dir: Path, patient: str) -> dict:
    plots_dir = dataset_dir / patient / "analysis" / "plots"
    area_series = load_patient_area_series(dataset_dir, patient)
    tracked_variants = set(discover_patient_variants_with_tracking(dataset_dir, patient))

    written = []
    for variant, entries in sorted(area_series.items()):
        out1 = plots_dir / f"{patient}_{variant}_total_area_over_time.png"
        if plot_total_area_over_time(entries, patient, variant, out1):
            written.append(out1)

        if variant in tracked_variants:
            tracking_data = load_tracking_json(dataset_dir, patient, variant)
            if tracking_data:
                out2 = plots_dir / f"{patient}_{variant}_region_trajectories.png"
                if plot_region_trajectories(tracking_data, patient, variant, out2):
                    written.append(out2)

    if written:
        print(f"  ✓ {patient}: {len(written)} plot(s) → {plots_dir}")
    else:
        print(f"  – {patient}: no plots generated (need ≥2 scans with area data per variant)")
    return {"patient": patient, "plots": written}


def main():
    p = argparse.ArgumentParser(description="Step 1 — generate progress plots (PNG)")
    p.add_argument("--dataset", required=True, help="Dataset root")
    p.add_argument("--patient", default=None, help="Limit to one patient; omit for ALL patients")
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    if args.patient:
        generate_for_patient(dataset_dir, args.patient)
        return

    patients = discover_patients(dataset_dir)
    if not patients:
        print(f"  No patient folders found under {dataset_dir}")
        return
    print(f"  Found {len(patients)} patient folder(s)\n")
    for pd in patients:
        generate_for_patient(dataset_dir, pd.name)


if __name__ == "__main__":
    main()
