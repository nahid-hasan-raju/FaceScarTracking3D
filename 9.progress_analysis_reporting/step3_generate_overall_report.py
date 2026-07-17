#!/usr/bin/env python3
r"""
step3_generate_overall_report.py
===================================
Builds the ACROSS-ALL-PATIENTS summary: one HTML report + charts saved
directly at the dataset root (NOT inside any single patient's folder),
since this view spans every patient.

IMPORTANT CAVEAT baked into this report: each (patient, variant) is
treated as its own independent series. Variants are different camera
angles of the same patient and may capture overlapping burn area, so
patients with more variants are NOT "worse" just because they have more
rows -- don't sum across variants for a patient without checking that
first in folder 6's data.

Produces:
  <dataset>/overall_progress_report.html
  <dataset>/analysis/plots/overall_pct_change_by_series.png
  <dataset>/analysis/plots/overall_normalized_trajectories.png

USAGE:
  python step3_generate_overall_report.py --dataset D:\NahidW\Dataset\face_burn_dataset
"""

import argparse
import base64
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_common import discover_patients, load_patient_area_series, format_timepoint_label

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 11,
})


def gather_all_series(dataset_dir: Path):
    """Returns list of dicts, one per (patient, variant) series with >=2 scans."""
    out = []
    for pd in discover_patients(dataset_dir):
        patient = pd.name
        series = load_patient_area_series(dataset_dir, patient)
        for variant, entries in sorted(series.items()):
            if len(entries) < 2:
                continue
            baseline, latest = entries[0], entries[-1]
            pct_change = (100.0 * (latest["total_area"] - baseline["total_area"])
                          / baseline["total_area"]) if baseline["total_area"] else None
            out.append({
                "patient": patient, "variant": variant, "entries": entries,
                "baseline": baseline, "latest": latest, "pct_change": pct_change,
                "mixed_units": len({e["area_unit"] for e in entries}) > 1,
            })
    return out


def plot_pct_change_bar(all_series: list, out_path: Path):
    usable = [s for s in all_series if s["pct_change"] is not None]
    if not usable:
        return False
    usable.sort(key=lambda s: s["pct_change"])
    labels = [f"{s['patient']} ({s['variant']})" for s in usable]
    values = [s["pct_change"] for s in usable]
    colors = ["#1e8449" if v < 0 else "#b03a2e" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(usable))))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel("% change in total burn area, baseline → latest scan")
    ax.set_title("Burn Area Change by Patient / Variant\n(green = reduced, red = increased)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_normalized_trajectories(all_series: list, out_path: Path):
    """Every series normalized to 100% at baseline, overlaid, so healing
    RATE is comparable across patients regardless of starting size."""
    usable = [s for s in all_series if len(s["entries"]) >= 2]
    if not usable:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    for idx, s in enumerate(usable):
        entries = s["entries"]
        baseline = entries[0]["total_area"]
        if not baseline:
            continue
        days = [e["elapsed_days"] for e in entries]
        pct = [100.0 * e["total_area"] / baseline for e in entries]
        ax.plot(days, pct, marker="o", markersize=4, linewidth=1.5,
                 color=cmap(idx % 20), label=f"{s['patient']} ({s['variant']})", alpha=0.85)

    ax.axhline(100, color="#444444", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Days since Day 0")
    ax.set_ylabel("Burn area, % of baseline")
    ax.set_title("Normalized Healing Trajectories — All Patients",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _b64_img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_overall_report(dataset_dir: Path) -> Path:
    all_series = gather_all_series(dataset_dir)
    plots_dir = dataset_dir / "analysis" / "plots"

    bar_path = plots_dir / "overall_pct_change_by_series.png"
    traj_path = plots_dir / "overall_normalized_trajectories.png"
    has_bar = plot_pct_change_bar(all_series, bar_path)
    has_traj = plot_normalized_trajectories(all_series, traj_path)

    n_patients = len({s["patient"] for s in all_series})
    improved = sum(1 for s in all_series if s["pct_change"] is not None and s["pct_change"] < -1)
    worsened = sum(1 for s in all_series if s["pct_change"] is not None and s["pct_change"] > 1)
    flat = len(all_series) - improved - worsened
    mixed_flag_count = sum(1 for s in all_series if s["mixed_units"])

    rows_html = "<table><tr><th>Patient</th><th>Variant</th><th>Scans</th>" \
                "<th>Baseline</th><th>Latest</th><th>% change</th></tr>"
    for s in sorted(all_series, key=lambda s: (s["patient"], s["variant"])):
        pct_str = f"{s['pct_change']:+.1f}%" if s["pct_change"] is not None else "n/a"
        cls = "flat"
        if s["pct_change"] is not None:
            cls = "improved" if s["pct_change"] < -1 else ("worsened" if s["pct_change"] > 1 else "flat")
        warn = " ⚠" if s["mixed_units"] else ""
        rows_html += (
            f"<tr><td>{s['patient']}</td><td>{s['variant']}</td>"
            f"<td>{len(s['entries'])}</td>"
            f"<td>{format_timepoint_label(s['baseline']['timepoint'])}</td>"
            f"<td>{format_timepoint_label(s['latest']['timepoint'])}{warn}</td>"
            f"<td class='{cls}'>{pct_str}</td></tr>"
        )
    rows_html += "</table>"

    body = [f"<h1>All-Patients Burn Progress Report</h1>"]
    body.append(f"<p class='meta'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                f"across {n_patients} patient(s), {len(all_series)} patient/variant series, "
                f"from <code>{dataset_dir}</code></p>")
    body.append(
        f"<h2>At a glance</h2><ul>"
        f"<li><span class='improved'>{improved}</span> series show a meaningful area reduction (&gt;1%)</li>"
        f"<li><span class='worsened'>{worsened}</span> series show a meaningful area increase (&gt;1%)</li>"
        f"<li><span class='flat'>{flat}</span> series are roughly flat (within ±1%)</li>"
        f"</ul>"
    )
    body.append(
        "<div class='caveat'>⚠ Each row is one (patient, camera-angle variant) series, "
        "treated independently. Variants are different views of the same patient and may "
        "overlap in what they capture, so don't sum variants together for a patient without "
        "checking for overlap first. Rows marked ⚠ mix mm² and pixel units across their own "
        "scans (partial alignment coverage) — treat the % change on those as approximate.</div>"
    )
    if has_traj:
        body.append("<h2>Normalized Healing Trajectories</h2>")
        body.append(f"<img src='{_b64_img(traj_path)}' alt='normalized trajectories'>")
    if has_bar:
        body.append("<h2>% Change by Patient / Variant</h2>")
        body.append(f"<img src='{_b64_img(bar_path)}' alt='pct change bar chart'>")
    body.append("<h2>Detail Table</h2>")
    body.append(rows_html)

    if not all_series:
        body.append("<p><em>No patient/variant series with at least two scans were found. "
                     "Run folder 6 steps 1-3 first, then step1_generate_plots.py in this "
                     "folder before generating this report.</em></p>")

    from step2_generate_patient_report import HTML_HEAD, HTML_TAIL
    html = HTML_HEAD.format(title="All-Patients Burn Progress Report") + "\n".join(body) + HTML_TAIL

    out_path = dataset_dir / "overall_progress_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Step 3 — generate the overall (all-patients) report")
    p.add_argument("--dataset", required=True, help="Dataset root")
    args = p.parse_args()

    out = build_overall_report(Path(args.dataset))
    print(f"  ✓ Overall report → {out}")


if __name__ == "__main__":
    main()
