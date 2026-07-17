#!/usr/bin/env python3
r"""
step2_generate_patient_report.py
===================================
Builds ONE self-contained HTML report per patient, combining:
  - a summary table (per variant: baseline area, latest area, % change,
    latest timepoint, how many regions are being tracked, any QC flags)
  - the PNG charts from step1_generate_plots.py, embedded inline as
    base64 so the report is a single file you can email/open anywhere
    with no broken image links.

Run step1_generate_plots.py first (or use run_all.py, which does both).

Output:
  <dataset>/<patient>/analysis/<patient>_progress_report.html

USAGE:
  # ALL patients
  python step2_generate_patient_report.py --dataset D:\NahidW\Dataset\face_burn_dataset

  # one patient
  python step2_generate_patient_report.py --dataset D:\NahidW\Dataset\face_burn_dataset --patient PAT01
"""

import argparse
import base64
from pathlib import Path
from datetime import datetime

from analysis_common import (
    discover_patients,
    load_patient_area_series,
    discover_patient_variants_with_tracking,
    load_tracking_json,
    format_timepoint_label,
)

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ font-size: 26px; border-bottom: 3px solid #c0392b; padding-bottom: 10px; }}
  h2 {{ font-size: 20px; margin-top: 40px; color: #c0392b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #f7f2f2; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .improved {{ color: #1e8449; font-weight: 600; }}
  .worsened {{ color: #b03a2e; font-weight: 600; }}
  .flat {{ color: #7f8c8d; font-weight: 600; }}
  img {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; margin: 8px 0 20px; }}
  .caveat {{ background: #fff8e1; border-left: 4px solid #f39c12; padding: 10px 16px;
             font-size: 13px; margin: 16px 0; }}
  .meta {{ color: #888; font-size: 13px; }}
</style>
</head>
<body>
"""

HTML_TAIL = """
</body>
</html>
"""


def _b64_img(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _change_class(pct):
    if pct is None:
        return "flat"
    if pct < -1:
        return "improved"
    if pct > 1:
        return "worsened"
    return "flat"


def build_patient_report(dataset_dir: Path, patient: str) -> Path:
    area_series = load_patient_area_series(dataset_dir, patient)
    plots_dir = dataset_dir / patient / "analysis" / "plots"
    tracked_variants = set(discover_patient_variants_with_tracking(dataset_dir, patient))

    rows = []
    for variant, entries in sorted(area_series.items()):
        if not entries:
            continue
        baseline_e, latest_e = entries[0], entries[-1]
        baseline, latest = baseline_e["total_area"], latest_e["total_area"]
        pct_change = (100.0 * (latest - baseline) / baseline) if baseline else None
        unit_label = "mm²" if baseline_e["area_unit"] == "mm2" else "px"
        rows.append({
            "variant": variant,
            "n_scans": len(entries),
            "baseline_tp": format_timepoint_label(baseline_e["timepoint"]),
            "latest_tp": format_timepoint_label(latest_e["timepoint"]),
            "baseline_area": f"{baseline:.1f} {unit_label}",
            "latest_area": f"{latest:.1f} {unit_label}",
            "pct_change": pct_change,
            "mixed_units": len({e["area_unit"] for e in entries}) > 1,
        })

    table_html = "<table><tr><th>Variant</th><th>Scans</th><th>Baseline</th>"\
                  "<th>Latest</th><th>Baseline area</th><th>Latest area</th>" \
                  "<th>% change</th></tr>"
    for r in rows:
        cls = _change_class(r["pct_change"])
        pct_str = f"{r['pct_change']:+.1f}%" if r["pct_change"] is not None else "n/a"
        warn = " ⚠" if r["mixed_units"] else ""
        table_html += (
            f"<tr><td>{r['variant']}</td><td>{r['n_scans']}</td>"
            f"<td>{r['baseline_tp']}</td><td>{r['latest_tp']}</td>"
            f"<td>{r['baseline_area']}</td><td>{r['latest_area']}{warn}</td>"
            f"<td class='{cls}'>{pct_str}</td></tr>"
        )
    table_html += "</table>"

    mixed_any = any(r["mixed_units"] for r in rows)

    body_parts = [f"<h1>{patient} — Burn Progress Report</h1>"]
    body_parts.append(
        f"<p class='meta'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"from data in <code>{dataset_dir / patient}</code></p>"
    )
    body_parts.append("<h2>Summary</h2>")
    body_parts.append(table_html)
    if mixed_any:
        body_parts.append(
            "<div class='caveat'>⚠ A variant marked above mixes mm² and pixel "
            "measurements across its scans (some scans lack alignment data). "
            "The % change is still computed correctly per-scan-pair only when "
            "units agree within that pair; treat mixed-unit rows as a data-quality "
            "flag worth checking in folder 6 step 1's output.</div>"
        )

    if not rows:
        body_parts.append("<p><em>No area data with at least two scans was found "
                           "for this patient yet — run steps 1–3 in folder 6 first.</em></p>")

    for variant in sorted(area_series.keys()):
        total_png = plots_dir / f"{patient}_{variant}_total_area_over_time.png"
        region_png = plots_dir / f"{patient}_{variant}_region_trajectories.png"
        if not total_png.exists() and not region_png.exists():
            continue
        body_parts.append(f"<h2>Variant {variant}</h2>")
        if total_png.exists():
            body_parts.append(f"<img src='{_b64_img(total_png)}' alt='total area over time'>")
        if region_png.exists():
            body_parts.append(f"<img src='{_b64_img(region_png)}' alt='region trajectories'>")
        elif variant in tracked_variants:
            body_parts.append("<p><em>Region-level tracking data exists for this variant "
                               "but didn't produce a chart (fewer than 2 comparable points "
                               "per region).</em></p>")
        else:
            body_parts.append("<p><em>No region-tracking file (folder 6 step 3) found yet "
                               "for this variant — only total area is shown.</em></p>")

    html = HTML_HEAD.format(title=f"{patient} Progress Report") + "\n".join(body_parts) + HTML_TAIL

    out_dir = dataset_dir / patient / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient}_progress_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Step 2 — generate per-patient HTML report")
    p.add_argument("--dataset", required=True, help="Dataset root")
    p.add_argument("--patient", default=None, help="Limit to one patient; omit for ALL patients")
    args = p.parse_args()

    dataset_dir = Path(args.dataset)
    if args.patient:
        out = build_patient_report(dataset_dir, args.patient)
        print(f"  ✓ {args.patient} → {out}")
        return

    patients = discover_patients(dataset_dir)
    if not patients:
        print(f"  No patient folders found under {dataset_dir}")
        return
    for pd in patients:
        out = build_patient_report(dataset_dir, pd.name)
        print(f"  ✓ {pd.name} → {out}")


if __name__ == "__main__":
    main()
