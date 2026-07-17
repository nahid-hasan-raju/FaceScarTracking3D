# 9. Progress Analysis & Reporting

Reads what folder **6** (scar tracking) already produced and turns it into
plots and human-readable reports. Doesn't recompute any measurements or
tracking — pure visualization/reporting layer on top of:

- `<scan>_burn_polygons_aligned.json` (or `_burn_polygons.json` as fallback)
  — from folder 6 step 1
- `<scan>_scan_features.json` — from folder 6 step 2 (used as a last-resort
  fallback for total area if step 1 hasn't been run for a scan yet)
- `<patient>_<variant>_tracking.json` — from folder 6 step 3, in
  `<patient>/tracking/`

## Files

| File | Purpose |
|---|---|
| `analysis_common.py` | Shared data-loading helpers. Not run directly. |
| `step1_generate_plots.py` | Generates PNG charts per patient/variant. |
| `step2_generate_patient_report.py` | Builds one self-contained HTML report per patient. |
| `step3_generate_overall_report.py` | Builds the single cross-patient report. |
| `run_all.py` | Runs all of the above in the right order, for convenience. |

## Where things get written

```
<dataset>/
├── overall_progress_report.html          <- ALL-PATIENTS report (folder root)
├── analysis/plots/                       <- ALL-PATIENTS charts
│     overall_pct_change_by_series.png
│     overall_normalized_trajectories.png
│
├── PAT01/
│   ├── D00/ ... D14/ ...                 <- existing scan folders (untouched)
│   ├── tracking/                         <- existing, from folder 6 step 3
│   └── analysis/
│       ├── PAT01_progress_report.html    <- PAT01-only report
│       └── plots/
│             PAT01_A_total_area_over_time.png
│             PAT01_A_region_trajectories.png
│             PAT01_B_total_area_over_time.png
│             PAT01_B_region_trajectories.png
├── PAT02/
│   └── analysis/ ...
└── ...
```

Nothing under `<dataset>/<patient>/D00/...` etc. is modified — this folder
only adds new `analysis/` subfolders and the one root-level HTML file.

## Usage

```bash
# Everything: every patient's plots + report, then the overall report
python run_all.py --dataset D:\path\to\face_burn_dataset

# Just one patient (skips the overall report, since that's dataset-wide)
python run_all.py --dataset D:\path\to\face_burn_dataset --patient PAT01

# Or run the three steps separately if you only need one piece:
python step1_generate_plots.py            --dataset D:\path\to\face_burn_dataset
python step2_generate_patient_report.py   --dataset D:\path\to\face_burn_dataset
python step3_generate_overall_report.py   --dataset D:\path\to\face_burn_dataset
```

Re-running is always safe — every output file is fully regenerated
(overwritten) each time, nothing is appended to.

## What each chart/report actually shows

- **Total area over time** (per patient, per variant): the headline
  "is this healing" chart. Sourced from folder 6 step 1's
  `total_burn_area_mm2` (always computed there, alignment or not), so it
  doesn't depend on region-to-region tracking having gone well.
- **Region trajectories** (per patient, per variant): one line per
  individually-tracked burn region, from folder 6 step 3's `tracking.json`.
  More granular ("this specific scar is healing faster than that one") but
  only appears once tracking.json exists, and needs ≥2 comparable points per
  region to draw a line.
- **Per-patient HTML report**: summary table (baseline vs. latest area, %
  change, scan count) + the charts above, all inlined as base64 so it's one
  file you can email or open anywhere without broken image links.
- **Overall report**: normalized trajectories (every patient's series
  rescaled to start at 100%, so healing *rate* is comparable regardless of
  how big the burn started), a sorted bar chart of % change per
  patient/variant, and a detail table.

## Known caveats (also called out inline in the reports themselves)

1. **Variants aren't merged.** Each (patient, variant) is its own series.
   Variants are different camera angles and may capture overlapping burn
   area — the overall report deliberately does *not* sum variants together
   for a patient, and flags this in its caveat box.
2. **Mixed units.** If a patient's variant has some scans aligned (mm²) and
   others not (pixels, because alignment failed or wasn't run for that
   scan), the % change is still computed correctly scan-to-scan, but the
   report marks that row with ⚠ since baseline and latest might be in
   different units. Worth checking folder 6 step 1's `alignment_status`
   field for those scans.
3. **Missing data just isn't plotted.** If a scan or a whole variant is
   missing for some timepoint, the chart simply skips that point — it does
   not interpolate or fake a value.
