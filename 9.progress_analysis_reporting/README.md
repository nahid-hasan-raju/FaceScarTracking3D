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
| `roi_lock.py` | NEW — Day-0 fixed-ROI tracking (see its own section below). Not run directly; called from `step1_generate_plots.py`. |
| `step1_generate_plots.py` | Generates PNG charts per patient/variant, then runs the ROI-locked analysis. |
| `step2_generate_patient_report.py` | Builds one self-contained HTML report per patient (now includes an ROI-Locked Tracking section). |
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
│       ├── PAT01_progress_report.html    <- PAT01-only report (both analyses)
│       ├── plots/
│       │     PAT01_A_total_area_over_time.png
│       │     PAT01_A_region_trajectories.png
│       │     PAT01_B_total_area_over_time.png
│       │     PAT01_B_region_trajectories.png
│       └── roi_locked/                   <- NEW, separate subfolder
│             PAT01_A_roi_reference.json       <- the Day-0 ROI definition
│             PAT01_D00_A_roi_severity.json    <- one per scan
│             PAT01_D14_A_roi_severity.json
│             plots/PAT01_A_roi_vs_independent.png
├── PAT02/
│   └── analysis/ ...
└── ...
```

Nothing under `<dataset>/<patient>/D00/...` etc. is modified — this folder
only adds new `analysis/` subfolders and the one root-level HTML file.

## ROI-Locked Tracking (new)

The existing approach (above) re-detects burn regions independently on every
scan. That's a good baseline view, but an imperfect detector can make the
total area jump around day-to-day for reasons that have nothing to do with
healing — e.g. a partially-healed wound segmenting more noisily/fragmented
than a fresh one.

`roi_lock.py` adds a second, complementary measurement: it fixes the tracked
region to what was detected on the **baseline (earliest) scan** — as a padded
bounding sphere per baseline region, in the already-registered 3D coordinate
frame — and on every later scan, only sums the area of regions that still
fall inside that same fixed region. A later scan's own detection is used
*only* to apply a small, capped local re-centering correction (for the minor
day-to-day registration drift that's expected even with good alignment); it
never changes the ROI's size or contributes area on its own.

Regions that fall clearly outside the ROI are never added to the tracked
number. They're logged as "external candidates," and only escalated to a
`confirmed_recurring` flag (surfaced in the report as a manual-review note)
if a similarly-located one shows up in the very next scan too — a single
stray false-positive can't trigger a false alarm.

Tunable parameters (defaults in `roi_lock.py`, can be overridden by calling
`generate_roi_analysis_for_patient(...)` with different kwargs):

| Parameter | Default | What it controls |
|---|---|---|
| `pad_mm` | 8.0 | Margin added around each baseline region's own detected extent, when building the fixed ROI. |
| `max_shift_mm` | 6.0 | Cap on the local re-centering correction. If your dataset's registration drift regularly exceeds this, scans will show `shift_capped: true` and matching regions may get missed (see caveat below) — raise this if so. |
| `min_flag_area_mm2` | 50.0 | Minimum size for a region outside the ROI to be worth logging as an external candidate (avoids flagging tiny detector noise). |
| `cluster_dist_mm` | 6.0 | How close two external candidates on consecutive scans need to be to count as "the same spot" (→ `confirmed_recurring`). |

**Not yet implemented:** color/erythema scoring and 3D relief/height scoring
*within* the locked ROI (both are `None` placeholders in the output schema).
These need pixel-level TIF access and the full facial mesh respectively —
see the module docstring in `roi_lock.py` for the planned integration
points.

**Real-data caveat worth knowing:** when tested against actual PAT01 data,
one scan's dominant region missed the fixed ROI by ~6.5mm — just past the
default 6mm shift cap — and got flagged as external instead of tracked. This
isn't a bug; it's the system correctly refusing to silently stretch the
correction further than its cap allows. If you see this happening often for
a patient, it means their real day-to-day registration drift is larger than
the default assumption — worth raising `max_shift_mm` (and/or `pad_mm`) for
that dataset rather than trusting a wider correction blindly.

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
