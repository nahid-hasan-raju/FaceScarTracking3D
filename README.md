# Face Burn Scar — Full Pipeline Overview

This document explains what every numbered folder in `D:\NahidW\Coding` does,
what each script reads and writes, and what order to run things in to go
from raw 3D scanner output to a "how much has this patient healed" report.

If you're new here: read this top-to-bottom once, then use the per-folder
sections as reference while you actually run things.

Live page: https://facescartracking3d-live-119i.onrender.com/ 
Live page Repo: https://github.com/nahid-hasan-raju/FaceScarTracking3D-live 
---

## 1. What this project does, in one paragraph

Patients with facial burns are scanned with a Cyberware 3D face scanner at
multiple timepoints (Day 0, Day 14, Day 28, Month 2, Month 6, ...) and from
multiple camera angles ("variants" A/B/C/D). The pipeline turns each raw scan
into a 3D point cloud, detects facial landmarks to align every timepoint into
the same coordinate frame, runs a trained segmentation model to find the burn
region, measures that region's real-world area (mm²) and how it moves, and
finally tracks each burn region's size over time to show whether — and how
fast — it's healing.

## 2. Folder-by-folder, in pipeline order

```
1.  <raw-to-3D>              range file + tif        -> .ply
2.  <landmark-detection>     range file + tif         -> landmarks/*.json
3.  <face-alignment>         .ply + landmarks.json    -> *_aligned.ply, *_alignment.json
4.  <segmentation-models>    (offline training/eval, not run per-scan)
5.  <burn-segmentation-3d>   tif + range + alignment  -> *_burn_polygons.json, *_burn3d.ply
6.  <face-scar-tracking>     burn_polygons + alignment -> tracking/*.json  (progress over time)
7.  <color-normalization>    (exploratory / not finalized)
8.  <webapp>                 (live demo, separate — not covered in depth here)
9.  <progress-analysis>      tracking.json            -> plots + HTML reports
```

Folders 4 and 7 don't process scans one-by-one like the rest — folder 4 is
where models are trained once and produces checkpoint files that folder 5
loads; folder 7 is a still-in-progress side project for reducing
lighting/camera color drift before segmentation.

Every other folder reads files that a previous folder already wrote, and
writes new files back into the **same scan folder** (`<dataset>/PAT01/D00/
PAT01_D00_A/`) rather than a separate output tree — so by the time you've run
folders 1→6, every scan folder has accumulated all of these files (see
section 4 for the exact list).

---

## 3. Folder 1 — Raw scan → 3D point cloud

**Script:** `step1_convert_all_to_3d.py`

Parses the Cyberware scanner's raw binary range file (cylindrical
radius-per-(θ,z) grid), converts it to Cartesian XYZ, and colors each 3D
point using the matching `.tif` texture.

| | |
|---|---|
| **Reads** | `<scan>` (range file, no extension), `<scan>.tif` |
| **Writes** | `<scan>.ply` (same folder) |
| **CLI** | `--dataset` (required), `--patient`, `--timepoint`, `--scan` (all optional filters), `--overwrite` |
| **Deps** | numpy, pillow |

```powershell
python step1_convert_all_to_3d.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python step1_convert_all_to_3d.py --dataset "D:/..." --patient PAT01 --timepoint D00 --scan A
```

This is the first thing that has to run for a scan to be usable by
anything downstream.

---

## 4. Folder 2 — Facial landmark detection

**Script:** `step2_detect_landmarks.py`

Runs MediaPipe Face Mesh (~478 points) on the `.tif` texture in 2D, then
converts every detected point into real 3D coordinates using the *same*
range-file grid math as folder 1 — so depth comes from the actual scanner
reading, not MediaPipe's own (less accurate) depth guess. Has a fallback
detection ladder (strict → lenient → lenient+contrast-enhanced) for
difficult faces, and flags in its output which method was needed.

| | |
|---|---|
| **Reads** | `<scan>` (range file), `<scan>.tif` |
| **Writes** | `landmarks/<scan>_landmarks.json`, optionally `landmarks/<scan>_landmarks_debug.png` |
| **CLI** | `--dataset` (required), `--patient`, `--timepoint`, `--scan`, `--scan-id` (shortcut for all three), `--overwrite`, `--debug` |
| **Deps** | mediapipe, opencv-python, numpy, pillow (auto-downloads a ~4MB model file on first run) |

```powershell
python step2_detect_landmarks.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python step2_detect_landmarks.py --dataset "D:/..." --scan-id PAT01_D00_A --debug
```

Only depends on folder 1's inputs (range file + tif), not folder 1's output —
but its own output feeds folder 3.

---

## 5. Folder 3 — Landmark-based alignment

**Scripts:** `step3_align_landmarks.py` (does the work), `visualize_alignment.py` (QC)

Aligns every scan of a patient into one consistent coordinate frame using
only the landmarks folder 2 already found — no manual axis guessing. Picks a
reference scan per (patient, variant) — default: earliest timepoint,
variant A preferred — then computes the rigid transform (rotation +
translation, Kabsch/SVD, no scaling) that best maps each other scan's
bone-anchored landmarks (nose, eyes, forehead — deliberately *not*
mouth/chin, which shift with expression) onto the reference's. Variant D
gets an extended fallback landmark set (adds mouth/chin) since it often
doesn't capture eyes/forehead.

| | |
|---|---|
| **Reads** | `<scan>.ply` (folder 1), `landmarks/<scan>_landmarks.json` (folder 2) |
| **Writes** | `<scan>_aligned.ply`, `<scan>_alignment.json` (rotation + translation + fit residual), optionally `<scan>_aligned_preview.png` |
| **CLI** | `--dataset` (required), `--patient`, `--timepoint`, `--scan`, `--scan-id`, `--reference` (override which scan is the reference), `--overwrite`, `--debug` |
| **Deps** | numpy, matplotlib |

```powershell
python step3_align_landmarks.py --dataset "D:/..." --patient PAT01
python step3_align_landmarks.py --dataset "D:/..." --patient PAT01 --reference PAT01_D00_A
```

Reports a fit residual (RMS/max, in mm) per scan and flags anything above
5mm as high-residual — worth a visual sanity check. That's what
`visualize_alignment.py` is for:

| | |
|---|---|
| **Reads** | `<scan>_aligned.ply` (all timepoints of one patient+variant), `landmarks/<scan>_landmarks.json` (for view direction) |
| **Writes** | `<dataset>/alignment_qc/<patient>_<variant>_overlay.png` |

Overlays every timepoint in a different color, in one 2D projection. If
alignment worked, stable anatomy (nose, eyes, hairline) should look like one
blended color; only the burn region (and any real change) should show
visibly separated colors.

```powershell
python visualize_alignment.py --dataset "D:/..." --patient PAT01
```

**Note:** `<scan>_alignment.json` is the single most important artifact this
folder produces — folders 5 and 6 both read it directly.

---

## 6. Folder 4 — Burn segmentation models (training & evaluation)

This folder doesn't touch the dataset's scan folders at all — it trains and
evaluates 4 candidate segmentation models on a separate train/valid/test
image+mask dataset, and produces the checkpoint files that folder 5 actually
uses in production. Each method has its own subfolder with matching
`train.py` / `predict.py`:

| Method | Subfolder (implied) | Notes |
|---|---|---|
| SAM2 (Hiera-Large, fine-tuned) | `method1_sam2/` | Full-image prompt, no bbox hint — "honest" version, trains only the mask decoder |
| MedSAM (SAM ViT-B, fine-tuned) | `method2_medsam/` | Same full-image-prompt philosophy |
| UNet++ (EfficientNet-B4 encoder) | `method3_unetpp/` | No prompt needed, plain semantic segmentation, class-imbalance-weighted loss |
| SegFormer-B5 (fine-tuned) | `method4_segformer/` | No prompt needed, 2-class (background/burn) semantic segmentation |

Shared root-level scripts (work across all 4 methods):

| Script | Purpose |
|---|---|
| `test_model.py` | Run any one trained model on a single image or a folder of images (no ground-truth needed) — produces `*_mask.png` + `*_overlay.png` |
| `test_model_interactive.py` | Interactive variant of the above *(not yet reviewed in detail — paste it in if it differs meaningfully from `test_model.py`)* |
| `evaluate_all.py` | Compares Dice/IoU/Precision/Recall across all 4 methods side-by-side, reading each method's already-generated `outputs/<method>/*_mask.png` against ground-truth masks |
| `tune_threshold.py` | Sweeps binarization thresholds per method on the validation set to find the Dice-optimal cutoff, then re-evaluates the test set with that tuned threshold |

**Expected folder layout** (inferred from the scripts' `BASE`-relative paths):
```
4.face_burn_segmentation_models/
  checkpoints/{sam2,medsam,unetpp,segformer}/...
  dataset/{train,valid,test}/{images,masks}/
  outputs/{sam2,medsam,unetpp,segformer}/
  segment-anything-2/          <- cloned repo, for SAM2
  utils/{metrics.py, losses.py, dataset.py}
```

```powershell
python method1_sam2/train.py
python method1_sam2/predict.py
python evaluate_all.py
python tune_threshold.py
```

The output of this whole folder that matters for the rest of the pipeline is
just the **checkpoint files** (`checkpoints/<method>/best.pth` or, for
SegFormer, the whole checkpoint folder) — folder 5's `config.yaml` points at
these.

---

## 7. Folder 5 — Burn segmentation → colored 3D point cloud

**Main script:** `step1_burn_segmented_into_3d_pipeline.py`, configured via `config.yaml`

Loads a scan's `.tif`, runs whichever trained model `config.yaml` selects
(pointing at folder 4's checkpoints), gets a binary burn mask, blends burn
pixels toward red (skin texture still visible underneath) with a yellow
boundary outline, then wraps that colored image back onto the scan's 3D
geometry using the *same* alignment folder 3 already computed for this scan.

| | |
|---|---|
| **Reads** | `<scan>.tif`, `<scan>` (range file), `<scan>_alignment.json` (folder 3, optional — see below) |
| **Writes** (all in-place, same scan folder) | `<scan>_seg.tif` (segmented texture), `<scan>_burn3d.ply` (colored 3D point cloud), `<scan>_burn_polygons.json` (2D polygons + per-region confidence + a rough per-vertex 3D projection), optionally `<scan>_burn_mask.png` with `--save-mask` |
| **CLI** | `--dataset` / `--scandir`, `--patient`, `--timepoint`, `--model`, `--ckpt`, `--base-ckpt`, `--sam2-repo`, `--threshold`, `--fine-yaw`, `--save-mask`, `--show` — all model/checkpoint settings can instead live in `config.yaml` so you don't have to type them every run |
| **Deps** | numpy, pillow, open3d, opencv-python, pyyaml, imagecodecs, tifffile + whichever model's own deps (see folder 4) |

**Important behavior:** if `<scan>_alignment.json` doesn't exist yet for a
scan (folder 3 hasn't processed it, or failed), this script does **not**
guess an alignment — it saves the point cloud unaligned and marks it as such,
so it's still usable on its own (area, shape) but flagged as not yet
comparable across timepoints.

**Per-patient threshold override:** `config.yaml` has a `patient_thresholds`
map (e.g. `PAT01: 0.92`) that overrides the general `threshold` default for
specific patients — worth checking here first if a patient's burn mask looks
off.

```powershell
# config.yaml sitting next to the script is auto-loaded if you don't pass --config
python step1_burn_segmented_into_3d_pipeline.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python step1_burn_segmented_into_3d_pipeline.py --dataset "D:/..." --patient PAT01
python step1_burn_segmented_into_3d_pipeline.py --scandir "D:/.../PAT01/D00/PAT01_D00_A"
```

⚠️ **`no_boundary_step1_marked_burn_3d_pipeline.py` also exists in this
folder and looks like an earlier iteration** — it writes to a separate
`--output` folder instead of in-place, uses a fixed `Rx=-90°, Ry=+90°`
rotation guess instead of folder 3's real alignment, and doesn't produce
`_burn_polygons.json` at all. It doesn't match this README or `config.yaml`.
Worth confirming whether it's safe to delete or archive — it's easy to
accidentally run the wrong one since the filenames are similar.

⚠️ Also worth knowing: this folder's own rough per-vertex `polygon_3d`
projection (baked into `_burn_polygons.json`) is a *different, coarser*
calculation than the full-mask/real-scanner-point area calculation folder 6
does later from the same file's 2D `polygon` field. They're not
contradictory, just two different "3D-ified" views of the same polygon
living in the same JSON — worth knowing so it doesn't look like duplicated
or conflicting work.

---

## 8. Folder 6 — Region measurement, features & progress tracking

Three sequential scripts, each optionally run in "all patients" / "one
patient" / "one scan" modes.

### Step 1 — `step1_compute_region_measurements.py`

Does **not** compute alignment — reads the one folder 3 already made.
Turns each burn region's 2D pixel polygon into a real physical measurement:
rasterizes the polygon into a mask, finds which range-file 3D points fall
inside it, and computes area in real mm² via the cylindrical surface-patch
formula (`Σ radius × dθ × dz` — exact for this scanner geometry, works even
on unaligned scans). If alignment exists for this scan, also computes an
aligned 3D centroid.

| | |
|---|---|
| **Reads** | `<scan>_burn_polygons.json` (folder 5), `<scan>_alignment.json` (folder 3, optional), `<scan>.tif` + `<scan>` (range file) |
| **Writes** | `<scan>_burn_polygons_aligned.json` (same structure + `area_mm2`, `n_points_3d`, `aligned_centroid_xyz`, `alignment_status`, `total_burn_area_mm2`) |
| **CLI** | `--dataset` (all patients), `--dataset --patient` (one patient), `--scandir` (one scan) |

```powershell
python step1_compute_region_measurements.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
```

### Step 2 — `step2_extract_features.py`

Adds richer per-scan geometry on top of step 1's polygons: perimeter,
compactness (`4πA/P²` — 1.0 = perfect circle, lower = more irregular),
bounding box, and scan-level confidence rollups (area-weighted mean,
global min/max, a count of "low confidence" regions sitting close to the
segmentation threshold — a QC flag, not a tracking decision).

| | |
|---|---|
| **Reads** | `<scan>_burn_polygons.json` |
| **Writes** | `<scan>_scan_features.json` |
| **CLI** | `--dataset` (all patients), `--dataset --patient`, `--polygons <path>` (one file) |

### Step 3 — `step3_track_progress.py`

The actual over-time tracking. Per (patient, variant), matches each burn
region across consecutive scans — first by IoU of rasterized polygon masks,
then by centroid distance as a fallback for anything IoU couldn't match
(using a dimension-safe comparison: 3D mm only when *both* scans being
compared have alignment, otherwise both fall back to 2D pixel centroids, so
patients with partial alignment coverage don't crash the matcher). Anything
that appears with no match is a new track; anything that disappears is
logged as a gap, not silently dropped.

| | |
|---|---|
| **Reads** | `<scan>_burn_polygons_aligned.json` (preferred) or `<scan>_burn_polygons.json` (fallback) |
| **Writes** | `tracking/<patient>_<variant>_tracking.json` — `tracks` (per-region time series: area, % change from baseline/previous, confidence, compactness) + `match_log` (every pairwise match decision + score, for manual QC) |
| **CLI** | `--dataset` (all patients), `--dataset --patient` (one patient, all variants), `--patient --variant` (one variant), `--variant` alone (that variant, across all patients) |

```powershell
python step3_track_progress.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
```

**`<patient>_<variant>_tracking.json` is the key output of the whole
pipeline up to this point** — it's what folder 9 reads to build plots and
reports.

---

## 9. Folder 7 — Color normalization (exploratory, not finalized)

You flagged this folder as still being explored — treat everything below as
a snapshot of where it's at, not a finished/wired-in stage.

**Goal:** remove lighting/camera color drift across timepoints so that any
remaining color difference in the burn region reflects real tissue healing,
not a different day's lighting.

| Script | Role |
|---|---|
| `select_skin_reference.py` | **Interactive, run once per patient.** Opens the D00 TIF, you click to draw a polygon over clean normal skin (no burn/hair/shadow). Saves that region's LAB color stats as the patient's reference. |
| `normalize_color.py` | Batch-applies Reinhard LAB color transfer: shifts/scales every later scan's *detected skin pixels* (auto-detected via YCrCb skin-color thresholds, burn region excluded from the *statistics* but still corrected) toward the reference's L*/a*/b* stats. Per variant: `D00_A` reference → all `A` scans, etc. |
| `explore_color_norm.py` | Standalone research script (explicitly *not* part of the main pipeline per its own docstring) — tests and visually compares 8 different normalization methods (Reinhard LAB variants, histogram matching, linear RGB scaling, Retinex+Reinhard) against each other, to help decide which method (if any) is worth wiring into production. |

| | |
|---|---|
| **`select_skin_reference.py` writes** | `<patient>/color_normalization/<patient>_skin_reference.json`, `..._skin_reference.png` |
| **`normalize_color.py` reads** | the reference JSON above, plus each scan's `.tif` and (if present) `<scan>_burn_polygons.json` (to exclude burn from the *statistics*, though correction is still applied *to* burn pixels) |
| **`normalize_color.py` writes** | `<scan>_normalized.tif` (in-place, meant as a drop-in replacement for the original TIF), `<patient>/color_normalization/<patient>_<variant>_norm_report.txt` |

```powershell
python select_skin_reference.py --dataset "D:/..." --patient PAT01
python normalize_color.py --dataset "D:/..." --patient PAT01
```

⚠️ **Not yet wired into the rest of the pipeline.** `normalize_color.py`'s
own report says its output should be used "instead of the original TIF" for
segmentation — but folder 5's actual script reads `<scan>.tif` directly, not
`<scan>_normalized.tif`. So right now this folder produces a file that
nothing downstream consumes yet. There's also a soft circular dependency
worth knowing about: `normalize_color.py` uses `<scan>_burn_polygons.json`
(folder 5's output) to exclude the burn from its skin-color statistics — so
it currently has to run *after* folder 5, even though its stated purpose is
to feed a corrected image *back into* folder 5. Worth deciding deliberately
once this folder is finalized (e.g., a two-pass approach: segment once on
raw TIF to get a rough burn mask, normalize, then re-segment on the
corrected TIF).

---

## 10. Folder 8 — Live web app

Not detailed here yet — you mentioned this is a live/demo web version of the
pipeline, still being improved. Add its scripts here once it's ready for
documentation the same way the others are.

---

## 11. Folder 9 — Progress analysis & reporting

**Scripts:** `analysis_common.py` (shared helpers), `step1_generate_plots.py`,
`step2_generate_patient_report.py`, `step3_generate_overall_report.py`,
`run_all.py`

Pure visualization/reporting layer — reads folder 6's outputs, doesn't
recompute or re-measure anything.

| | |
|---|---|
| **Reads** | `<scan>_burn_polygons_aligned.json` (preferred) / `_burn_polygons.json` (fallback) / `_scan_features.json` (last-resort fallback) for total area; `tracking/<patient>_<variant>_tracking.json` for per-region trajectories |
| **Writes** | `<dataset>/<patient>/analysis/plots/*.png`, `<dataset>/<patient>/analysis/<patient>_progress_report.html` (self-contained, images inlined as base64), `<dataset>/overall_progress_report.html` (all-patients view, sits at the dataset root) |

```powershell
python run_all.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python run_all.py --dataset "D:/..." --patient PAT01   # skips the overall report
```

Produces: a total-burn-area-over-time chart per patient/variant (the
headline "is it healing" view, sourced from folder 6 step 1's numbers so it
doesn't depend on region-matching having gone perfectly), per-region
trajectory charts (from folder 6 step 3's tracking, more granular), a
per-patient HTML summary, and a cross-patient normalized-healing-rate
comparison + bar chart in the overall report.

Known caveats it flags inline rather than silently smoothing over: mixed
mm²/pixel units within one patient's series (partial alignment coverage),
and that different camera-angle variants are treated as independent series
(not summed together, since they may capture overlapping burn area).

---

## 12. Everything a fully-processed scan folder contains

After running folders 1 → 3 → 5 → 6 (folder 4 is offline training, folder 7
is optional/experimental, folder 9 writes to `<patient>/analysis/` not the
scan folder itself), one scan folder looks like:

```
PAT01/D00/PAT01_D00_A/
  PAT01_D00_A                          <- raw range file (original data)
  PAT01_D00_A.tif                      <- raw texture (original data)
  PAT01_D00_A.lnd                      <- (if present — not used by these scripts directly)
  PAT01_D00_A.ply                      <- folder 1
  landmarks/
    PAT01_D00_A_landmarks.json         <- folder 2
    PAT01_D00_A_landmarks_debug.png    <- folder 2, --debug only
  PAT01_D00_A_aligned.ply              <- folder 3
  PAT01_D00_A_alignment.json           <- folder 3
  PAT01_D00_A_aligned_preview.png      <- folder 3, --debug only
  PAT01_D00_A_seg.tif                  <- folder 5
  PAT01_D00_A_burn3d.ply               <- folder 5
  PAT01_D00_A_burn_polygons.json       <- folder 5
  PAT01_D00_A_burn_mask.png            <- folder 5, --save-mask only
  PAT01_D00_A_burn_polygons_aligned.json <- folder 6 step 1
  PAT01_D00_A_scan_features.json       <- folder 6 step 2
  PAT01_D00_A_normalized.tif           <- folder 7, if run (not yet consumed downstream)
```

Plus, one level up at the patient folder:
```
PAT01/
  tracking/PAT01_A_tracking.json, PAT01_B_tracking.json, ...   <- folder 6 step 3
  analysis/PAT01_progress_report.html, analysis/plots/*.png    <- folder 9
  color_normalization/...                                       <- folder 7, if run
```

And at the dataset root:
```
face_burn_dataset/
  overall_progress_report.html          <- folder 9
  analysis/plots/overall_*.png          <- folder 9
  alignment_qc/<patient>_<variant>_overlay.png   <- folder 3's visualize_alignment.py
```

---

## 13. Reproducing everything from scratch — run order

```powershell
# 1. Raw scans -> 3D point clouds
python 1.<folder>/step1_convert_all_to_3d.py --dataset "D:/NahidW/Dataset/face_burn_dataset"

# 2. Facial landmarks (2D + 3D)
python 2.<folder>/step2_detect_landmarks.py --dataset "D:/NahidW/Dataset/face_burn_dataset"

# 3. Align every timepoint into one coordinate frame per patient/variant
python 3.face_alignment/step3_align_landmarks.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python 3.face_alignment/visualize_alignment.py --dataset "D:/NahidW/Dataset/face_burn_dataset"   # QC — check overlays before trusting alignment

# 4. (One-time / as-needed) train or update segmentation models
python 4.face_burn_segmentation_models/method1_sam2/train.py   # or whichever method you're using
python 4.face_burn_segmentation_models/evaluate_all.py         # compare methods
python 4.face_burn_segmentation_models/tune_threshold.py       # pick best threshold

# 5. Segment burns + wrap to 3D (edit config.yaml first: model + checkpoints + thresholds)
python 5.burn_segmentation_3d_pipeline/step1_burn_segmented_into_3d_pipeline.py --dataset "D:/NahidW/Dataset/face_burn_dataset"

# 6. Measure regions, extract features, track over time
python 6.face_scar_tracking/step1_compute_region_measurements.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python 6.face_scar_tracking/step2_extract_features.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
python 6.face_scar_tracking/step3_track_progress.py --dataset "D:/NahidW/Dataset/face_burn_dataset"

# 7. (Optional / experimental) color normalization — not yet consumed by step 5
# python 7.<folder>/select_skin_reference.py --dataset "D:/..." --patient PAT01
# python 7.<folder>/normalize_color.py --dataset "D:/..." --patient PAT01

# 9. Generate plots + reports
python 9.progress_analysis_reporting/run_all.py --dataset "D:/NahidW/Dataset/face_burn_dataset"
```

Every script above supports re-running safely — outputs are overwritten
(or skipped with a `↷` if already done and `--overwrite` wasn't passed),
nothing is appended to.

---

## 14. Open items / things worth cleaning up

Flagging these here rather than silently working around them, since a new
person hitting them cold would reasonably assume they're bugs:

1. **Folder 5 has two segmentation pipeline scripts** — `step1_burn_segmented_into_3d_pipeline.py` (current, matches `config.yaml`/README) and `no_boundary_step1_marked_burn_3d_pipeline.py` (looks like an earlier iteration, different output structure, no real alignment, no polygon JSON). Confirm the second one is safe to archive.
2. **Folder 5's own `README.md` still shows example paths under `5.face_burn_segmentation/checkpoints/...`**, but `config.yaml` (and the actual script) point at `4.face_burn_segmentation_models/checkpoints/...`. Looks like the README wasn't updated after a folder renumbering.
3. **`explore_color_norm.py`'s docstring says `5.color_norm_research/`** as its home folder, but it's currently in folder 7. Same renumbering-drift issue as #2.
4. **Folder 7's `normalize_color.py` output (`_normalized.tif`) isn't read by folder 5 yet** — the color-correction step exists but isn't wired into the segmentation input. Also has a soft circular dependency (needs folder 5's burn polygon to run well, but is meant to improve folder 5's input) — worth a deliberate two-pass design once finalized.
5. **`step1_convert_all_to_3d.py`'s internal docstring calls itself `convert_to_3d_v2.py`** — cosmetic only, but confusing if someone greps for the wrong filename.
6. **`test_model_interactive.py` (folder 4)** hasn't been reviewed here in detail — only `test_model.py`'s content was available. Worth a quick check whether it does something meaningfully different or is just a UI variant.
