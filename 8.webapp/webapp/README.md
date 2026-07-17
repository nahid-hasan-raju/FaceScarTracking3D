# Burn Region Polygon Editor

A browser-based tool to review and hand-correct SAM2's predicted burn
region polygons on Day-0 scans, or draw new ones from scratch. Built for
your workflow: correct Day-0 (A/B/C/D) only, then those polygons become
the reference region tracked across later timepoints.

## 0. Easiest way to run it (recommended): `start_editor.bat`

1. Open `start_editor.bat` in a text editor (right-click → Edit, not double-click).
2. Edit these two lines near the top to match your machine:
   ```
   SET VENV_PATH=D:\NahidW\Coding\seg_env
   SET DATASET_PATH=D:\NahidW\Dataset\face_burn_dataset
   ```
3. Save it. From now on, just **double-click `start_editor.bat`**.

It activates your environment, installs Flask/Pillow if missing, starts
the server, and opens your browser automatically to the **Patients** page.
Navigate: **Patients → Timepoints → Scans → Editor**. Each level has a
search box, and the editor/timepoints/scans pages have breadcrumbs
(top-left) to go back up. Close the black console window to stop the server.

This runs `run_with_picker.py` under the hood — see section 1 below if
you'd rather run it manually from a terminal, or section 1b for the
single-scan version without any browsing.

## 1. Running manually (patient browser + editor, one server)

```bash
pip install flask pillow
python run_with_picker.py --dataset "D:\NahidW\Dataset\face_burn_dataset"
```

Opens `http://127.0.0.1:5050/` — lists every patient found under your
dataset root. Click a patient → see its timepoints (Day-0 tagged and
listed first) → click a timepoint → see its scans (A/B/C/D...) → click
a scan to open the editor.

## 1b. Single-scan standalone tool (no browsing)

Point it at one scan folder at a time — no "back to list" link, since
there's only the one scan.

```bash
pip install flask pillow
python app.py --scandir "D:\NahidW\Dataset\face_burn_dataset\PAT01\D00\PAT01_D00_A"
```

Then open the URL it prints (default `http://127.0.0.1:5050`).

It auto-detects:
- the scan's `.tif` image (ignores any `*_seg.tif`)
- an existing `PAT01_D00_A_burn_polygons.json` from SAM2, preloaded as the starting polygon(s)

On save, if the existing json was pure SAM2 output, a one-time backup is
written to `PAT01_D00_A_burn_polygons.sam2_backup.json` before overwriting.

## 2. Editor controls

| Action | How |
|---|---|
| Start a new polygon | Click **+ New Polygon**, then click on the image to place points |
| Close the polygon | Double-click, or press **Enter** |
| Cancel current drawing | **Esc** |
| Move a point | Click-drag it |
| Delete a point | Right-click it |
| Delete a whole region | Select it (click in sidebar or on canvas), press **Delete** |
| Undo | **Ctrl+Z** |
| Zoom | Scroll wheel |
| Pan | Alt+drag, or middle-mouse drag |
| Rename a region | Edit the text field in the left sidebar |
| Save | **Save** button (top right) |

Each region tracks a `source`: `sam2` (untouched model output), `manual`
(drawn from scratch), or you can treat any edited SAM2 region as
`manual_edit` if you want that distinction — currently the tool keeps
edited SAM2 regions labeled `sam2` unless you delete/redraw them; say the
word if you'd rather it auto-flip to `manual_edit` on first drag.

## 3. Integrating into your existing webapp

See `webapp_integration/polygon_editor_blueprint.py` — it's a Flask
Blueprint, not a separate server, so it plugs into your existing
`webapp/app.py`:

1. Copy `webapp_integration/polygon_editor_blueprint.py` next to your
   existing `webapp/app.py`.
2. Copy `templates/editor.html`, `static/editor.css`, `static/editor.js`
   into your webapp's `templates/` / `static/` folders, and copy
   `webapp_integration/picker.html` into `templates/` too.
3. In your existing `webapp/app.py`:

   ```python
   from pathlib import Path
   from polygon_editor_blueprint import make_polygon_editor_blueprint

   polygon_bp = make_polygon_editor_blueprint(dataset_root=Path(args.dataset))
   app.register_blueprint(polygon_bp, url_prefix="/polygon-editor")
   ```

4. Visit `/polygon-editor/` — it lists every patient's Day-0 (A/B/C/D)
   scans it finds under your dataset root, with a link to open each one
   in the editor.

The blueprint assumes your Day-0 folders are named one of `D00`, `D0`,
`DAY0`, `Day0` (case-insensitive) — edit `DAY0_NAMES` at the top of
`polygon_editor_blueprint.py` if yours differs.

## 4. JSON schema

Both the standalone tool and the blueprint read/write:

```json
{
  "scan_id": "PAT01_D00_A",
  "image_size": [480, 457],
  "regions": [
    {
      "id": 1,
      "label": "region_1",
      "source": "manual",
      "confidence": null,
      "polygon": [[123.4, 88.1], [130.0, 90.2], ...]
    }
  ]
}
```

**This is a guessed schema** based on what your pipeline logs ("Polygons
saved... (N region(s))") — I didn't have your actual
`*_burn_polygons.json` to match exactly. Both `load_polygons_file()` in
`app.py` and its twin in `polygon_editor_blueprint.py` already tolerate
some variation (`regions` or `polygons` as the top key, `polygon`/
`points`/`coords` for the point list) — but if your real file uses
different field names or nesting, send me one example file and I'll
adjust those two functions (they're the only place schema translation
happens; the rest of the tool is schema-agnostic).

## 5. Notes / assumptions

- Coordinates are stored in raw image pixel space (same space SAM2's
  polygons already use), so no reprojection step is needed downstream.
- Only Day-0 scans are meant to be edited here, per your plan — the
  picker page filters to Day-0 folders only, but the standalone tool
  will happily open any scan folder if you ever want to spot-check a
  later timepoint too.
- Multiple regions per scan are fully supported (multiple burn areas).
