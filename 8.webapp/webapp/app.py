"""
Standalone Burn Region Polygon Editor
======================================

Loads a single scan folder (a Day-0 scan, e.g. PAT01_D00_A) and lets you
review / correct SAM2's predicted burn polygons, or draw new ones from
scratch, in the browser. Saves back to a *_burn_polygons.json file next
to the .tif, matching the naming convention your pipeline already uses.

USAGE
-----
    python app.py --scandir "D:\\NahidW\\Dataset\\face_burn_dataset\\PAT01\\D00\\PAT01_D00_A"
    python app.py --scandir /path/to/PAT01_D00_A --port 5050

It auto-detects:
  - the scan's .tif image (skips any *_seg.tif segmentation overlay file)
  - an existing *_burn_polygons.json (SAM2 output) to preload as a starting point

If no _burn_polygons.json exists yet, the editor opens empty and you draw
from scratch. On save, a one-time backup of the original SAM2 json is
kept alongside as *_burn_polygons.sam2_backup.json.

SCHEMA
------
This tool reads/writes a JSON shape of:
    {
      "scan_id": "PAT01_D00_A",
      "image_size": [W, H],
      "regions": [
        {"id": 1, "label": "region_1", "source": "sam2"|"manual"|"manual_edit",
         "confidence": 0.92, "polygon": [[x,y], [x,y], ...]}
      ]
    }

If your existing pipeline's _burn_polygons.json uses different field
names, adjust `load_polygons_file()` / `save_polygons_file()` below --
those two functions are the only place schema translation happens.
"""

import argparse
import io
import json
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__, static_folder="static", template_folder="templates")

SCAN_DIR: Path = None
SCAN_ID: str = None
TIF_PATH: Path = None
JSON_PATH: Path = None


def find_scan_files(scandir: Path):
    """Locate the scan's .tif image and its polygons json inside scandir."""
    tif_candidates = sorted(
        p for p in scandir.glob("*.tif") if "_seg" not in p.stem.lower()
    )
    if not tif_candidates:
        raise FileNotFoundError(f"No .tif file found in {scandir}")
    tif_path = tif_candidates[0]

    scan_id = tif_path.stem
    json_path = scandir / f"{scan_id}_burn_polygons.json"
    return scan_id, tif_path, json_path


def load_polygons_file(json_path: Path, image_size):
    """Read existing polygons json (e.g. SAM2 output). Returns normalized dict.
    ADJUST HERE if your existing schema differs from the one documented above.
    """
    if not json_path.exists():
        return {"scan_id": SCAN_ID, "image_size": list(image_size), "regions": []}

    with open(json_path, "r") as f:
        raw = json.load(f)

    # Best-effort adaptation: accept either "regions" or "polygons" as the key,
    # and either "polygon"/"points"/"coords" for the point list.
    regions_raw = raw.get("regions") or raw.get("polygons") or []
    regions = []
    for i, r in enumerate(regions_raw):
        poly = r.get("polygon") or r.get("points") or r.get("coords") or []
        regions.append({
            "id": r.get("id", i + 1),
            "label": r.get("label", f"region_{i + 1}"),
            "source": r.get("source", "sam2"),
            "confidence": r.get("confidence"),
            "polygon": poly,
        })

    return {
        "scan_id": raw.get("scan_id", SCAN_ID),
        "image_size": raw.get("image_size", list(image_size)),
        "regions": regions,
    }


def save_polygons_file(json_path: Path, payload: dict):
    """Write edited polygons back to disk, backing up an original SAM2 file once."""
    if json_path.exists():
        backup_path = json_path.with_name(json_path.stem + ".sam2_backup.json")
        if not backup_path.exists():
            # Only back up if the existing file wasn't already manually edited
            try:
                with open(json_path) as f:
                    existing = json.load(f)
                if all(r.get("source", "sam2") == "sam2" for r in existing.get("regions", existing.get("polygons", []))):
                    shutil.copy(json_path, backup_path)
            except Exception:
                pass

    # Mark any region whose source was "sam2" but got touched as "manual_edit".
    # (The frontend already sets brand-new regions to "manual".)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


@app.route("/")
def index():
    return render_template(
        "editor.html",
        scan_id=SCAN_ID,
        static_prefix="/static",
        image_url="/api/image",
        polygons_url="/api/polygons",
        polygons_save_url="/api/polygons",
        back_url=None,
    )


@app.route("/api/image")
def api_image():
    from PIL import Image

    try:
        im = Image.open(TIF_PATH)
        im = im.convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read tif: {e}"}), 500

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/polygons", methods=["GET"])
def api_get_polygons():
    from PIL import Image

    with Image.open(TIF_PATH) as im:
        size = im.size  # (W, H)
    data = load_polygons_file(JSON_PATH, size)
    return jsonify(data)


@app.route("/api/polygons", methods=["POST"])
def api_save_polygons():
    payload = request.get_json(force=True)
    save_polygons_file(JSON_PATH, payload)
    return jsonify({"status": "ok"})


def main():
    global SCAN_DIR, SCAN_ID, TIF_PATH, JSON_PATH

    parser = argparse.ArgumentParser(description="Standalone burn polygon editor")
    parser.add_argument("--scandir", required=True, help="Path to a single scan folder, e.g. .../PAT01/D00/PAT01_D00_A")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    SCAN_DIR = Path(args.scandir)
    if not SCAN_DIR.exists():
        raise SystemExit(f"scandir not found: {SCAN_DIR}")

    SCAN_ID, TIF_PATH, JSON_PATH = find_scan_files(SCAN_DIR)

    print("=" * 60)
    print("  Burn Polygon Editor")
    print(f"  Scan     : {SCAN_ID}")
    print(f"  Image    : {TIF_PATH}")
    print(f"  Polygons : {JSON_PATH}{'  (new)' if not JSON_PATH.exists() else ''}")
    print(f"  URL      : http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
