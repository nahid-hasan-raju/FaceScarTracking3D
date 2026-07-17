"""
Polygon Editor — integration blueprint for the existing burn-tracker webapp
=============================================================================

HOW TO WIRE THIS INTO YOUR EXISTING webapp/app.py
--------------------------------------------------
1. Copy this folder's `polygon_editor_blueprint.py` into your existing
   webapp's project (e.g. next to webapp/app.py).
2. Copy the shared `templates/editor.html` and `static/editor.css` /
   `static/editor.js` files (from the standalone tool) into your existing
   webapp's `templates/` and `static/` folders. Also copy `picker.html`
   (in this same folder) into your webapp's `templates/`.
3. In your existing webapp/app.py, near where you create the Flask app
   and read --dataset from argparse, add:

       from polygon_editor_blueprint import make_polygon_editor_blueprint

       polygon_bp = make_polygon_editor_blueprint(dataset_root=Path(args.dataset))
       app.register_blueprint(polygon_bp, url_prefix="/polygon-editor")

4. Add a link/button somewhere in your existing UI pointing to:
       /polygon-editor/
   which lists every patient's Day-0 A/B/C/D scans to pick from.

This blueprint only touches routes under /polygon-editor/*, so it won't
collide with your existing routes.
"""

import io
import json
import shutil
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file

DAY0_NAMES = {"D00", "D0", "DAY0", "Day0"}  # adjust if your Day-0 folder is named differently


def _is_day0(timepoint_name: str) -> bool:
    return timepoint_name.upper().replace("-", "") in {n.upper() for n in DAY0_NAMES}


def discover_day0_scans(dataset_root: Path):
    """
    Walk dataset_root / <patient> / <timepoint> / <scan> and return every
    Day-0 scan folder found, across all patients and A/B/C/D versions.
    Adjust the glob depth here if your folder layout differs.
    """
    scans = []
    if not dataset_root.exists():
        return scans
    for patient_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for timepoint_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            if not _is_day0(timepoint_dir.name):
                continue
            for scan_dir in sorted(p for p in timepoint_dir.iterdir() if p.is_dir()):
                tifs = [t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()]
                if not tifs:
                    continue
                json_path = scan_dir / f"{tifs[0].stem}_burn_polygons.json"
                scans.append({
                    "patient": patient_dir.name,
                    "timepoint": timepoint_dir.name,
                    "scan_id": tifs[0].stem,
                    "scandir": str(scan_dir),
                    "has_polygons": json_path.exists(),
                })
    return scans


def _find_scan_dir(dataset_root: Path, scan_id: str):
    """Locate a scan's folder by scan_id by searching the dataset tree."""
    for patient_dir in dataset_root.iterdir():
        if not patient_dir.is_dir():
            continue
        for timepoint_dir in patient_dir.iterdir():
            if not timepoint_dir.is_dir():
                continue
            candidate = timepoint_dir / scan_id
            if candidate.is_dir():
                return candidate
    return None


def load_polygons_file(json_path: Path, scan_id: str, image_size):
    if not json_path.exists():
        return {"scan_id": scan_id, "image_size": list(image_size), "regions": []}
    with open(json_path, "r") as f:
        raw = json.load(f)
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
        "scan_id": raw.get("scan_id", scan_id),
        "image_size": raw.get("image_size", list(image_size)),
        "regions": regions,
    }


def save_polygons_file(json_path: Path, payload: dict):
    if json_path.exists():
        backup_path = json_path.with_name(json_path.stem + ".sam2_backup.json")
        if not backup_path.exists():
            try:
                with open(json_path) as f:
                    existing = json.load(f)
                if all(r.get("source", "sam2") == "sam2" for r in existing.get("regions", existing.get("polygons", []))):
                    shutil.copy(json_path, backup_path)
            except Exception:
                pass
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def make_polygon_editor_blueprint(dataset_root: Path) -> Blueprint:
    bp = Blueprint(
        "polygon_editor",
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/polygon-editor-static",
    )

    @bp.route("/")
    def picker():
        scans = discover_day0_scans(dataset_root)
        return render_template("picker.html", scans=scans)

    @bp.route("/edit/<scan_id>")
    def edit(scan_id):
        return render_template(
            "editor.html",
            scan_id=scan_id,
            static_prefix="/polygon-editor-static",
            image_url=f"/polygon-editor/api/image/{scan_id}",
            polygons_url=f"/polygon-editor/api/polygons/{scan_id}",
            polygons_save_url=f"/polygon-editor/api/polygons/{scan_id}",
        )

    @bp.route("/api/image/<scan_id>")
    def api_image(scan_id):
        from PIL import Image

        scan_dir = _find_scan_dir(dataset_root, scan_id)
        if scan_dir is None:
            return jsonify({"error": "scan not found"}), 404
        tif_path = next((t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()), None)
        if tif_path is None:
            return jsonify({"error": "no tif found"}), 404

        im = Image.open(tif_path).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @bp.route("/api/polygons/<scan_id>", methods=["GET"])
    def api_get_polygons(scan_id):
        from PIL import Image

        scan_dir = _find_scan_dir(dataset_root, scan_id)
        if scan_dir is None:
            return jsonify({"error": "scan not found"}), 404
        tif_path = next((t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()), None)
        json_path = scan_dir / f"{scan_id}_burn_polygons.json"
        with Image.open(tif_path) as im:
            size = im.size
        return jsonify(load_polygons_file(json_path, scan_id, size))

    @bp.route("/api/polygons/<scan_id>", methods=["POST"])
    def api_save_polygons(scan_id):
        scan_dir = _find_scan_dir(dataset_root, scan_id)
        if scan_dir is None:
            return jsonify({"error": "scan not found"}), 404
        json_path = scan_dir / f"{scan_id}_burn_polygons.json"
        payload = request.get_json(force=True)
        save_polygons_file(json_path, payload)
        return jsonify({"status": "ok"})

    return bp
