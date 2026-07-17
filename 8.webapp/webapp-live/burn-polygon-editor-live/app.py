"""
Burn Polygon Editor — live version, backed by Google Drive
=============================================================

Same picker (Patients -> Timepoints -> Scans -> Editor) and same editor UI
as the local tool, but reads/writes scans and polygon JSON from a Google
Drive folder instead of a local disk path, so it can run as a normal web
service instead of `python run_with_picker.py --dataset ...` on your PC.

REQUIRED ENVIRONMENT VARIABLES (set these on your host, not in code):
    GOOGLE_CREDENTIALS_JSON   service account key JSON, as one string
    DRIVE_ROOT_FOLDER_ID      Drive folder ID of your dataset root

See README_DEPLOY.md for the full step-by-step setup.

Local dev run:
    export GOOGLE_CREDENTIALS_JSON="$(cat service-account.json)"
    export DRIVE_ROOT_FOLDER_ID="1AbCxyz..."
    python app.py
"""

import io
import json
import mimetypes
import os

from flask import Flask, jsonify, render_template, request, send_file

import drive_storage as ds

app = Flask(__name__, static_folder="static", template_folder="templates")

ROOT_FOLDER_ID = os.environ.get("DRIVE_ROOT_FOLDER_ID")


def _require_root_id():
    if not ROOT_FOLDER_ID:
        raise RuntimeError("DRIVE_ROOT_FOLDER_ID environment variable is not set")
    return ROOT_FOLDER_ID


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/")
def patients_page():
    tree = ds.get_tree(_require_root_id())
    patients = ds.discover_patients(tree)
    return render_template("patients.html", patients=patients)


@app.route("/api/patient/<patient>/scans")
def api_patient_scans(patient):
    tree = ds.get_tree(_require_root_id())
    if patient not in tree:
        return jsonify({"error": "patient not found"}), 404
    return jsonify(ds.all_scans_for_patient(tree, patient))


@app.route("/api/thumbnail/<scan_id>")
def api_thumbnail(scan_id):
    from PIL import Image

    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc

    raw = ds.download_file_bytes(scan["tif"]["id"])
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((220, 220))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/patient/<patient>")
def timepoints_page(patient):
    tree = ds.get_tree(_require_root_id())
    timepoints = ds.discover_timepoints(tree, patient)
    return render_template("timepoints.html", patient=patient, timepoints=timepoints)


@app.route("/patient/<patient>/<timepoint>")
def scans_page(patient, timepoint):
    tree = ds.get_tree(_require_root_id())
    scans = ds.discover_scans_in_timepoint(tree, patient, timepoint)
    return render_template("scans.html", patient=patient, timepoint=timepoint, scans=scans)


@app.route("/edit/<scan_id>")
def edit(scan_id):
    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    back_url = f"/?patient={loc[0]}" if loc else "/"
    return render_template(
        "editor.html",
        scan_id=scan_id,
        static_prefix="/static",
        image_url=f"/api/image/{scan_id}",
        polygons_url=f"/api/polygons/{scan_id}",
        polygons_save_url=f"/api/polygons/{scan_id}",
        back_url=back_url,
    )


@app.route("/api/image/<scan_id>")
def api_image(scan_id):
    from PIL import Image

    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc

    raw = ds.download_file_bytes(scan["tif"]["id"])
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/scan_files/<scan_id>")
def api_scan_files(scan_id):
    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc
    return jsonify(ds.classify_scan_files(scan))


def _find_file_in_scan(scan: dict, file_id: str):
    """Only serve files that actually belong to this scan's folder listing —
    never an arbitrary Drive file id passed in the URL."""
    for f in scan.get("files", []):
        if f["id"] == file_id:
            return f
    return None


@app.route("/api/raw/<scan_id>/<file_id>")
def api_raw_file(scan_id, file_id):
    """Serve a non-primary scan file as-is: png/jpg display natively,
    json/other files download or open depending on the browser."""
    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc
    f = _find_file_in_scan(scan, file_id)
    if f is None:
        return jsonify({"error": "file not found on this scan"}), 404

    raw = ds.download_file_bytes(file_id)
    mime, _ = mimetypes.guess_type(f["name"])
    return send_file(io.BytesIO(raw), mimetype=mime or "application/octet-stream", download_name=f["name"])


@app.route("/api/preview_tif/<scan_id>/<file_id>")
def api_preview_tif(scan_id, file_id):
    """Convert a secondary .tif (e.g. the *_seg.tif overlay) to PNG for
    inline display, same as the main scan image conversion."""
    from PIL import Image

    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc
    f = _find_file_in_scan(scan, file_id)
    if f is None:
        return jsonify({"error": "file not found on this scan"}), 404

    raw = ds.download_file_bytes(file_id)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/preview_json/<scan_id>/<file_id>")
def api_preview_json(scan_id, file_id):
    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc
    f = _find_file_in_scan(scan, file_id)
    if f is None:
        return jsonify({"error": "file not found on this scan"}), 404

    raw = ds.download_file_bytes(file_id)
    try:
        return jsonify(json.loads(raw))
    except Exception:
        return raw.decode("utf-8", errors="replace"), 200, {"Content-Type": "text/plain"}



def api_get_polygons(scan_id):
    from PIL import Image

    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc

    raw = ds.download_file_bytes(scan["tif"]["id"])
    with Image.open(io.BytesIO(raw)) as im:
        size = im.size
    return jsonify(ds.load_polygons(scan, scan_id, size))


@app.route("/api/polygons/<scan_id>", methods=["POST"])
def api_save_polygons(scan_id):
    tree = ds.get_tree(_require_root_id())
    loc = ds.find_scan(tree, scan_id)
    if loc is None:
        return jsonify({"error": "scan not found"}), 404
    _, _, scan = loc

    payload = request.get_json(force=True)
    ds.save_polygons(scan, scan_id, payload)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local dev only. In production, gunicorn serves this via the Procfile.
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
