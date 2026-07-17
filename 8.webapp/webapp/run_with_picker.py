"""
Burn Polygon Editor — combined picker + editor, one server
=============================================================

The easiest way to run this: use start_editor.bat (double-click it),
which calls this script for you. See that file's comments if you need
to change your dataset path or Python environment.

Manual usage:
    python run_with_picker.py --dataset "D:\\NahidW\\Dataset\\face_burn_dataset"
    python run_with_picker.py --dataset "D:\\...\\face_burn_dataset" --port 5050

Opens:
    http://127.0.0.1:5050/            -> picker: every patient's Day-0 (A/B/C/D) scans
    http://127.0.0.1:5050/edit/<id>    -> editor for one scan, with a link back to the picker

SCHEMA / FOLDER LAYOUT ASSUMPTIONS
-----------------------------------
Expects: <dataset_root>/<patient>/<timepoint>/<scan_id>/<scan_id>.tif
Day-0 folder name must be one of: D00, D0, DAY0, Day0 (case-insensitive) --
edit DAY0_NAMES below if yours differs.

Polygon JSON schema read/written (adjust load/save functions if yours differs):
    {
      "scan_id": "PAT01_D00_A",
      "image_size": [W, H],
      "regions": [
        {"id": 1, "label": "region_1", "source": "sam2"|"manual"|"manual_edit",
         "confidence": 0.92, "polygon": [[x,y], [x,y], ...]}
      ]
    }
"""

import argparse
import io
import json
import shutil
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__, static_folder="static", template_folder="templates")

DATASET_ROOT: Path = None
DAY0_NAMES = {"D00", "D0", "DAY0", "Day0"}


def _is_day0(name: str) -> bool:
    return name.upper().replace("-", "") in {n.upper() for n in DAY0_NAMES}


def discover_patients(dataset_root: Path):
    """Level 1: one row per patient, with rollup counts."""
    patients = []
    if not dataset_root.exists():
        return patients
    for patient_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        timepoint_dirs = sorted(p for p in patient_dir.iterdir() if p.is_dir())
        scan_count = 0
        day0_count = 0
        polygons_saved = 0
        for tp_dir in timepoint_dirs:
            for scan_dir in tp_dir.iterdir():
                if not scan_dir.is_dir():
                    continue
                tifs = [t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()]
                if not tifs:
                    continue
                scan_count += 1
                if _is_day0(tp_dir.name):
                    day0_count += 1
                if (scan_dir / f"{tifs[0].stem}_burn_polygons.json").exists():
                    polygons_saved += 1
        patients.append({
            "patient": patient_dir.name,
            "timepoint_count": len(timepoint_dirs),
            "scan_count": scan_count,
            "day0_count": day0_count,
            "polygons_saved": polygons_saved,
        })
    return patients


def discover_timepoints(dataset_root: Path, patient: str):
    """Level 2: one row per timepoint folder for a given patient."""
    patient_dir = dataset_root / patient
    timepoints = []
    if not patient_dir.exists():
        return timepoints
    for tp_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        scan_dirs = []
        for scan_dir in sorted(tp_dir.iterdir()):
            if not scan_dir.is_dir():
                continue
            if any(t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()):
                scan_dirs.append(scan_dir)
        timepoints.append({
            "timepoint": tp_dir.name,
            "is_day0": _is_day0(tp_dir.name),
            "scan_count": len(scan_dirs),
        })
    # Day-0 first, then alphabetical/natural order for the rest
    timepoints.sort(key=lambda t: (not t["is_day0"], t["timepoint"]))
    return timepoints


def discover_scans_in_timepoint(dataset_root: Path, patient: str, timepoint: str):
    """Level 3: one row per scan (A/B/C/D...) inside a patient+timepoint."""
    tp_dir = dataset_root / patient / timepoint
    scans = []
    if not tp_dir.exists():
        return scans
    for scan_dir in sorted(p for p in tp_dir.iterdir() if p.is_dir()):
        tifs = [t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()]
        if not tifs:
            continue
        json_path = scan_dir / f"{tifs[0].stem}_burn_polygons.json"
        scans.append({
            "scan_id": tifs[0].stem,
            "has_polygons": json_path.exists(),
        })
    return scans


def _find_scan_dir(scan_id: str):
    for patient_dir in DATASET_ROOT.iterdir():
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


@app.route("/")
def patients_page():
    patients = discover_patients(DATASET_ROOT)
    return render_template("patients.html", patients=patients)


@app.route("/api/patient/<patient>/scans")
def api_patient_scans(patient):
    """All scans for one patient, across every timepoint, for the home-page grid."""
    patient_dir = DATASET_ROOT / patient
    if not patient_dir.exists():
        return jsonify({"error": "patient not found"}), 404
    out = []
    for tp_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        for scan_dir in sorted(p for p in tp_dir.iterdir() if p.is_dir()):
            tifs = [t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()]
            if not tifs:
                continue
            json_path = scan_dir / f"{tifs[0].stem}_burn_polygons.json"
            out.append({
                "timepoint": tp_dir.name,
                "is_day0": _is_day0(tp_dir.name),
                "scan_id": tifs[0].stem,
                "has_polygons": json_path.exists(),
            })
    out.sort(key=lambda s: (not s["is_day0"], s["timepoint"], s["scan_id"]))
    return jsonify(out)


@app.route("/api/thumbnail/<scan_id>")
def api_thumbnail(scan_id):
    from PIL import Image

    scan_dir = _find_scan_dir(scan_id)
    if scan_dir is None:
        return jsonify({"error": "scan not found"}), 404
    tif_path = next((t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()), None)
    if tif_path is None:
        return jsonify({"error": "no tif found"}), 404

    im = Image.open(tif_path).convert("RGB")
    im.thumbnail((220, 220))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/patient/<patient>")
def timepoints_page(patient):
    timepoints = discover_timepoints(DATASET_ROOT, patient)
    return render_template("timepoints.html", patient=patient, timepoints=timepoints)


@app.route("/patient/<patient>/<timepoint>")
def scans_page(patient, timepoint):
    scans = discover_scans_in_timepoint(DATASET_ROOT, patient, timepoint)
    return render_template("scans.html", patient=patient, timepoint=timepoint, scans=scans)


@app.route("/edit/<scan_id>")
def edit(scan_id):
    scan_dir = _find_scan_dir(scan_id)
    if scan_dir is not None:
        patient = scan_dir.parent.parent.name
        back_url = f"/?patient={patient}"
    else:
        back_url = "/"
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

    scan_dir = _find_scan_dir(scan_id)
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


@app.route("/api/polygons/<scan_id>", methods=["GET"])
def api_get_polygons(scan_id):
    from PIL import Image

    scan_dir = _find_scan_dir(scan_id)
    if scan_dir is None:
        return jsonify({"error": "scan not found"}), 404
    tif_path = next((t for t in scan_dir.glob("*.tif") if "_seg" not in t.stem.lower()), None)
    json_path = scan_dir / f"{scan_id}_burn_polygons.json"
    with Image.open(tif_path) as im:
        size = im.size
    return jsonify(load_polygons_file(json_path, scan_id, size))


@app.route("/api/polygons/<scan_id>", methods=["POST"])
def api_save_polygons(scan_id):
    scan_dir = _find_scan_dir(scan_id)
    if scan_dir is None:
        return jsonify({"error": "scan not found"}), 404
    json_path = scan_dir / f"{scan_id}_burn_polygons.json"
    payload = request.get_json(force=True)
    save_polygons_file(json_path, payload)
    return jsonify({"status": "ok"})


def main():
    global DATASET_ROOT

    parser = argparse.ArgumentParser(description="Burn polygon editor with picker page")
    parser.add_argument("--dataset", required=True, help="Path to dataset root, e.g. D:\\...\\face_burn_dataset")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    DATASET_ROOT = Path(args.dataset)
    if not DATASET_ROOT.exists():
        raise SystemExit(f"Dataset root not found: {DATASET_ROOT}")

    patients = discover_patients(DATASET_ROOT)
    total_scans = sum(p["scan_count"] for p in patients)
    total_day0 = sum(p["day0_count"] for p in patients)
    url = f"http://{args.host}:{args.port}/"

    print("=" * 60)
    print("  Burn Polygon Editor")
    print(f"  Dataset  : {DATASET_ROOT}")
    print(f"  Patients : {len(patients)}")
    print(f"  Scans    : {total_scans} total ({total_day0} Day-0)")
    print(f"  URL      : {url}")
    print("=" * 60)

    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()