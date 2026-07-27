"""
drive_storage.py
=================
Replaces the local-filesystem dataset walk (`Path.iterdir()` etc.) with an
equivalent walk over a Google Drive folder tree, using a service account.

Expected Drive layout (mirrors the old local layout exactly):

    <ROOT_FOLDER_ID>/
        <patient>/
            <timepoint>/
                <scan_id>/
                    <scan_id>.tif
                    <scan_id>_burn_polygons.json   (optional)

SETUP (see README_DEPLOY.md for the full walkthrough):
    1. Create a Google Cloud service account, enable the Drive API.
    2. Share your Drive dataset folder with the service account's email
       (Editor access) so it can read/write.
    3. Set two environment variables on the host:
         GOOGLE_CREDENTIALS_JSON  -> the full service account JSON key, as one string
         DRIVE_ROOT_FOLDER_ID     -> the Drive folder ID of the dataset root

Caching: walking the whole Drive tree is one API call per folder, which
gets slow on every request for a large dataset. The tree is cached in
memory for TREE_TTL_SECONDS and rebuilt lazily after that, or immediately
after a save (see invalidate()).
"""

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httplib2
from google.oauth2 import service_account
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
DAY0_NAMES = {"D00", "D0", "DAY0", "Day0"}
TREE_TTL_SECONDS = 300
MAX_WORKERS = 6  # concurrent Drive API calls per level during a tree walk
HTTP_TIMEOUT_SECONDS = 30  # fail fast instead of hanging on a stalled connection

_service = None
_CACHE = {"tree": None, "ts": 0}
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Auth / low-level Drive helpers
# ---------------------------------------------------------------------------

_credentials = None
_credentials_lock = threading.Lock()


def _get_credentials():
    """Build the service account credentials exactly once, no matter how
    many threads call this. Every thread's Drive service wraps this SAME
    credentials object, so they share its cached OAuth token instead of
    each thread fetching its own token from Google (which is slow, and
    under enough concurrent threads, Google will rate-limit repeated
    token requests -- that's what was causing the multi-minute "hangs
    then nothing happens" behavior)."""
    global _credentials
    if _credentials is not None:
        return _credentials
    with _credentials_lock:
        if _credentials is not None:
            return _credentials
        creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        creds_path = os.environ.get("GOOGLE_CREDENTIALS_FILE")
        if creds_json:
            info = json.loads(creds_json)
            _credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        elif creds_path:
            _credentials = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        else:
            raise RuntimeError(
                "No Google credentials found. Set GOOGLE_CREDENTIALS_JSON (the service "
                "account key as one JSON string) or GOOGLE_CREDENTIALS_FILE (a path)."
            )
        return _credentials


def get_service():
    """One Drive service (and one underlying HTTP transport) per thread --
    the transport itself isn't safe to share across concurrent requests --
    but all threads reuse the SAME credentials object above, so they share
    its cached token instead of each re-authenticating from scratch.

    The transport gets an explicit timeout: httplib2's default is to wait
    forever on a stalled connection, which is what made failures look like
    an infinite hang instead of a clean, fast error."""
    svc = getattr(_thread_local, "service", None)
    if svc is None:
        raw_http = httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS)
        authed_http = AuthorizedHttp(_get_credentials(), http=raw_http)
        svc = build("drive", "v3", http=authed_http, cache_discovery=False)
        _thread_local.service = svc
    return svc


def _list_children(folder_id):
    """Direct children of folder_id: list of {id, name, mimeType}."""
    service = get_service()
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def download_file_bytes(file_id: str) -> bytes:
    for attempt in range(2):
        service = get_service()
        req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        data = buf.getvalue()
        if data:
            return data
        # An empty result with no exception raised is unusual -- retry once
        # with a fresh transport before giving up, rather than silently
        # handing back 0 bytes (which just shows up later as a confusing
        # "cannot identify image file" error with no obvious cause).
        _thread_local.service = None
    raise RuntimeError(f"Drive returned 0 bytes for file_id={file_id!r} after retry")


def upload_json(file_id, parent_id, filename, data: dict):
    """Create a new json file (file_id=None) or overwrite an existing one."""
    service = get_service()
    content = json.dumps(data, indent=2).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/json", resumable=False)
    if file_id:
        return service.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
    meta = {"name": filename, "parents": [parent_id]}
    return service.files().create(
        body=meta, media_body=media, fields="id, name", supportsAllDrives=True
    ).execute()


def copy_file(file_id, new_name, parent_id):
    service = get_service()
    body = {"name": new_name, "parents": [parent_id]}
    return service.files().copy(fileId=file_id, body=body, supportsAllDrives=True).execute()


# ---------------------------------------------------------------------------
# Tree walk + cache
# ---------------------------------------------------------------------------

def _is_day0(name: str) -> bool:
    return name.upper().replace("-", "") in {n.upper() for n in DAY0_NAMES}


# Fixed naming convention observed across scan folders:
#   {scan_id}.tif             -> original
#   {scan_id}_normalized.tif  -> normalized
#   {scan_id}_seg.tif         -> seg (segmentation overlay)
# Listed in the order they should appear in any UI dropdown.
VARIANT_SUFFIXES = [
    ("original", ""),
    ("normalized", "_normalized"),
    ("seg", "_seg"),
]
VARIANT_ORDER = [name for name, _ in VARIANT_SUFFIXES]


def _scan_folder_to_entry(files, scan_folder):
    """scan_folder is the Drive folder dict ({id, name, mimeType}) for this
    scan -- its name IS the scan_id (e.g. 'PAT01_D14_A'), which is far more
    reliable than trying to derive scan_id from whichever tif happens to be
    picked. Each variant is matched by an EXACT filename, not a substring
    guess, so an original and a normalized tif sitting side by side can
    never be confused with each other."""
    scan_id = scan_folder["name"]
    scan_folder_id = scan_folder["id"]

    by_name = {f["name"].lower(): f for f in files}

    variants = {}
    for variant_name, suffix in VARIANT_SUFFIXES:
        for ext in (".tif", ".tiff"):
            target = f"{scan_id}{suffix}{ext}".lower()
            f = by_name.get(target)
            if f:
                variants[variant_name] = f
                break

    original = variants.get("original")
    if not original:
        # Fallback for scans that don't exactly match "{scan_id}.tif" --
        # better to show something than to silently drop the scan.
        original = next(
            (f for f in files
             if f["name"].lower().endswith((".tif", ".tiff"))
             and "_seg" not in f["name"].lower()
             and "_normalized" not in f["name"].lower()),
            None,
        )
        if original:
            variants["original"] = original
    if not original:
        return None

    json_file = next((f for f in files if f["name"] == f"{scan_id}_burn_polygons.json"), None)
    return scan_id, {
        "id": scan_folder_id, "tif": original, "json": json_file,
        "files": files, "variants": variants,
    }


def _build_tree(root_id: str) -> dict:
    """
    Walk root -> patients -> timepoints -> scans -> files level by level,
    using ONE bounded thread pool per level (never nested pools — nesting
    a pool inside a pool inside a pool can spawn hundreds of threads at
    once on a larger dataset and exhaust memory on a small instance).
    """
    tree = {}

    # Level 1: patients (single call, no pool needed)
    patient_folders = [c for c in _list_children(root_id) if c["mimeType"] == FOLDER_MIME]
    for pf in patient_folders:
        tree[pf["name"]] = {"id": pf["id"], "timepoints": {}}
    if not patient_folders:
        return tree

    # Level 2: timepoints for every patient, in parallel (bounded)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        tp_children_lists = list(pool.map(lambda p: _list_children(p["id"]), patient_folders))

    tp_jobs = []  # (patient_name, tp_folder)
    for pf, children in zip(patient_folders, tp_children_lists):
        for tp in children:
            if tp["mimeType"] != FOLDER_MIME:
                continue
            tree[pf["name"]]["timepoints"][tp["name"]] = {
                "id": tp["id"], "is_day0": _is_day0(tp["name"]), "scans": {},
            }
            tp_jobs.append((pf["name"], tp))
    if not tp_jobs:
        return tree

    # Level 3: scans for every timepoint, in parallel (bounded)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        scan_children_lists = list(pool.map(lambda job: _list_children(job[1]["id"]), tp_jobs))

    scan_jobs = []  # (patient_name, tp_name, scan_folder)
    for (patient_name, tp_folder), children in zip(tp_jobs, scan_children_lists):
        for scan in children:
            if scan["mimeType"] != FOLDER_MIME:
                continue
            scan_jobs.append((patient_name, tp_folder["name"], scan))
    if not scan_jobs:
        return tree

    # Level 4: files for every scan folder, in parallel (bounded)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        file_lists = list(pool.map(lambda job: _list_children(job[2]["id"]), scan_jobs))

    for (patient_name, tp_name, scan_folder), files in zip(scan_jobs, file_lists):
        entry = _scan_folder_to_entry(files, scan_folder)
        if entry:
            scan_id, scan_dict = entry
            tree[patient_name]["timepoints"][tp_name]["scans"][scan_id] = scan_dict

    return tree


def get_tree(root_id: str, force: bool = False) -> dict:
    now = time.time()
    if force or _CACHE["tree"] is None or (now - _CACHE["ts"]) > TREE_TTL_SECONDS:
        _CACHE["tree"] = _build_tree(root_id)
        _CACHE["ts"] = now
    return _CACHE["tree"]


def invalidate():
    _CACHE["ts"] = 0


def find_scan(tree: dict, scan_id: str):
    """Return (patient_name, timepoint_name, scan_dict) or None."""
    for patient_name, patient in tree.items():
        for tp_name, tp in patient["timepoints"].items():
            if scan_id in tp["scans"]:
                return patient_name, tp_name, tp["scans"][scan_id]
    return None


# ---------------------------------------------------------------------------
# High-level listing functions (mirror the old local-filesystem versions)
# ---------------------------------------------------------------------------

def discover_patients(tree: dict):
    patients = []
    for patient_name, patient in sorted(tree.items()):
        scan_count = 0
        day0_count = 0
        polygons_saved = 0
        for tp in patient["timepoints"].values():
            for scan in tp["scans"].values():
                scan_count += 1
                if tp["is_day0"]:
                    day0_count += 1
                if scan["json"] is not None:
                    polygons_saved += 1
        patients.append({
            "patient": patient_name,
            "timepoint_count": len(patient["timepoints"]),
            "scan_count": scan_count,
            "day0_count": day0_count,
            "polygons_saved": polygons_saved,
        })
    return patients


def discover_timepoints(tree: dict, patient_name: str):
    patient = tree.get(patient_name)
    if not patient:
        return []
    timepoints = []
    for tp_name, tp in patient["timepoints"].items():
        timepoints.append({
            "timepoint": tp_name,
            "is_day0": tp["is_day0"],
            "scan_count": len(tp["scans"]),
        })
    timepoints.sort(key=lambda t: (not t["is_day0"], t["timepoint"]))
    return timepoints


def discover_scans_in_timepoint(tree: dict, patient_name: str, timepoint_name: str):
    patient = tree.get(patient_name)
    if not patient:
        return []
    tp = patient["timepoints"].get(timepoint_name)
    if not tp:
        return []
    scans = []
    for scan_id, scan in sorted(tp["scans"].items()):
        scans.append({"scan_id": scan_id, "has_polygons": scan["json"] is not None})
    return scans


def all_scans_for_patient(tree: dict, patient_name: str):
    patient = tree.get(patient_name)
    if not patient:
        return []
    out = []
    for tp_name, tp in patient["timepoints"].items():
        for scan_id, scan in tp["scans"].items():
            available = scan.get("variants") or {}
            out.append({
                "timepoint": tp_name,
                "is_day0": tp["is_day0"],
                "scan_id": scan_id,
                "has_polygons": scan["json"] is not None,
                "variants": [v for v in VARIANT_ORDER if v in available],
            })
    out.sort(key=lambda s: (not s["is_day0"], s["timepoint"], s["scan_id"]))
    return out


# ---------------------------------------------------------------------------
# Polygon JSON schema translation (unchanged from the local version)
# ---------------------------------------------------------------------------

def normalize_polygons(raw: dict, scan_id: str, image_size):
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


def classify_scan_files(scan: dict):
    """Return the scan's non-primary files, tagged with how the UI should
    show each one: seg_image (tif needing conversion), image (png/jpg as-is),
    mesh_3d (.ply), data (json to pretty-print), other (plain download link).
    """
    primary_tif_id = scan["tif"]["id"]
    primary_json_id = scan["json"]["id"] if scan.get("json") else None

    out = []
    for f in scan.get("files", []):
        if f["id"] == primary_tif_id or f["id"] == primary_json_id:
            continue
        name_lower = f["name"].lower()
        if name_lower.endswith(".ply"):
            category = "mesh_3d"
        elif name_lower.endswith(".tif") or name_lower.endswith(".tiff"):
            category = "seg_image"
        elif name_lower.endswith((".png", ".jpg", ".jpeg")):
            category = "image"
        elif name_lower.endswith(".json"):
            category = "data"
        else:
            category = "other"
        out.append({"id": f["id"], "name": f["name"], "category": category})

    # Stable, readable order: 3D first, then images, then data, then rest.
    order = {"mesh_3d": 0, "seg_image": 1, "image": 2, "data": 3, "other": 4}
    out.sort(key=lambda f: (order.get(f["category"], 9), f["name"]))
    return out



def load_polygons(scan: dict, scan_id: str, image_size):
    json_file = scan.get("json")
    if not json_file:
        return {"scan_id": scan_id, "image_size": list(image_size), "regions": []}
    raw = json.loads(download_file_bytes(json_file["id"]))
    return normalize_polygons(raw, scan_id, image_size)


def save_polygons(scan: dict, scan_id: str, payload: dict):
    """Write payload to Drive, backing up an original all-SAM2 file once."""
    json_file = scan.get("json")

    if json_file:
        try:
            existing = json.loads(download_file_bytes(json_file["id"]))
            regions = existing.get("regions") or existing.get("polygons") or []
            if regions and all(r.get("source", "sam2") == "sam2" for r in regions):
                backup_name = f"{scan_id}_burn_polygons.sam2_backup.json"
                siblings = _list_children(scan["id"])
                if not any(f["name"] == backup_name for f in siblings):
                    copy_file(json_file["id"], backup_name, scan["id"])
        except Exception:
            pass
        upload_json(json_file["id"], None, None, payload)
    else:
        upload_json(None, scan["id"], f"{scan_id}_burn_polygons.json", payload)

    invalidate()
