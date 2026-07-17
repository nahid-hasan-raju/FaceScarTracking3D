#!/usr/bin/env python3
"""
utils/landmarks.py
===================
Fully automated calibration-dot landmark detection + 3D lifting + cross-scan
rigid alignment, for face-burn-tracking registration.

PIPELINE:
  1. detect_candidates_2d(img_rgb)
       Finds all red-or-green calibration-dot-like blobs in the unwrapped
       2D texture. Deliberately permissive (color + size + shape only) --
       false positives are expected and get rejected in step 3.

  2. lift_candidates_to_3d(candidates, pts, tif_row, tif_col)
       Converts each 2D blob into a single 3D point (median of all 3D
       points whose tif_row/tif_col falls inside that blob).

  3. match_to_template(candidate_pts_3d, template_3d, tol_mm, min_inliers)
       RANSAC-style: searches for the rigid transform (R, t) that aligns
       the largest subset of candidates to the labeled reference template
       (built once from a clean reference scan -- see build_template()).
       Tolerates missing landmarks and rejects false-positive candidates
       automatically -- no per-scan manual tuning required.

  4. Apply the resulting (R, t) to a scan's burn-region 3D points before
     any cross-scan comparison.

USAGE (see __main__ at the bottom for a runnable example):
    from utils.landmarks import (
        detect_candidates_2d, lift_candidates_to_3d, match_to_template, kabsch
    )
"""

import numpy as np
import cv2
from itertools import combinations, permutations

# ─────────────────────────────────────────────────────────────────────────────
#  COLOR DETECTION (red OR green calibration dots)
# ─────────────────────────────────────────────────────────────────────────────

def _red_mask(img_rgb: np.ndarray) -> np.ndarray:
    R, G, B = img_rgb[:, :, 0].astype(int), img_rgb[:, :, 1].astype(int), img_rgb[:, :, 2].astype(int)
    return ((R - G > 40) & (R - B > 25) & (G < 60) & (B < 75)).astype(np.uint8) * 255


def _green_mask(img_rgb: np.ndarray) -> np.ndarray:
    R, G, B = img_rgb[:, :, 0].astype(int), img_rgb[:, :, 1].astype(int), img_rgb[:, :, 2].astype(int)
    return ((G - R > 8) & (B - R > -5) & (R < 65) & (G < 80) & (B < 80)).astype(np.uint8) * 255


def detect_candidates_2d(img_rgb: np.ndarray,
                          min_area: int = 8, max_area: int = 90,
                          min_aspect: float = 0.3, max_aspect: float = 3.0,
                          min_fill: float = 0.35):
    """
    Returns a list of dicts: {cx, cy, area, color, mask} -- one per candidate
    blob, for BOTH red and green color schemes combined. `mask` is a boolean
    (H, W) array marking exactly that blob's pixels (used for 3D lifting).
    Deliberately permissive -- expect false positives, handled downstream.
    """
    candidates = []
    for color_name, mask_fn in [("red", _red_mask), ("green", _green_mask)]:
        m = mask_fn(img_rgb)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            aspect = w / h if h > 0 else 0
            fill = area / (w * h) if w * h > 0 else 0
            if min_area <= area <= max_area and min_aspect <= aspect <= max_aspect and fill > min_fill:
                cx, cy = centroids[i]
                candidates.append({
                    "cx": cx, "cy": cy, "area": int(area), "color": color_name,
                    "mask": (labels == i),
                })
    candidates = _merge_nearby(candidates, radius=15)
    return candidates


def _merge_nearby(candidates, radius=15):
    """Merge candidates of the SAME color within `radius` px (handles a
    single dot fragmented into multiple components by anti-aliasing)."""
    merged = []
    used = [False] * len(candidates)
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        group = [c]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if used[j] or candidates[j]["color"] != c["color"]:
                continue
            if np.hypot(candidates[j]["cx"] - c["cx"], candidates[j]["cy"] - c["cy"]) <= radius:
                group.append(candidates[j])
                used[j] = True
        combined_mask = group[0]["mask"].copy()
        for g in group[1:]:
            combined_mask |= g["mask"]
        ys, xs = np.where(combined_mask)
        merged.append({
            "cx": float(xs.mean()), "cy": float(ys.mean()),
            "area": int(combined_mask.sum()), "color": c["color"],
            "mask": combined_mask,
        })
    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  2D -> 3D LIFTING
# ─────────────────────────────────────────────────────────────────────────────

def lift_candidates_to_3d(candidates, pts, tif_row, tif_col):
    """Adds 'xyz' (or None) and 'n3d' to each candidate dict, in place."""
    for c in candidates:
        in_blob = c["mask"][tif_row, tif_col]
        n3d = int(in_blob.sum())
        c["n3d"] = n3d
        c["xyz"] = np.median(pts[in_blob], axis=0) if n3d > 0 else None
    return [c for c in candidates if c["xyz"] is not None]


# ─────────────────────────────────────────────────────────────────────────────
#  RIGID ALIGNMENT (Kabsch) + Z-AXIS CONSTRAINT + RANSAC MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def kabsch(src: np.ndarray, dst: np.ndarray):
    """Best-fit rigid transform (R, t) mapping src -> dst (both (N,3), N>=3)."""
    src_c = src - src.mean(0)
    dst_c = dst - dst.mean(0)
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = dst.mean(0) - R @ src.mean(0)
    return R, t


def _decompose_zyx(R: np.ndarray):
    """
    Decompose rotation matrix into ZYX Euler angles (yaw, pitch, roll).
    In scanner coordinates: Z = vertical body axis, X/Y = horizontal.
      yaw   = rotation around Z (head turning left/right)
      pitch = rotation around Y (head nodding forward/back)
      roll  = rotation around X (head tilting sideways)
    """
    import math
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return yaw, pitch, roll


def _compose_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build rotation matrix from ZYX Euler angles (radians)."""
    def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def constrain_rotation(R: np.ndarray, max_tilt_deg: float = 20.0) -> np.ndarray:
    """
    Allow free yaw (rotation around Z — the vertical body/scanner axis),
    but clamp pitch and roll to ±max_tilt_deg.

    WHY: in a Cyberware cylindrical scanner the patient is always upright
    so Z is constant between sessions.  What varies is small head yaw
    (turning left/right) plus minor pitch/roll from posture differences.
    Without this constraint, Kabsch may solve a 3-point problem by wildly
    rotating the body in pitch/roll — mathematically valid but physically
    impossible. Clamping pitch/roll prevents the body from rotating and
    forces the solver to adjust only what actually changes session to session.
    """
    import math
    yaw, pitch, roll = _decompose_zyx(R)
    max_rad = math.radians(max_tilt_deg)
    pitch   = float(np.clip(pitch, -max_rad, max_rad))
    roll    = float(np.clip(roll,  -max_rad, max_rad))
    return _compose_zyx(yaw, pitch, roll)


def match_to_template(candidates_3d: list, template_3d: dict,
                       tol_mm: float = 8.0, min_inliers: int = 3,
                       pairdist_slack_mm: float = 16.0,
                       require_same_color: bool = True,
                       max_rotation_deg: float = 90.0,
                       max_tilt_deg: float = 20.0):
    """
    candidates_3d : list of candidate dicts (post lift_candidates_to_3d).
    template_3d   : {landmark_name: [x,y,z], ...} reference positions.

    require_same_color : calibration dots are one consistent color per
                     session (all-red or all-green). Rejects mixed-color
                     hypotheses.

    max_rotation_deg : reject any hypothesis whose total rotation exceeds
                     this. Catches gross mismatches (e.g. head-flip).

    max_tilt_deg  : after solving Kabsch, clamp pitch and roll to
                     ±max_tilt_deg while leaving yaw (Z-axis rotation)
                     free. This is the key physical constraint:
                     - Z axis = vertical body axis, always upright
                     - Yaw = head turning left/right, can vary freely
                     - Pitch/roll = head nodding/tilting, should be small
                     Without this, the solver may rotate the entire body
                     in pitch/roll just to satisfy 3 landmark constraints,
                     which is physically impossible in a clinical scan.
                     Confirmed fix: M15_A wrong alignment required ~90deg
                     pitch — clamping to 20deg rejects it outright.

    Returns dict {R, t, matches, n_inliers, mean_residual_mm,
                  rotation_deg, yaw_deg, pitch_deg, roll_deg}
    or None if nothing scored >= min_inliers.
    """
    import math
    names        = list(template_3d.keys())
    template_pts = np.array([template_3d[n] for n in names])
    cand_pts     = np.array([c["xyz"] for c in candidates_3d])
    cand_colors  = [c.get("color") for c in candidates_3d]
    n_t, n_c     = len(template_pts), len(cand_pts)
    if n_c < 3:
        return None

    def pdists(p):
        return sorted(np.linalg.norm(p[i] - p[j]) for i, j in combinations(range(3), 2))

    def rotation_angle_deg(R):
        return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))

    best = None
    for t_idx in combinations(range(n_t), 3):
        t_triplet = template_pts[list(t_idx)]
        dt = pdists(t_triplet)
        for c_idx in permutations(range(n_c), 3):
            if require_same_color:
                cs = {cand_colors[i] for i in c_idx}
                if len(cs) > 1:
                    continue
            c_triplet = cand_pts[list(c_idx)]
            dc = pdists(c_triplet)
            if any(abs(a - b) > pairdist_slack_mm for a, b in zip(dt, dc)):
                continue
            R, t = kabsch(c_triplet, t_triplet)
            if rotation_angle_deg(R) > max_rotation_deg:
                continue

            # Apply Z-axis constraint: clamp pitch and roll, keep yaw free
            R = constrain_rotation(R, max_tilt_deg)
            t = t_triplet.mean(0) - R @ c_triplet.mean(0)

            transformed_all = (R @ cand_pts.T).T + t

            matches, residuals = {}, []
            for ti, tp in enumerate(template_pts):
                d = np.linalg.norm(transformed_all - tp, axis=1)
                j = int(np.argmin(d))
                if d[j] < tol_mm:
                    if require_same_color and matches:
                        existing_color = cand_colors[next(iter(matches.values()))]
                        if cand_colors[j] != existing_color:
                            continue
                    matches[names[ti]] = j
                    residuals.append(d[j])
            if len(matches) >= min_inliers:
                score = (len(matches), -np.mean(residuals) if residuals else 0)
                if best is None or score > best["_score"]:
                    yaw, pitch, roll = _decompose_zyx(R)
                    best = {
                        "R": R, "t": t, "matches": matches,
                        "n_inliers": len(matches),
                        "mean_residual_mm": float(np.mean(residuals)),
                        "rotation_deg": rotation_angle_deg(R),
                        "yaw_deg":   float(np.degrees(yaw)),
                        "pitch_deg": float(np.degrees(pitch)),
                        "roll_deg":  float(np.degrees(roll)),
                        "_score": score,
                    }
    if best is not None:
        del best["_score"]
    return best


def apply_transform(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a rigid (R, t) to an (N,3) array of 3D points."""
    return (R @ points_3d.T).T + t


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD A LABELED REFERENCE TEMPLATE (one-time, semi-manual bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def build_template(img_rgb, pts, tif_row, tif_col, named_pixel_targets: dict, merge_radius: int = 15):
    """
    named_pixel_targets : {landmark_name: (px_x, px_y), ...} -- approximate
    2D pixel locations for each true landmark, confirmed once by a human on
    a clean reference scan (see conversation for how PAT01_D00_A's 7 points
    were identified). Returns {landmark_name: [x,y,z]}.
    """
    candidates = detect_candidates_2d(img_rgb)
    template = {}
    for name, (tx, ty) in named_pixel_targets.items():
        combined_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        for c in candidates:
            if np.hypot(c["cx"] - tx, c["cy"] - ty) <= merge_radius:
                combined_mask |= c["mask"]
        in_blob = combined_mask[tif_row, tif_col]
        if in_blob.sum() == 0:
            continue
        template[name] = np.median(pts[in_blob], axis=0).tolist()
    return template