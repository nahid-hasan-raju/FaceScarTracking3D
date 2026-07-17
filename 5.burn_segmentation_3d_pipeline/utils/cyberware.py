"""
utils/cyberware.py
==================
Cyberware range file parser and 3D reconstruction utility.
Extracted from step1_convert_all_to_3d.py as a clean importable module.

Returns (pts, colors) given a range file + RGB image (numpy array).
"""

import math
import numpy as np
from pathlib import Path

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4   # 457.2 mm
SCANNER_RADIUS_MM = 9  * 25.4   # 228.6 mm


def parse_header(filepath: Path):
    with open(filepath, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"Cyberware"):
        raise ValueError(f"Not a Cyberware range file: {filepath}")
    idx = raw.find(b"DATA=\n")
    if idx == -1:
        raise ValueError("Could not find DATA= marker.")
    header_end = idx + len(b"DATA=\n")
    params = {}
    for line in raw[:header_end].decode("ascii", errors="replace").split("\n"):
        if "=" in line and not line.startswith("DATA"):
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params, header_end, raw


def range_to_3d(range_path: Path, color_img_rgb: np.ndarray):
    """
    Convert a Cyberware range file to 3D point cloud.

    Parameters
    ----------
    range_path    : Path to the Cyberware range file (no extension)
    color_img_rgb : (H, W, 3) uint8 numpy array — the unwrapped 2D texture image.
                    Can be the original TIF OR the segmentation-colored image.

    Returns
    -------
    pts    : (N, 3) float32  — XYZ coordinates in mm
    colors : (N, 3) uint8   — RGB colors sampled from color_img_rgb
    tif_rows : (N,) int     — row index into color_img_rgb for each point
    tif_cols : (N,) int     — col index into color_img_rgb for each point
    valid_mask : (NLG, NLT) bool — full grid validity mask
    """
    params, header_end, raw = parse_header(range_path)

    NLG    = int(params["NLG"])
    NLT    = int(params["NLT"])
    RSHIFT = int(params["RSHIFT"])
    LGINCR = int(params["LGINCR"])

    r_scale_mm = LGINCR / 32768.0
    z_scale_mm = SCANNER_HEIGHT_MM / NLT
    theta_step = (2.0 * math.pi) / NLG

    # Range data
    data = (np.frombuffer(raw[header_end:header_end + NLG * NLT * 2], dtype=">u2")
              .reshape(NLG, NLT)
              .astype(np.float32))

    valid_mask = (data != INVALID_SENTINEL) & (data > 0)
    radius_mm  = np.where(valid_mask, (data / (2 ** RSHIFT)) * r_scale_mm, np.nan)
    valid_mask = (~np.isnan(radius_mm) &
                  (radius_mm > 0) &
                  (radius_mm <= SCANNER_RADIUS_MM))

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise RuntimeError("No valid range points found.")

    # Cylindrical → Cartesian
    Z_grid, THETA = np.meshgrid(
        np.arange(NLT) * z_scale_mm,
        np.arange(NLG) * theta_step,
    )
    X = np.where(valid_mask, radius_mm * np.cos(THETA), np.nan)
    Y = np.where(valid_mask, radius_mm * np.sin(THETA), np.nan)

    rows, cols = np.where(valid_mask)   # rows = angular (0..NLG-1), cols = height (0..NLT-1)
    pts = np.column_stack([X[valid_mask], Y[valid_mask], Z_grid[valid_mask]])

    # Map each 3D point → pixel in the 2D unwrapped image
    ch, cw = color_img_rgb.shape[:2]
    tif_row = (ch - 1 - (cols * ch / NLT).astype(int).clip(0, ch - 1))
    tif_col = (rows * cw / NLG).astype(int).clip(0, cw - 1)
    colors  = color_img_rgb[tif_row, tif_col].astype(np.uint8)

    return pts, colors, tif_row, tif_col, valid_mask
