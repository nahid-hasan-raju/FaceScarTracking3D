"""
utils/writers.py
================
PLY writer and 2D snapshot renderer.
"""

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d

IMG_W = 800
IMG_H = 1000


def write_ply(pts: np.ndarray, colors: np.ndarray, output_path: Path) -> int:
    """Write binary PLY with XYZ + RGB per vertex. Returns file size in bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    packed = np.zeros(len(pts), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
    ])
    packed["x"], packed["y"], packed["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    packed["r"], packed["g"], packed["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())
    return output_path.stat().st_size


def render_snapshot(pcd: o3d.geometry.PointCloud, out_path: Path):
    """Render a top-down 2D snapshot of the aligned point cloud."""
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    order       = np.argsort(pts[:, 2])
    pts         = pts[order]
    cols        = cols[order]
    cols_bright = np.clip(cols * 1.3, 0, 1)
    cols_uint8  = (cols_bright * 255).astype(np.uint8)

    x = pts[:, 0];  y = pts[:, 1]
    pad = 0.02
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ix = ((x - x_min) / (x_max - x_min + 1e-9) * (1 - 2*pad) + pad) * (IMG_W - 1)
    iy = ((y - y_min) / (y_max - y_min + 1e-9) * (1 - 2*pad) + pad) * (IMG_H - 1)
    ix = ix.astype(int).clip(0, IMG_W - 1)
    iy = iy.astype(int).clip(0, IMG_H - 1)

    img  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    N    = 3;  half = N // 2
    for i in range(len(pts)):
        y0 = max(0, iy[i]-half);  y1 = min(IMG_H, iy[i]+half+1)
        x0 = max(0, ix[i]-half);  x1 = min(IMG_W, ix[i]+half+1)
        img[y0:y1, x0:x1] = cols_uint8[i]

    img     = np.flipud(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)
