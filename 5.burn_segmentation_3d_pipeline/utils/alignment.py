"""
utils/alignment.py
==================
3D alignment utility (from step2_align_3d_front.py).
Applies fixed Rx=-90°, Ry=+90° rotation + optional fine yaw correction.
"""

import numpy as np
import open3d as o3d


def make_rotation(rx=0.0, ry=0.0, rz=0.0) -> np.ndarray:
    """Build a 4x4 homogeneous rotation matrix from Euler angles (degrees)."""
    def Rx(d):
        r = np.radians(d)
        return np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    def Ry(d):
        r = np.radians(d)
        return np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
    def Rz(d):
        r = np.radians(d)
        return np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def align_front(pcd: o3d.geometry.PointCloud, fine_yaw: float = 0.0) -> o3d.geometry.PointCloud:
    """
    Apply standard front-facing alignment:
      Step 1: Rx = -90°
      Step 2: Ry = +90°
      Step 3: optional fine yaw correction (Ry = fine_yaw degrees)

    Returns the transformed point cloud (in-place + returned).
    """
    pcd.transform(make_rotation(rx=-90.0, ry=90.0))
    if fine_yaw != 0.0:
        pcd.transform(make_rotation(ry=fine_yaw))
    return pcd


def build_pcd(pts: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """Build an open3d PointCloud from (N,3) float pts and (N,3) uint8 colors."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return pcd
