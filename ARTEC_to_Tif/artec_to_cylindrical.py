"""
artec_to_cylindrical.py

Converts Artec OBJ+MTL+texture scans into a continuous cylindrical-projection
image (color + range/depth map + metadata), instead of Artec's fragmented
multi-island UV atlas.

USAGE:
    python artec_to_cylindrical.py                                  (batch, uses DEFAULT_DATASET_ROOT below)
    python artec_to_cylindrical.py "D:\\...\\dataset_root"            (batch: folder of scan_1/, scan_2/, ...)
    python artec_to_cylindrical.py mesh.obj texture.jpg out_prefix  (single scan)

Batch mode writes into "<dataset_root>/ARTEC_UNWRAP_OUTPUTS/scan_N/":
    scan_N_full.jpg / .tif   - 360-degree color unwrap (an ordinary TIFF/JPG)
    scan_N.cyl               - Cyberware-style header + raw float32 range data,
                               everything needed to invert back to 3D (see
                               save_geometry_file() below for the exact format)

HOW IT WORKS:
    1. Parse vertices/UVs/faces from the OBJ; sample color for every UV from
       the texture atlas (bakes the fragmented atlas into per-vertex color).
    2. Cylindrical coords per vertex: theta=angle around a vertical axis
       through the head, height=Y, radius=distance from that axis.
    3. Auto-center: find the largest angular gap with NO mesh data (e.g. the
       back of the head, unreachable by a frontal scanner) and rotate so
       that gap becomes the seam - this centers the face and keeps it
       left/right symmetric.
    4. Rasterize triangles into (theta,height) image space, painter's-algorithm
       ordered by radius (outermost/visible surface drawn last). Triangles
       straddling the seam are duplicated on both edges so it tiles cleanly.

Flat-shaded (each triangle = one average color/radius) - looks continuous at
Artec mesh density. head_fraction/width_px below are the main tunables.
"""

import sys, os, re, math, traceback
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

DEFAULT_DATASET_ROOT = r"D:\NahidW\Dataset\EXAMPLE OF SCANS FROM ARTEC EVA"


def parse_obj(obj_path):
    verts, vts, face_v, face_vt = [], [], [], []
    with open(obj_path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split(); verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('vt'):
                p = line.split(); vts.append((float(p[1]), float(p[2])))
            elif line.startswith('f '):
                p = line.split(); fv, fvt = [], []
                for corner in p[1:4]:
                    a, b = corner.split('/')[:2]
                    fv.append(int(a) - 1); fvt.append(int(b) - 1)
                face_v.append(fv); face_vt.append(fvt)
    return (np.array(verts), np.array(vts),
            np.array(face_v, dtype=np.int64), np.array(face_vt, dtype=np.int64))


def sample_texture(vts, texture_path):
    tex = np.array(Image.open(texture_path).convert('RGB'))
    H, W, _ = tex.shape
    u, v = vts[:, 0], vts[:, 1]
    px = np.clip(u * (W - 1), 0, W - 1); py = np.clip((1.0 - v) * (H - 1), 0, H - 1)
    x0 = np.floor(px).astype(int); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(py).astype(int); y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (px - x0)[:, None]; fy = (py - y0)[:, None]
    c00 = tex[y0, x0].astype(np.float64); c10 = tex[y0, x1].astype(np.float64)
    c01 = tex[y1, x0].astype(np.float64); c11 = tex[y1, x1].astype(np.float64)
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def find_seam_rotation(theta_v):
    """Largest angular gap with NO mesh data (e.g. unscanned back of head) ->
    rotation offset that puts that gap at the seam, centering the face."""
    st = np.sort(theta_v)
    gaps = np.append(np.diff(st), (st[0] + 2 * np.pi) - st[-1])
    i = np.argmax(gaps)
    a, b = (st[i], st[i + 1]) if i < len(st) - 1 else (st[-1], st[0] + 2 * np.pi)
    return (a + b) / 2 + np.pi


def build_cylindrical(verts, face_v, face_vt, vt_colors, head_fraction=0.35):
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    head_mask = verts[:, 1] > (ymax - head_fraction * (ymax - ymin))
    cx, cz = verts[head_mask, 0].mean(), verts[head_mask, 2].mean()

    dx, dz = verts[:, 0] - cx, verts[:, 2] - cz
    theta_v = np.arctan2(dx, dz)
    radius_v = np.sqrt(dx ** 2 + dz ** 2)
    height_v = verts[:, 1]

    rotation_offset = find_seam_rotation(theta_v)
    theta_v = (theta_v - rotation_offset + np.pi) % (2 * np.pi) - np.pi

    th = theta_v[face_v].copy()
    seam_mask = np.zeros(th.shape[0], dtype=bool)
    for i in range(1, 3):
        diff = th[:, i] - th[:, 0]
        seam_mask |= (diff > np.pi) | (diff < -np.pi)
        th[diff > np.pi, i] -= 2 * np.pi
        th[diff < -np.pi, i] += 2 * np.pi

    ht, rad = height_v[face_v], radius_v[face_v]
    col = vt_colors[face_vt] / 255.0

    tri_xy = np.stack([th, ht], axis=-1)
    tri_colors = col.mean(axis=1)
    tri_radius = rad.mean(axis=1)

    # Seam-straddling triangles: duplicate on BOTH edges of the unwrap (like a
    # world map repeating its edge longitude), so neighboring seam triangles
    # that independently land on opposite sides still tile with no gap.
    if seam_mask.any():
        dup_xy = tri_xy[seam_mask].copy()
        shift = np.where(dup_xy[:, :, 0].mean(axis=1) > 0, -2 * np.pi, 2 * np.pi)
        dup_xy[:, :, 0] += shift[:, None]
        tri_xy = np.concatenate([tri_xy, dup_xy])
        tri_colors = np.concatenate([tri_colors, tri_colors[seam_mask]])
        tri_radius = np.concatenate([tri_radius, tri_radius[seam_mask]])
        rad_for_order = np.concatenate([rad.max(axis=1), rad.max(axis=1)[seam_mask]])
    else:
        rad_for_order = rad.max(axis=1)

    order = np.argsort(rad_for_order)  # far first, near last (painter's algorithm)
    mean_radius = radius_v[head_mask].mean()
    return tri_xy[order], tri_colors[order], tri_radius[order], (cx, cz), mean_radius, rotation_offset


def render(tri_xy, tri_colors, mean_radius, width_px, out_prefix):
    theta_min, theta_max = tri_xy[:, :, 0].min(), tri_xy[:, :, 0].max()
    height_min, height_max = tri_xy[:, :, 1].min(), tri_xy[:, :, 1].max()
    height_px = max(256, int(width_px * (height_max - height_min) / ((theta_max - theta_min) * mean_radius)))

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(theta_min, theta_max); ax.set_ylim(height_min, height_max)
    ax.axis('off'); ax.set_facecolor('black')
    ax.add_collection(PolyCollection(tri_xy, facecolors=tri_colors, edgecolors='face',
                                      linewidths=0.3, antialiaseds=True))
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    Image.fromarray(rgb).save(f'{out_prefix}_full.tif')              # lossless, saved first
    Image.fromarray(rgb).save(f'{out_prefix}_full.jpg', quality=95)  # lossy convenience copy
    return theta_min, theta_max, height_min, height_max, width_px, height_px


def save_geometry_file(out_prefix, tri_xy, tri_radius, bounds, axis, mean_radius, verts_count, faces_count, rotation_offset):
    """
    Rasterizes the radius/range map AND writes it together with a text header
    into ONE file, mirroring the real Cyberware digitizer format (a text
    header + raw binary range array, with color kept in a separate ordinary
    TIFF - exactly what a real Cyberware .obj-era scan file looks like):

        Cylindrical Digitizer Data
        SPACE=CYLINDRICAL
        WIDTH=...        NLG equivalent
        HEIGHT=...       NLT equivalent
        THETA_MIN=...    THETA_MAX=...
        HEIGHT_MIN=...   HEIGHT_MAX=...
        AXIS_X=...       AXIS_Z=...
        ROTATION_OFFSET_RAD=...
        COLOR_FILE=<name>_full.tif
        DATA=
        <raw float32 radius values, row-major, WIDTH*HEIGHT of them>

    To reconstruct 3D: theta = theta_min + (x/W)*(theta_max-theta_min) + rotation_offset;
    height = height_max - (y/H)*(height_max-height_min); X=axis_x+r*sin(theta);
    Y=height; Z=axis_z+r*cos(theta).
    """
    theta_min, theta_max, height_min, height_max, width_px, height_px = bounds
    theta_min, theta_max = float(theta_min), float(theta_max)
    height_min, height_max = float(height_min), float(height_max)
    axis_x, axis_z = float(axis[0]), float(axis[1])
    mean_radius, rotation_offset = float(mean_radius), float(rotation_offset)

    px = (tri_xy[:, :, 0] - theta_min) / (theta_max - theta_min) * width_px
    py = (height_max - tri_xy[:, :, 1]) / (height_max - height_min) * height_px

    im = Image.new('F', (width_px, height_px), 0.0)
    d = ImageDraw.Draw(im)
    for i in range(tri_xy.shape[0]):
        d.polygon(list(zip(px[i], py[i])), fill=float(tri_radius[i]))
    radius_array = np.array(im, dtype='<f4')

    header = (
        "Cylindrical Digitizer Data\n"
        "SPACE=CYLINDRICAL\n"
        f"WIDTH={width_px}\nHEIGHT={height_px}\n"
        f"THETA_MIN={theta_min!r}\nTHETA_MAX={theta_max!r}\n"
        f"HEIGHT_MIN={height_min!r}\nHEIGHT_MAX={height_max!r}\n"
        f"AXIS_X={axis_x!r}\nAXIS_Z={axis_z!r}\n"
        f"ROTATION_OFFSET_RAD={rotation_offset!r}\n"
        f"MEAN_RADIUS={mean_radius!r}\n"
        f"SOURCE_VERTEX_COUNT={int(verts_count)}\nSOURCE_FACE_COUNT={int(faces_count)}\n"
        "RANGE_DTYPE=float32\n"
        f"COLOR_FILE={os.path.basename(out_prefix)}_full.tif\n"
        "DATA=\n"
    )
    with open(f'{out_prefix}.cyl', 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(radius_array.tobytes())


def convert_one(obj_path, texture_path, out_prefix, verbose_prefix=''):
    print(f"{verbose_prefix}Parsing OBJ ...")
    verts, vts, face_v, face_vt = parse_obj(obj_path)
    print(f"{verbose_prefix}  {len(verts)} vertices, {len(vts)} uv coords, {len(face_v)} faces")

    print(f"{verbose_prefix}Sampling texture atlas ...")
    vt_colors = sample_texture(vts, texture_path)

    print(f"{verbose_prefix}Computing cylindrical projection ...")
    tri_xy, tri_colors, tri_radius, axis, mean_radius, rot = build_cylindrical(verts, face_v, face_vt, vt_colors)
    print(f"{verbose_prefix}  axis (x,z)={axis}, rotation={math.degrees(rot):.1f} deg")

    print(f"{verbose_prefix}Rendering color unwrap ...")
    bounds = render(tri_xy, tri_colors, mean_radius, 4096, out_prefix)

    print(f"{verbose_prefix}Writing geometry file (header + range data) ...")
    save_geometry_file(out_prefix, tri_xy, tri_radius, bounds, axis, mean_radius, len(verts), len(face_v), rot)

    print(f"{verbose_prefix}Done: {out_prefix}_full.jpg/.tif (color), {out_prefix}.cyl (header+range)")


# ----------------------------------------------------------------------
# Batch mode
# ----------------------------------------------------------------------

def find_mtl_texture(mtl_path):
    with open(mtl_path, 'r', errors='replace') as f:
        for line in f:
            m = re.match(r'\s*map_Kd\s+(.+)', line.strip())
            if m:
                return m.group(1).strip().strip('"')
    return None


def process_scan_folder(scan_dir, output_root):
    scan_name = os.path.basename(os.path.normpath(scan_dir))
    print(f"\n=== Processing {scan_name} ===")
    files = os.listdir(scan_dir)
    obj_files = [f for f in files if f.lower().endswith('.obj')]
    mtl_files = [f for f in files if f.lower().endswith('.mtl')]
    if not obj_files or not mtl_files:
        print(f"  [SKIP] missing .obj or .mtl in {scan_dir}"); return

    texture_name = find_mtl_texture(os.path.join(scan_dir, mtl_files[0]))
    if not texture_name:
        print(f"  [SKIP] no map_Kd entry in {mtl_files[0]}"); return
    texture_path = os.path.join(scan_dir, texture_name)
    if not os.path.exists(texture_path):
        print(f"  [SKIP] texture '{texture_name}' not found in {scan_dir}"); return

    out_dir = os.path.join(output_root, scan_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        convert_one(os.path.join(scan_dir, obj_files[0]), texture_path,
                    os.path.join(out_dir, scan_name), verbose_prefix='  ')
    except Exception as e:
        print(f"  [ERROR] {scan_name}: {e}"); traceback.print_exc()


def run_batch(dataset_root, output_root=None):
    output_root = output_root or os.path.join(dataset_root, "ARTEC_UNWRAP_OUTPUTS")
    os.makedirs(output_root, exist_ok=True)
    print(f"Dataset root: {dataset_root}\nOutput root:  {output_root}")

    scan_dirs = [os.path.join(dataset_root, d) for d in sorted(os.listdir(dataset_root))
                 if os.path.isdir(os.path.join(dataset_root, d))]
    if not scan_dirs:
        print("No subfolders found - is this the right path?"); sys.exit(1)
    for scan_dir in scan_dirs:
        process_scan_folder(scan_dir, output_root)
    print("\nAll done.")


def main():
    if len(sys.argv) < 2:
        if DEFAULT_DATASET_ROOT and os.path.isdir(DEFAULT_DATASET_ROOT):
            print(f"No arguments given - using default dataset root:\n  {DEFAULT_DATASET_ROOT}\n")
            return run_batch(DEFAULT_DATASET_ROOT)
        print(__doc__); sys.exit(1)

    first_arg = sys.argv[1]
    if os.path.isdir(first_arg):
        run_batch(first_arg, sys.argv[2] if len(sys.argv) > 2 else None)
    elif os.path.isfile(first_arg) and first_arg.lower().endswith('.obj'):
        if len(sys.argv) < 4:
            print("Usage: python artec_to_cylindrical.py mesh.obj texture.jpg out_prefix"); sys.exit(1)
        obj_path, texture_path, out_prefix = sys.argv[1:4]
        if not os.path.exists(texture_path):
            print(f"Texture file not found: {texture_path}"); sys.exit(1)
        convert_one(obj_path, texture_path, out_prefix)
    else:
        print(f"Could not understand first argument: {first_arg}"); print(__doc__); sys.exit(1)


if __name__ == '__main__':
    main()