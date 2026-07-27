"""
cylindrical_to_obj.py

Reconstructs a 3D mesh (binary .ply, vertex-colored) using ONLY the 2D
outputs of artec_to_cylindrical.py: <name>.cyl (Cyberware-style text header +
raw float32 range data) and its companion <name>_full.tif (color, pointed to
by the .cyl header's COLOR_FILE line). The original .obj/.mtl/texture are
NEVER read here - this proves the 2D files alone carry enough info to
rebuild 3D.

USAGE:
    python cylindrical_to_obj.py                                   (batch, uses DEFAULT_DATASET_ROOT below)
    python cylindrical_to_obj.py "D:\\...\\ARTEC_UNWRAP_OUTPUTS"      (batch)
    python cylindrical_to_obj.py myscan_prefix out_prefix           (single scan)
    add --overwrite to regenerate existing outputs

By default: builds the full pixel-resolution mesh (zero downsampling), then
runs quadric-error DECIMATION (via open3d) down to the SAME face count as
the original scan (stored in the .cyl header) - this collapses triangles in
flat/low-detail areas (neck, forehead) while preserving curved/detailed
areas (face features), giving a file close to the original's size (~40MB)
without the uniform detail loss that naive downsampling causes everywhere.

    --no-decimate     skip decimation, use only --stride downsampling instead
    --stride N        pick every Nth pixel as a vertex (default 1 = every
                       pixel); combine with --no-decimate for a quick, crude,
                       fast preview instead of the quadric-decimated result

Batch mode writes into a "reconstructed-3D" subfolder INSIDE each scan folder:
    ARTEC_UNWRAP_OUTPUTS/
        scan_1/
            scan_1.cyl / scan_1_full.tif   (unchanged 2D outputs)
            reconstructed-3D/scan_1_reconstructed.ply
        scan_2/...

NOT a lossless round-trip in the full pipeline sense: flat-shading + concave
surfaces (nostrils, inner ear, under chin) are inherent losses already baked
into the forward script's .cyl/_full.tif. This script does not add any
FURTHER loss on top at stride=1 - every pixel of that 2D data is preserved.
"""

import sys, os
import numpy as np
from PIL import Image

DEFAULT_DATASET_ROOT = r"D:\NahidW\Dataset\EXAMPLE OF SCANS FROM ARTEC EVA\ARTEC_UNWRAP_OUTPUTS"
DEFAULT_STRIDE = 3  # safe pre-decimation base; quadric decimation then reduces
                     # this down to match the original scan's face count


def read_cyl(path):
    """Parses the Cyberware-style header (text key=value lines up to 'DATA=')
    followed by a raw float32 W*H range array."""
    with open(path, 'rb') as f:
        data = f.read()
    marker = b'DATA=\n'
    idx = data.index(marker) + len(marker)
    header_text = data[:idx].decode('ascii')
    meta = {}
    for line in header_text.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            meta[k] = v
    W, H = int(meta['WIDTH']), int(meta['HEIGHT'])
    radius = np.frombuffer(data[idx:], dtype='<f4').reshape(H, W).astype(np.float64)
    for k in ('THETA_MIN', 'THETA_MAX', 'HEIGHT_MIN', 'HEIGHT_MAX', 'AXIS_X', 'AXIS_Z', 'ROTATION_OFFSET_RAD'):
        meta[k] = float(meta[k])
    return meta, radius


def reconstruct(in_prefix, out_prefix, stride, overwrite, decimate=True):
    if not overwrite and os.path.exists(f'{out_prefix}.ply'):
        print(f"[SKIP] {out_prefix}.ply exists (use --overwrite)")
        return

    # --- read ONLY the 2D outputs; original obj/mtl/texture are not touched ---
    meta, radius = read_cyl(f'{in_prefix}.cyl')
    color = np.array(Image.open(os.path.join(os.path.dirname(in_prefix), meta['COLOR_FILE'])).convert('RGB'))

    H, W = radius.shape
    theta = meta['THETA_MIN'] + (np.arange(W) / W) * (meta['THETA_MAX'] - meta['THETA_MIN'])
    height = meta['HEIGHT_MAX'] - (np.arange(H)[:, None] / H) * (meta['HEIGHT_MAX'] - meta['HEIGHT_MIN'])
    theta_true = np.tile(theta, (H, 1)) + meta['ROTATION_OFFSET_RAD']
    height = np.tile(height, (1, W))

    X = meta['AXIS_X'] + radius * np.sin(theta_true)
    Y = height
    Z = meta['AXIS_Z'] + radius * np.cos(theta_true)
    valid = radius > 1e-6

    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    target_faces = int(meta.get('SOURCE_FACE_COUNT', 0))

    for attempt in range(5):
        verts, vcolor, faces = build_grid_mesh(X, Y, Z, valid, color, stride)

        if not decimate:
            break
        print(f"  decimating {len(faces)} -> ~{target_faces} faces (matching original scan density) ...")
        try:
            verts, vcolor, faces = decimate_mesh(verts, vcolor, faces, target_faces)
            break
        except MemoryError:
            stride *= 2
            print(f"  [out of memory] retrying with a coarser starting mesh (stride={stride}) ...")
    else:
        print("  [WARNING] decimation kept running out of memory - saving without decimation instead")

    write_ply_binary(f'{out_prefix}.ply', verts, vcolor, faces)
    print(f"[OK] {out_prefix}.ply  ({len(verts)} verts, {len(faces)} faces)")


def build_grid_mesh(X, Y, Z, valid, color, stride):
    Xs, Ys, Zs, valids = X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride], valid[::stride, ::stride]
    rgb = color[::stride, ::stride]
    h, w = Xs.shape
    vid = -np.ones((h, w), dtype=np.int64)
    vid[valids] = np.arange(valids.sum())
    verts = np.stack([Xs[valids], Ys[valids], Zs[valids]], axis=-1).astype(np.float32)
    vcolor = rgb[valids].astype(np.uint8)

    v00, v01, v10, v11 = vid[:-1, :-1], vid[:-1, 1:], vid[1:, :-1], vid[1:, 1:]
    q = (v00 >= 0) & (v01 >= 0) & (v10 >= 0) & (v11 >= 0)
    faces = np.concatenate([np.stack([v00[q], v01[q], v10[q]], -1),
                             np.stack([v01[q], v11[q], v10[q]], -1)], axis=0).astype(np.int32)
    return verts, vcolor, faces


def decimate_mesh(verts, vcolor, faces, target_faces):
    """Quadric-error decimation via open3d: collapses triangles in flat/low-
    detail areas while preserving curved/detailed areas (face features),
    unlike uniform --stride downsampling which blurs everything equally."""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcolor.astype(np.float64) / 255.0)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    out_verts = np.asarray(mesh.vertices, dtype=np.float32)
    out_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    out_faces = np.asarray(mesh.triangles, dtype=np.int32)
    return out_verts, out_colors, out_faces


def write_ply_binary(path, verts, vcolor, faces):
    n_v, n_f = len(verts), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    vdtype = np.dtype([('xyz', '<f4', 3), ('rgb', 'u1', 3)])
    vdata = np.empty(n_v, dtype=vdtype)
    vdata['xyz'] = verts
    vdata['rgb'] = vcolor

    fdtype = np.dtype([('count', 'u1'), ('idx', '<i4', 3)])
    fdata = np.empty(n_f, dtype=fdtype)
    fdata['count'] = 3
    fdata['idx'] = faces

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vdata.tobytes())
        f.write(fdata.tobytes())


def find_scans(root):
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        cyls = [f for f in os.listdir(d) if f.endswith('.cyl')] if os.path.isdir(d) else []
        if cyls:
            out.append((name, os.path.join(d, cyls[0][:-len('.cyl')])))
    return out


def main():
    args = sys.argv[1:]
    overwrite = '--overwrite' in args
    decimate = '--no-decimate' not in args
    args = [a for a in args if a not in ('--overwrite', '--no-decimate')]
    stride = DEFAULT_STRIDE
    if '--stride' in args:
        i = args.index('--stride'); stride = int(args[i+1]); args = args[:i] + args[i+2:]

    root = args[0] if args else DEFAULT_DATASET_ROOT
    if root and os.path.isdir(root) and not os.path.exists(f"{root}.cyl"):
        scans = find_scans(root)
        if not scans:
            print(f"No scan subfolders with a *.cyl file found under {root}"); sys.exit(1)
        for name, in_prefix in scans:
            out_dir = os.path.join(os.path.dirname(in_prefix), "reconstructed-3D")
            reconstruct(in_prefix, os.path.join(out_dir, f"{name}_reconstructed"), stride, overwrite, decimate)
    elif len(args) >= 2:
        reconstruct(args[0], args[1], stride, overwrite, decimate)
    else:
        print(__doc__); sys.exit(1)


if __name__ == '__main__':
    main()