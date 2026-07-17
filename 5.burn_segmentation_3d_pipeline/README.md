# Burn 3D Pipeline

Unwrapped 2D TIFF → Burn Segmentation → 3D Point Cloud with burn areas marked in red.

## Folder Structure

```
burn_3d_pipeline/
  burn_3d_pipeline.py       ← main script (run this)
  config.yaml               ← edit this to configure inputs/model/output
  requirements.txt
  utils/
    cyberware.py            ← Cyberware range file parser + 3D reconstruction
    alignment.py            ← front-facing rotation alignment
    writers.py              ← PLY writer + 2D snapshot renderer
    models/
      unetpp.py             ← UNet++ (EfficientNet-B4) inference
      segformer.py          ← SegFormer-B5 inference
      medsam.py             ← MedSAM (SAM ViT-B) inference
      sam2.py               ← SAM2 (Hiera-Large) inference
  outputs/                  ← generated outputs land here
```

## What It Does

```
TIFF (unwrapped 2D face scan)
        ↓
Run trained model → binary burn mask
        ↓
Color burn pixels RED, keep skin colors everywhere else
        ↓
Cyberware range file → cylindrical → Cartesian 3D points
        ↓
Each 3D point looks up its pixel in the colored texture
        ↓
Apply Rx=-90°, Ry=+90° alignment
        ↓
Output: outputs/<model>/<patient>/scan_burn3d.ply  +  scan_front.png
```

## Install

```powershell
# Activate your environment first
..\seg_env\Scripts\activate

# Install base requirements
python -m pip install open3d pyyaml

# Model-specific (install only what you need):
python -m pip install segmentation-models-pytorch albumentations   # UNet++
python -m pip install transformers==4.40.0                         # SegFormer
python -m pip install git+https://github.com/facebookresearch/segment-anything.git  # MedSAM
```

## How to Run

**Step 1 — Edit config.yaml** to set your paths and choose your model (see below).

**Step 2 — Run:**
```powershell
python burn_3d_pipeline.py --config config.yaml
```

That's it. No other arguments needed.

---

## config.yaml — Model Selection

Open `config.yaml` and uncomment ONE model block, keep the rest commented.

### UNet++ (simplest — single .pth file)
```yaml
model:  unetpp
ckpt:   "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/unetpp/best.pth"
```

### SegFormer (ckpt is a FOLDER, not a file)
```yaml
model:  segformer
ckpt:   "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/segformer"
```

### MedSAM (needs base weights + your fine-tuned weights)
```yaml
model:      medsam
ckpt:       "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/medsam/best.pth"
base_ckpt:  "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/medsam/medsam_vit_b.pth"
```

### SAM2 (needs base weights + fine-tuned weights + repo path)
```yaml
model:      sam2
ckpt:       "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/sam2/best.pth"
base_ckpt:  "D:/NahidW/Coding/5.face_burn_segmentation/checkpoints/sam2/sam2_hiera_large.pt"
sam2_repo:  "D:/NahidW/Coding/5.face_burn_segmentation/segment-anything-2"
```

---

## Output Structure

Outputs are organized by model and patient automatically:

```
D:/NahidW/Dataset/3d_scans/burn_3d/
  unetpp/
    pat1/
      pat1day0C_burn3d.ply        ← 3D point cloud (burn = red, skin = original)
      pat1day0C_front.png         ← 2D front-view snapshot
      pat1day0C_burn_mask.png     ← binary burn mask       (with save_mask: true)
      pat1day0C_burn_overlay.png  ← original + burn highlighted (with save_mask: true)
      pat1day0C_burn_texture.png  ← colored texture before wrapping (with save_mask: true)
  segformer/
    pat1/
      pat1day0C_burn3d.ply
      ...
  sam2/
    pat1/
      pat1day0C_burn3d.ply
      ...
```

---

## Optional config.yaml Settings

| Key | Default | Description |
|-----|---------|-------------|
| `threshold` | `0.5` | Burn/normal cutoff  |
| `fine_yaw` | `0.0` | Yaw correction in degrees if face isn't centered in the PLY |
| `save_mask` | `false` | Also save 2D mask, overlay, and colored texture images |
| `show` | `false` | Open Open3D 3D viewer after processing |

---

## View the PLY Output

**MeshLab** (recommended, free): File → Import Mesh → open the .ply

**CloudCompare** (free): File → Open → select .ply

**Open3D quick view:**
```powershell
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('pat1day0C_burn3d.ply')])"
```

---

## Switching Models

Just edit `config.yaml` — comment out the current model block, uncomment the new one. Run the same command. Output goes to a separate subfolder automatically so results from different models never overwrite each other.
