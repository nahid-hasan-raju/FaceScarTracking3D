"""
Test any model on a single image or folder.

For SAM2/MedSAM:
  - LEFT CLICK  = add polygon point (rough ROI, guides where the burn area is)
  - RIGHT CLICK = undo last point
  - ENTER       = confirm polygon and run model
  - R           = reset polygon

IMPORTANT (why this pipeline works the way it does):
  The model runs on the FULL image (resized to 1024x1024), exactly like it
  was trained. Cropping to a small ROI and upscaling that crop to 1024x1024
  makes the burn area look artificially zoomed-in and removes surrounding
  tissue context -- that is out of the training distribution and tanks
  accuracy. So we do NOT crop.

  Instead your polygon is used two ways:
    1) Its bounding box is passed as the SAM/MedSAM box prompt (localization
       hint), same as the original working version.
    2) After the model predicts on the full image, the result is clipped to
       your polygon DILATED by a margin -- so the output can't appear in an
       unrelated part of the image, but you aren't penalized for imprecise
       clicking (SAM/MedSAM still get to refine the exact edge).

For UNet++/SegFormer: fully automatic, no interaction needed.

RUN:
  python test_single_image.py --input path/to/image.jpg --model sam2
  python test_single_image.py --input path/to/folder   --model sam2
  python test_single_image.py --input path/to/folder   --model unetpp
  python test_single_image.py --input path/to/folder   --model segformer

  Optional: --margin 0.15  (buffer around your polygon before clipping, as a
                             fraction of the polygon bbox diagonal. Higher =
                             more forgiving of a rough click shape. Default
                             0.15.)
"""
import os, sys, argparse
BASE     = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = os.path.join(BASE, 'segment-anything-2')
sys.path.insert(0, BASE)
sys.path.insert(0, SAM2_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE     = 1024
IMG_EXTS     = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
MARGIN_RATIO = 0.15   # overridden by --margin
THRESHOLD    = None   # None = auto (Otsu per-image); overridden by --threshold

# ── Polygon drawing state ──────────────────────────────────────────
points   = []
img_disp = None
scale    = 1.0


def mouse_callback(event, x, y, flags, param):
    global points, img_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()


def redraw():
    global img_disp
    tmp = img_disp.copy()
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        overlay = tmp.copy()
        cv2.fillPoly(overlay, [pts], (0, 120, 255))
        cv2.addWeighted(overlay, 0.25, tmp, 0.75, 0, tmp)
        cv2.polylines(tmp, [pts], isClosed=True,
                      color=(0, 200, 255), thickness=2)
    for i, pt in enumerate(points):
        cv2.circle(tmp, pt, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(tmp, points[i-1], pt, (0, 255, 0), 1)
    cv2.putText(tmp, 'LEFT=add point  RIGHT=undo  ENTER=confirm  R=reset',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(tmp, 'LEFT=add point  RIGHT=undo  ENTER=confirm  R=reset',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    cv2.putText(tmp, f'Points: {len(points)}  (need at least 3)',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)
    cv2.putText(tmp, f'Points: {len(points)}  (need at least 3)',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,150,0), 1)
    cv2.imshow('Draw polygon around burn area', tmp)


def get_user_polygon(img_np, img_size, margin_ratio=0.15, margin_px_min=15):
    """
    Opens a window. User clicks multiple points to form a rough polygon ROI.

    Returns:
        box           : (1,4) bbox of the polygon, scaled to img_size coords
                        -- passed to SAM2/MedSAM as the box prompt.
        full_res_mask : HxW uint8 (0/255) DILATED polygon mask at the
                        ORIGINAL image resolution, used to clip the model's
                        full-image prediction after inference.
    """
    global points, img_disp, scale
    points   = []
    h, w     = img_np.shape[:2]
    scale    = min(900/w, 750/h, 1.0)
    disp_w   = int(w * scale)
    disp_h   = int(h * scale)
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_disp = cv2.resize(img_bgr, (disp_w, disp_h))

    win = 'Draw polygon around burn area'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)
    redraw()

    print("  LEFT CLICK = add point | RIGHT CLICK = undo | ENTER = confirm | R = reset")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 3:   # ENTER
            break
        elif key in (ord('r'), ord('R')):
            points = []
            redraw()
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

    if len(points) < 3:
        print("  No polygon drawn — using full image")
        return (np.array([[0, 0, img_size, img_size]], dtype=np.float32), None)

    orig_pts = np.array([(int(x / scale), int(y / scale))
                         for x, y in points], dtype=np.int32)

    x1, y1 = orig_pts[:, 0].min(), orig_pts[:, 1].min()
    x2, y2 = orig_pts[:, 0].max(), orig_pts[:, 1].max()

    box = np.array([[
        x1 * img_size / w,
        y1 * img_size / h,
        x2 * img_size / w,
        y2 * img_size / h,
    ]], dtype=np.float32)

    # rasterize the polygon at full resolution, then dilate by a margin
    # (rough click shape -> forgiving buffer, not a pixel-accurate boundary)
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(raw_mask, [orig_pts], 255)

    diag = float(np.hypot(x2 - x1, y2 - y1))
    margin_px = max(margin_px_min, int(diag * margin_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    full_res_mask = cv2.dilate(raw_mask, kernel)

    print(f"  Polygon: {len(points)} points  BBox: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
          f"  Margin: {margin_px}px")
    return box, full_res_mask


# ── SAM2 ──────────────────────────────────────────────────────────
def load_sam2():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg   = os.path.join(SAM2_DIR, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
    ckpt  = os.path.join(BASE, 'checkpoints', 'sam2', 'sam2_hiera_large.pt')
    ft    = os.path.join(BASE, 'checkpoints', 'sam2', 'best.pth')
    model = build_sam2(cfg, ckpt, device=DEVICE)
    model.load_state_dict(torch.load(ft, map_location=DEVICE))
    model.eval()
    return model, SAM2ImagePredictor(model)


PERCENTILE = 85   # overridden by --percentile; keep only the top (100-PERCENTILE)% most confident ROI pixels
MIN_BLOB_AREA = 150  # overridden by --min-blob; remove speckle noise smaller than this many pixels
COLOR_WEIGHT = 0.5   # overridden by --color-weight; 0=pure model confidence, 1=pure color/redness


def redness_score(img_np, full_res_mask):
    """
    Scores each pixel by how "dark red" it is -- reddish hue, saturated,
    AND darker (lower brightness) -- since that combination is your actual
    clinical signal: a bright pink flush and a dark burned patch can have
    similar simple R-vs-(G+B) redness, but only the burn should also be
    darker/more saturated. Rescaled to 0-255 using percentiles WITHIN your
    ROI only.
    """
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32)   # 0-179 in OpenCV
    s = hsv[:, :, 1].astype(np.float32)   # 0-255
    v = hsv[:, :, 2].astype(np.float32)   # 0-255 (brightness)

    # how close the hue is to red (red sits at both 0 and 179 in OpenCV's scale)
    hue_dist = np.minimum(h, 180 - h)          # 0 = pure red .. 90 = furthest from red
    red_proximity = np.clip(1.0 - hue_dist / 90.0, 0, 1)

    saturation = s / 255.0
    darkness   = 1.0 - v / 255.0                # darker pixels score higher

    idx = red_proximity * saturation * darkness

    roi_vals = idx[full_res_mask > 0] if full_res_mask is not None else idx.flatten()
    if roi_vals.size < 50:
        return np.zeros_like(idx, dtype=np.uint8)
    lo, hi = np.percentile(roi_vals, [2, 98])
    if hi <= lo:
        return np.zeros_like(idx, dtype=np.uint8)
    scaled = np.clip((idx - lo) / (hi - lo), 0, 1) * 255
    return scaled.astype(np.uint8)


def combined_score(prob_u8, img_np, full_res_mask):
    """
    Blends model confidence with the color-based redness score:
        combined = COLOR_WEIGHT * redness + (1 - COLOR_WEIGHT) * model_prob
    COLOR_WEIGHT=0 -> ignore color, pure model output (previous behavior).
    COLOR_WEIGHT=1 -> ignore model, pure color/redness.
    """
    if COLOR_WEIGHT <= 0:
        return prob_u8
    red = redness_score(img_np, full_res_mask)
    blended = COLOR_WEIGHT * red.astype(np.float32) + (1 - COLOR_WEIGHT) * prob_u8.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def choose_threshold(score_u8, full_res_mask):
    """
    Picks a cutoff automatically instead of us guessing a fixed number.

    Uses a PERCENTILE cutoff on the score histogram WITHIN your ROI only:
    e.g. PERCENTILE=85 keeps only the top 15% of the score inside your ROI.

    Falls back to the manual THRESHOLD if it was set explicitly via
    --threshold, or to a flat 128 if there isn't enough ROI to work with.
    """
    if THRESHOLD is not None:
        return int(THRESHOLD * 255)
    roi_vals = score_u8[full_res_mask > 0] if full_res_mask is not None else score_u8.flatten()
    if roi_vals.size < 50:
        return 128

    # print the actual distribution so you can SEE whether there's a real
    # gradient to cut along, or whether the values are saturated/bunched up
    checkpoints = [50, 75, 85, 90, 95, 97, 99]
    vals = np.percentile(roi_vals, checkpoints)
    stats_str = "  ".join(f"p{c}={v:.0f}" for c, v in zip(checkpoints, vals))
    print(f"  [roi score distribution /255] min={roi_vals.min()} max={roi_vals.max()}  {stats_str}")

    return int(np.percentile(roi_vals, PERCENTILE))


def clean_speckles(pred, min_area=150):
    """
    Removes small isolated blobs (noise specks) and keeps only the larger,
    spatially coherent regions -- a real burn area should be one or a few
    connected patches, not a scatter of single-pixel dots.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (pred > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(pred)
    for i in range(1, n_labels):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def predict_sam2(model, predictor, img_np, prompt):
    """
    Runs on the FULL image (not cropped) so the model sees the scale/context
    it was trained on. prompt = (box, full_res_mask) from get_user_polygon.

    IMPORTANT: we do NOT pass your polygon as the `masks=` dense prompt.
    In SAM2, `masks=` is a foreground PRIOR -- it tells the decoder "trust
    this exact shape as the answer," not "search inside this region." Your
    ROI is only a search-space hint, so it belongs in `boxes=` only. The
    decoder must be free to look at actual image content within the box and
    decide what is/isn't burned -- that's the whole point of using a
    detection model instead of just returning your drawn shape back to you.
    """
    box, full_res_mask = prompt
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    box_t = torch.from_numpy(box).to(DEVICE)

    with torch.no_grad():
        predictor.set_image(img_r)
        feats    = predictor._features
        img_emb  = feats['image_embed']
        high_res = feats['high_res_feats']

        # masks=None: let the decoder discriminate burn vs. normal skin from
        # image features -- do NOT bias it with the ROI shape itself
        sparse, dense = model.sam_prompt_encoder(points=None, boxes=box_t, masks=None)
        out = model.sam_mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        prob   = torch.sigmoid(logits[0, 0]).cpu().numpy()
        prob_u8 = (prob * 255).astype(np.uint8)
        score = combined_score(prob_u8, img_np, full_res_mask)
        thr_val = choose_threshold(score, full_res_mask)
        pred = (score > thr_val).astype(np.uint8) * 255
        pred = clean_speckles(pred, MIN_BLOB_AREA)
        print(f"  [sam2] color_weight={COLOR_WEIGHT}  percentile-{PERCENTILE} threshold: {thr_val}/255 ({thr_val/255:.2f})")

    return clip_to_polygon(pred, full_res_mask), clip_to_polygon(prob_u8, full_res_mask)


# ── MedSAM ────────────────────────────────────────────────────────
def load_medsam():
    from segment_anything import sam_model_registry
    ckpt_in = os.path.join(BASE, 'checkpoints', 'medsam', 'medsam_vit_b.pth')
    ckpt_ft = os.path.join(BASE, 'checkpoints', 'medsam', 'best.pth')
    model = sam_model_registry['vit_b'](checkpoint=ckpt_in).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_ft, map_location=DEVICE))
    model.eval()
    return model, None


def predict_medsam(model, aux, img_np, prompt):
    box, full_res_mask = prompt
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img_r).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.
    box_t = torch.from_numpy(box).to(DEVICE)
    with torch.no_grad():
        img_emb       = model.image_encoder(img_t)
        sparse, dense = model.prompt_encoder(points=None, boxes=box_t.unsqueeze(1), masks=None)
        logits, _     = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        prob   = torch.sigmoid(logits[0, 0]).cpu().numpy()
        prob_u8 = (prob * 255).astype(np.uint8)
        score = combined_score(prob_u8, img_np, full_res_mask)
        thr_val = choose_threshold(score, full_res_mask)
        pred = (score > thr_val).astype(np.uint8) * 255
        pred = clean_speckles(pred, MIN_BLOB_AREA)
        print(f"  [medsam] color_weight={COLOR_WEIGHT}  percentile-{PERCENTILE} threshold: {thr_val}/255 ({thr_val/255:.2f})")

    return clip_to_polygon(pred, full_res_mask), clip_to_polygon(prob_u8, full_res_mask)


def clip_to_polygon(pred, full_res_mask):
    """Zero out anything outside the (dilated) polygon ROI."""
    if full_res_mask is None:
        return pred
    return cv2.bitwise_and(pred, pred, mask=(full_res_mask > 0).astype(np.uint8))


# ── UNet++ ────────────────────────────────────────────────────────
def load_unetpp():
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4', encoder_weights=None,
        in_channels=3, classes=1, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(BASE, 'checkpoints', 'unetpp', 'best.pth'), map_location=DEVICE))
    model.eval()
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    return model, transform


def predict_unetpp(model, transform, img_np, prompt=None):
    h, w = img_np.shape[:2]
    inp  = transform(image=img_np)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)[0,0]).cpu().numpy()
    prob = cv2.resize(prob, (w, h))
    return (prob > 0.5).astype(np.uint8) * 255


# ── SegFormer ─────────────────────────────────────────────────────
def load_segformer():
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    model = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(BASE, 'checkpoints', 'segformer')).to(DEVICE)
    model.eval()
    return model, SegformerImageProcessor()


def predict_segformer(model, processor, img_np, prompt=None):
    h, w   = img_np.shape[:2]
    inputs = processor(images=Image.fromarray(img_np), return_tensors='pt')
    pv     = inputs['pixel_values'].to(DEVICE)
    with torch.no_grad():
        out    = model(pixel_values=pv)
        logits = F.interpolate(out.logits, size=(h,w), mode='bilinear', align_corners=False)
        return logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255


# ── Config ────────────────────────────────────────────────────────
MODELS = {
    'sam2':      (load_sam2,      predict_sam2,      True),
    'medsam':    (load_medsam,    predict_medsam,    True),
    'unetpp':    (load_unetpp,    predict_unetpp,    False),
    'segformer': (load_segformer, predict_segformer, False),
}


def make_overlay(img_np, pred):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    color   = np.zeros_like(img_bgr)
    color[pred > 128] = (0, 0, 255)
    overlay = cv2.addWeighted(img_bgr, 0.6, color, 0.4, 0)
    cnts, _ = cv2.findContours((pred > 128).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)
    return overlay


def process_image(img_path, model_name, model, aux, needs_prompt, out_dir):
    img_np = np.array(Image.open(img_path).convert('RGB'))
    stem   = os.path.splitext(os.path.basename(img_path))[0]
    _, predict_fn, _ = MODELS[model_name]

    prompt = None
    if needs_prompt:
        print(f"\n{os.path.basename(img_path)}")
        prompt = get_user_polygon(img_np, IMG_SIZE, margin_ratio=MARGIN_RATIO)

    result = predict_fn(model, aux, img_np, prompt)
    if isinstance(result, tuple):
        pred, prob = result
        # heatmap: bright = model very confident "burned". If this is nearly
        # solid white across the whole face rather than concentrated on the
        # actual burn, thresholding won't fix it -- the model itself isn't
        # discriminating. If it shows real gradation (dark normal skin,
        # bright burn) but the binary mask still overshoots, THEN threshold
        # tuning (--threshold 0.7, 0.8, ...) is the right lever.
        heat = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f'{stem}_prob_heatmap.png'), heat)
    else:
        pred = result

    cv2.imwrite(os.path.join(out_dir, f'{stem}_mask.png'), pred)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_overlay.png'),
                make_overlay(img_np, pred))

    burn_pct = 100 * (pred > 128).sum() / pred.size
    print(f"  Burn area: {burn_pct:.2f}%  ->  {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--model',  default='sam2',
                        choices=['sam2','medsam','unetpp','segformer'])
    parser.add_argument('--output', default=None)
    parser.add_argument('--margin', type=float, default=0.15,
                        help='Buffer around your polygon before clipping the '
                             'prediction, as a fraction of the polygon bbox '
                             'diagonal. Higher = more forgiving of a rough '
                             'click shape. Default 0.15.')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Sigmoid probability cutoff for calling a pixel '
                             '"burned". If not set (default), a percentile '
                             'cutoff is used instead (see --percentile). Pass '
                             'a value (e.g. 0.7) to override with a fixed '
                             'threshold.')
    parser.add_argument('--percentile', type=float, default=85,
                        help='When --threshold is not set, keep only the top '
                             '(100-percentile) percent most confident pixels '
                             'inside your ROI. Higher = more selective/strict '
                             '(fewer, more confident pixels kept). Default 85 '
                             '(top 15%%). Try 90-95 if it is still too broad.')
    parser.add_argument('--min-blob', type=int, default=150,
                        help='Remove connected regions smaller than this many '
                             'pixels (speckle/noise cleanup). Default 150.')
    parser.add_argument('--color-weight', type=float, default=0.5,
                        help='How much to weight actual pixel color/redness '
                             'vs. model confidence when scoring each pixel. '
                             '0 = pure model output (old behavior). 1 = pure '
                             'color, ignore the model entirely. Useful when '
                             'the model\'s confidence is saturated/flat and '
                             'percentile tuning alone stops making a '
                             'difference. Default 0.5.')
    args = parser.parse_args()

    global MARGIN_RATIO, THRESHOLD, PERCENTILE, MIN_BLOB_AREA, COLOR_WEIGHT
    MARGIN_RATIO  = args.margin
    THRESHOLD     = args.threshold
    PERCENTILE    = args.percentile
    MIN_BLOB_AREA = args.min_blob
    COLOR_WEIGHT  = args.color_weight

    out_dir = args.output or os.path.join(BASE, 'outputs', 'single_test')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(args.input):
        images = [args.input]
    elif os.path.isdir(args.input):
        images = [os.path.join(args.input, f)
                  for f in sorted(os.listdir(args.input))
                  if f.lower().endswith(IMG_EXTS)
                  and os.path.isfile(os.path.join(args.input, f))]
    else:
        print(f"ERROR: {args.input} not found"); return

    if not images:
        print("No images found"); return

    _, _, needs_prompt = MODELS[args.model]
    print(f"Device  : {DEVICE}")
    print(f"Model   : {args.model}")
    print(f"Images  : {len(images)}")
    if needs_prompt:
        print("NOTE: Draw a polygon around the burn area for each image")
        print("      LEFT=add point | RIGHT=undo | ENTER=confirm | R=reset")
    print(f"Output  : {out_dir}\n")

    load_fn, _, _ = MODELS[args.model]
    model, aux    = load_fn()

    for img_path in images:
        process_image(img_path, args.model, model, aux, needs_prompt, out_dir)

    print(f"\nDone! Results in: {out_dir}")
    print("  *_mask.png    = binary burn mask")
    print("  *_overlay.png = burn area (red) on original image")


if __name__ == '__main__':
    main()

    # ex: python test_model_interactive.py --input "D:\NahidW\Coding\sample_images" --model sam2 --output "D:\NahidW\Coding\5.face_burn_segmentation\outputs\single_test\sam2"