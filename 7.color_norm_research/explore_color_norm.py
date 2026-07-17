#!/usr/bin/env python3
"""
explore_color_norm.py  [v2 — reference-based]
==============================================
Research script for 5.color_norm_research/ — NOT part of the main pipeline.

PURPOSE:
  Take D00_A as the fixed reference. Normalize every later scan so that
  normal (non-burn) skin matches D00_A's color. After correction, any
  remaining color difference in the burn area is REAL tissue change —
  not a lighting/camera artifact.

METHODS TESTED:
  01  Reinhard LAB (full image)        — shift+scale all LAB channels to ref mean/std
  02  Reinhard LAB (skin region)       — same but stats from detected skin pixels only
  03  Reinhard LAB (non-burn skin) ★   — skin region MINUS burn polygon (most accurate)
  04  Histogram match L* only          — match brightness curve, leave a*/b* alone
  05  Histogram match LAB (all)        — match full LAB histogram per channel
  06  Linear scale RGB (skin mean)     — scale R,G,B so skin mean matches reference
  07  Reinhard HSV (skin region)       — reinhard in HSV space
  08  Retinex + Reinhard LAB           — remove illumination first, then match

OUTPUT per scan:
  <scan_dir>/color_norm_exploration/ref_<ref_name>/
    <method>_rgb.png         corrected RGB image
    <method>_astar.png       a* redness channel after correction
    <method>_diff.png        what changed vs original
    comparison_grid.png      all methods: RGB + a* side by side
    burn_region_comparison.png  zoom of burn area: ref | original | each method
    stats_report.txt         ranked by normal-skin colour match quality

USAGE:
  python explore_color_norm.py --ref ".../PAT01/D00/PAT01_D00_A" --scan ".../PAT01/D14/PAT01_D14_A"
  python explore_color_norm.py --ref ".../PAT01/D00/PAT01_D00_A" --dataset "..." --patient PAT01 --variant A
  python explore_color_norm.py --ref ".../PAT01/D00/PAT01_D00_A" --dataset "..." --patient PAT01 --scans PAT01_D14_A PAT01_M03_A
"""

import re, json, argparse
import numpy as np
import cv2
import tifffile
from pathlib import Path
from skimage.exposure import match_histograms


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_tif(path):
    raw = tifffile.imread(str(path))
    if raw.ndim == 3 and raw.shape[0] < 10: raw = raw[0]
    raw = raw.astype(np.float32)
    lo, hi = raw.min(), raw.max()
    raw = ((raw-lo)/(hi-lo)*255).astype(np.uint8) if hi > lo else raw.astype(np.uint8)
    if raw.ndim == 2: raw = np.stack([raw]*3, -1)
    return raw[...,:3].copy()

def load_burn_mask(scan_dir):
    p = scan_dir / f"{scan_dir.name}_burn_polygons.json"
    if not p.exists(): return None
    data = json.loads(p.read_text())
    h, w = data["image_size"]["height"], data["image_size"]["width"]
    mask = np.zeros((h, w), bool)
    for r in data.get("regions", []):
        cnt = np.array(r["polygon"], np.int32).reshape(-1,1,2)
        m = np.zeros((h,w), np.uint8)
        cv2.fillPoly(m, [cnt], 1)
        mask |= m.astype(bool)
    return mask

def save_rgb(path, img):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def colormap(arr, cmap=cv2.COLORMAP_INFERNO):
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    n = np.clip((arr-p1)/max(p99-p1,1)*255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(n, cmap), cv2.COLOR_BGR2RGB)

def diff_map(a, b):
    d = np.abs(a.astype(float)-b.astype(float)).mean(-1)
    return colormap(np.clip(d*3,0,255), cv2.COLORMAP_HOT)


# ── MASKS ─────────────────────────────────────────────────────────────────────

def skin_mask(img):
    ycr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y,Cr,Cb = ycr[...,0], ycr[...,1], ycr[...,2]
    return (Y>30)&(Cr>133)&(Cr<177)&(Cb>77)&(Cb<127)

def bg_mask(img, thresh=20):
    return img.mean(-1) < thresh


# ── CORE: REINHARD TRANSFER ───────────────────────────────────────────────────

def reinhard(src, ref, src_mask=None, ref_mask=None):
    """Reinhard shift+scale per channel. src/ref are HxWx3 float arrays."""
    out = src.astype(float).copy()
    for c in range(3):
        sv = src[...,c][src_mask] if src_mask is not None else src[...,c].ravel()
        rv = ref[...,c][ref_mask] if ref_mask is not None else ref[...,c].ravel()
        if len(sv)<10 or len(rv)<10: continue
        sm,ss = sv.mean(), sv.std()+1e-6
        rm,rs = rv.mean(), rv.std()+1e-6
        out[...,c] = (src[...,c] - sm)*(rs/ss) + rm
    return np.clip(out,0,255).astype(np.uint8)

def retinex(img, sigma=80):
    f = img.astype(float)+1
    out = np.zeros_like(f)
    for c in range(3):
        ill = cv2.GaussianBlur(f[...,c],(0,0),sigma)
        out[...,c] = np.log(f[...,c]) - np.log(ill+1)
    out -= out.min()
    if out.max()>0: out = out/out.max()*255
    return out.astype(np.uint8)


# ── 8 METHODS ─────────────────────────────────────────────────────────────────

def run_all_methods(src_rgb, ref_rgb, src_burn, ref_burn):
    src_skin  = skin_mask(src_rgb) & ~bg_mask(src_rgb)
    ref_skin  = skin_mask(ref_rgb) & ~bg_mask(ref_rgb)
    src_norm  = src_skin & (~src_burn if src_burn is not None else np.ones(src_skin.shape,bool))
    ref_norm  = ref_skin & (~ref_burn if ref_burn is not None else np.ones(ref_skin.shape,bool))

    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(float)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(float)
    src_hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV).astype(float)
    ref_hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV).astype(float)

    results = []

    def add(key, label, img):
        results.append({"key":key,"label":label,"img":img})

    # 01
    add("01_reinhard_lab_full", "Reinhard LAB (full image)",
        cv2.cvtColor(reinhard(src_lab,ref_lab), cv2.COLOR_LAB2RGB))

    # 02
    add("02_reinhard_lab_skin", "Reinhard LAB (skin region)",
        cv2.cvtColor(reinhard(src_lab,ref_lab,src_skin,ref_skin), cv2.COLOR_LAB2RGB))

    # 03
    if src_norm.sum()>100 and ref_norm.sum()>100:
        add("03_reinhard_lab_nonburn", "Reinhard LAB (non-burn skin) ★",
            cv2.cvtColor(reinhard(src_lab,ref_lab,src_norm,ref_norm), cv2.COLOR_LAB2RGB))

    # 04
    sL = src_lab[...,0:1]; rL = ref_lab[...,0:1]
    mL = match_histograms(sL, rL)
    tmp = src_lab.copy(); tmp[...,0] = mL[...,0]
    add("04_histmatch_L_only", "Histogram match L* only",
        cv2.cvtColor(np.clip(tmp,0,255).astype(np.uint8), cv2.COLOR_LAB2RGB))

    # 05
    mlab = match_histograms(src_lab, ref_lab, channel_axis=-1)
    add("05_histmatch_lab_all", "Histogram match LAB all channels",
        cv2.cvtColor(np.clip(mlab,0,255).astype(np.uint8), cv2.COLOR_LAB2RGB))

    # 06
    out = src_rgb.astype(float).copy()
    for c in range(3):
        sv = src_rgb.astype(float)[...,c][src_skin]
        rv = ref_rgb.astype(float)[...,c][ref_skin]
        if len(sv)>10 and len(rv)>10 and sv.mean()>0:
            out[...,c] = np.clip(out[...,c]*(rv.mean()/sv.mean()),0,255)
    add("06_linear_rgb_skin", "Linear scale RGB (skin mean)",
        out.astype(np.uint8))

    # 07
    add("07_reinhard_hsv_skin", "Reinhard HSV (skin region)",
        cv2.cvtColor(reinhard(src_hsv,ref_hsv,src_skin,ref_skin).astype(np.uint8),
                     cv2.COLOR_HSV2RGB))

    # 08
    src_ret = retinex(src_rgb); ref_ret = retinex(ref_rgb)
    src_rl  = cv2.cvtColor(src_ret, cv2.COLOR_RGB2LAB).astype(float)
    ref_rl  = cv2.cvtColor(ref_ret, cv2.COLOR_RGB2LAB).astype(float)
    src_rs  = skin_mask(src_ret) & ~bg_mask(src_rgb)
    ref_rs  = skin_mask(ref_ret) & ~bg_mask(ref_rgb)
    m = None if src_rs.sum()<10 else src_rs
    n = None if ref_rs.sum()<10 else ref_rs
    add("08_retinex_reinhard", "Retinex + Reinhard LAB",
        cv2.cvtColor(reinhard(src_rl,ref_rl,m,n), cv2.COLOR_LAB2RGB))

    return results, src_skin, ref_skin, src_norm, ref_norm


# ── STATS ─────────────────────────────────────────────────────────────────────

def compute_stats(img, ref_rgb, src_skin, ref_skin, src_burn, ref_burn):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(float)
    src_lab_orig = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)

    ref_a  = ref_lab[...,1][ref_skin].mean() if ref_skin.sum()>0 else float('nan')
    corr_a = img_lab[...,1][skin_mask(img)&~bg_mask(img)].mean() if (skin_mask(img)&~bg_mask(img)).sum()>0 else float('nan')
    delta  = abs(corr_a - ref_a)

    burn_ref  = ref_lab[...,1][ref_burn].mean() if ref_burn is not None and ref_burn.sum()>0 else float('nan')
    burn_corr = img_lab[...,1][src_burn].mean() if src_burn is not None and src_burn.sum()>0 else float('nan')

    return {"skin_delta": delta, "burn_a_ref": burn_ref, "burn_a_corr": burn_corr,
            "skin_a_corr": corr_a, "skin_a_ref": ref_a}


# ── GRIDS ─────────────────────────────────────────────────────────────────────

def build_grid(ref_rgb, src_rgb, results, out_dir):
    TH, TW, pad, LH = 120, 120, 4, 28
    items = [("REF", "Reference D00", ref_rgb),
             ("ORIG","Original (uncorrected)", src_rgb)] + \
            [(r["key"], r["label"], r["img"]) for r in results]
    nc = 5
    nr = (len(items)+nc-1)//nc
    W = pad + nc*(TW*2+pad)
    H = pad + nr*(TH+LH+pad)
    canvas = np.ones((H,W,3),np.uint8)*18
    for i,(key,lbl,img) in enumerate(items):
        row,col = divmod(i,nc)
        x = pad+col*(TW*2+pad); y = pad+row*(TH+LH+pad)
        canvas[y:y+TH, x:x+TW] = cv2.resize(img,(TW,TH))
        ast = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[...,1].astype(float)
        canvas[y:y+TH, x+TW:x+TW*2] = cv2.resize(colormap(ast),(TW,TH))
        cv2.putText(canvas, lbl[:34], (x+2,y+TH+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, (160,160,160), 1)
    hdr = np.ones((28,W,3),np.uint8)*10
    cv2.putText(hdr,"LEFT=RGB  RIGHT=a*(redness burn discriminator)  skin_da*=normal skin error",
                (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.37,(150,150,150),1)
    cv2.imwrite(str(out_dir/"comparison_grid.png"),
                cv2.cvtColor(np.vstack([hdr,canvas]),cv2.COLOR_RGB2BGR))


def build_burn_crop(ref_rgb, src_rgb, results, ref_burn, src_burn, out_dir):
    H, W = src_rgb.shape[:2]
    if src_burn is not None and src_burn.sum()>0:
        ys,xs = np.where(src_burn)
        y0,y1,x0,x1 = max(ys.min()-15,0),min(ys.max()+15,H),max(xs.min()-15,0),min(xs.max()+15,W)
    else:
        y0,y1,x0,x1 = H//4, 3*H//4, W//2, W
    if ref_burn is not None and ref_burn.sum()>0:
        rys,rxs = np.where(ref_burn)
        ry0,ry1,rx0,rx1 = max(rys.min()-15,0),min(rys.max()+15,H),max(rxs.min()-15,0),min(rxs.max()+15,W)
    else:
        ry0,ry1,rx0,rx1 = y0,y1,x0,x1

    BH = 160
    def crop(img,a,b,c,d):
        cr = img[a:b,c:d]
        sc = BH/max(cr.shape[0],1)
        return cv2.resize(cr,(max(int(cr.shape[1]*sc),1),BH))

    items = [("Ref D00", crop(ref_rgb,ry0,ry1,rx0,rx1)),
             ("Original", crop(src_rgb,y0,y1,x0,x1))] + \
            [(r["label"][:18], crop(r["img"],y0,y1,x0,x1)) for r in results]

    maxbw = max(t[1].shape[1] for t in items)
    pad = 4; LH = 32
    SW = len(items)*(maxbw+pad)+pad

    def make_strip(items, fn):
        s = np.ones((BH+LH+pad*2, SW, 3), np.uint8)*18
        x = pad
        for name,tile in items:
            t = fn(tile)
            s[pad:pad+BH, x:x+t.shape[1]] = t
            cv2.putText(s, name, (x+2,pad+BH+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.29,(160,160,160),1)
            x += maxbw+pad
        return s

    rgb_strip  = make_strip(items, lambda t: t)
    ast_strip  = make_strip(items, lambda t:
                    colormap(cv2.cvtColor(t, cv2.COLOR_RGB2LAB)[...,1].astype(float)))

    hdr1 = np.ones((24,SW,3),np.uint8)*10
    hdr2 = np.ones((24,SW,3),np.uint8)*10
    cv2.putText(hdr1,"BURN REGION — RGB: reference | original | each correction",
                (6,16),cv2.FONT_HERSHEY_SIMPLEX,0.37,(150,150,150),1)
    cv2.putText(hdr2,"BURN REGION — a* redness (brighter = more red/burn tissue)",
                (6,16),cv2.FONT_HERSHEY_SIMPLEX,0.37,(150,150,150),1)

    cv2.imwrite(str(out_dir/"burn_region_comparison.png"),
                cv2.cvtColor(np.vstack([hdr1,rgb_strip,hdr2,ast_strip]),cv2.COLOR_RGB2BGR))


# ── PROCESS ONE PAIR ──────────────────────────────────────────────────────────

def process_pair(ref_dir, src_dir):
    ref_dir, src_dir = Path(ref_dir), Path(src_dir)
    if src_dir == ref_dir:
        print(f"  – Skipping {src_dir.name} (it IS the reference)"); return

    ref_tif = ref_dir/f"{ref_dir.name}.tif"
    src_tif = src_dir/f"{src_dir.name}.tif"
    if not ref_tif.exists(): raise FileNotFoundError(str(ref_tif))
    if not src_tif.exists(): raise FileNotFoundError(str(src_tif))

    out_dir = src_dir/"color_norm_exploration"/f"ref_{ref_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{src_dir.name}] vs [{ref_dir.name}]  →  {out_dir}")

    ref_rgb   = load_tif(ref_tif)
    src_rgb   = load_tif(src_tif)
    ref_burn  = load_burn_mask(ref_dir)
    src_burn  = load_burn_mask(src_dir)

    results, src_skin, ref_skin, src_norm, ref_norm = \
        run_all_methods(src_rgb, ref_rgb, src_burn, ref_burn)

    ref_lab_orig = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(float)
    src_lab_orig = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(float)

    ref_skin_a = ref_lab_orig[...,1][ref_skin].mean() if ref_skin.sum()>0 else float('nan')
    src_skin_a = src_lab_orig[...,1][src_skin].mean() if src_skin.sum()>0 else float('nan')
    ref_burn_a = ref_lab_orig[...,1][ref_burn].mean() if ref_burn is not None and ref_burn.sum()>0 else float('nan')
    src_burn_a = src_lab_orig[...,1][src_burn].mean() if src_burn is not None and src_burn.sum()>0 else float('nan')

    lines = [
        f"Reference-based colour normalization report",
        f"Reference : {ref_dir.name}",
        f"Source    : {src_dir.name}",
        "="*60, "",
        f"BEFORE ANY CORRECTION:",
        f"  Normal skin a* — ref={ref_skin_a:.1f}  src={src_skin_a:.1f}  Δ={abs(src_skin_a-ref_skin_a):.1f}",
        f"  Burn region a* — ref={ref_burn_a:.1f}  src={src_burn_a:.1f}",
        "",
        "METRIC: skin_Δa* = |corrected_skin_a* − ref_skin_a*|",
        "  Lower is better — means normal skin now matches reference.",
        "  burn_a* is what you compare clinically across timepoints.",
        "",
    ]

    for r in results:
        print(f"    {r['label'][:40]} ...", end=" ", flush=True)
        corr_lab = cv2.cvtColor(r["img"], cv2.COLOR_RGB2LAB).astype(float)
        corr_sk  = skin_mask(r["img"]) & ~bg_mask(r["img"])
        corr_skin_a = corr_lab[...,1][corr_sk].mean() if corr_sk.sum()>0 else float('nan')
        corr_burn_a = corr_lab[...,1][src_burn].mean() if src_burn is not None and src_burn.sum()>0 else float('nan')
        skin_delta  = abs(corr_skin_a - ref_skin_a)

        r["skin_delta"] = skin_delta
        r["burn_a"]     = corr_burn_a

        save_rgb(out_dir/f"{r['key']}_rgb.png",   r["img"])
        save_rgb(out_dir/f"{r['key']}_astar.png", colormap(corr_lab[...,1]))
        save_rgb(out_dir/f"{r['key']}_diff.png",  diff_map(src_rgb, r["img"]))

        lines += [
            f"[{r['key']}] {r['label']}",
            f"  Normal skin a* after: {corr_skin_a:.1f}  (ref={ref_skin_a:.1f}  Δ={skin_delta:.1f})  ← lower=better",
            f"  Burn region a* after: {corr_burn_a:.1f}",
            "",
        ]
        print(f"skin_Δa*={skin_delta:.1f}  burn_a*={corr_burn_a:.1f}")

    ranked = sorted(results, key=lambda r: r.get("skin_delta",999))
    lines += ["","RANKING by skin colour match quality (lower Δ = better):",""]
    for i,r in enumerate(ranked,1):
        lines.append(f"  {i}. {r['label']:<45}  skin_Δa*={r.get('skin_delta',float('nan')):.1f}  burn_a*={r.get('burn_a',float('nan')):.1f}")

    (out_dir/"stats_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("    Building grids ...", end=" ", flush=True)
    build_grid(ref_rgb, src_rgb, results, out_dir)
    build_burn_crop(ref_rgb, src_rgb, results, ref_burn, src_burn, out_dir)
    print("done")
    print(f"    → {len(results)} methods saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

SCAN_RE = re.compile(r"^(?P<patient>PAT\d+)_(?P<timepoint>[DM]\d+)_(?P<variant>[A-Z][A-Z0-9]?)$",re.IGNORECASE)
def tp_days(tp):
    m=re.match(r'([DM])(\d+)',tp.upper())
    return int(m.group(2)) if m and m.group(1)=='D' else int(m.group(2))*30 if m else 9999

def find_scans(dataset, patient, variant=None, names=None):
    pat = Path(dataset)/patient
    dirs = sorted([p for p in pat.glob("*/*") if p.is_dir() and SCAN_RE.match(p.name)])
    if names: return [d for d in dirs if d.name in names]
    if variant: dirs=[d for d in dirs if SCAN_RE.match(d.name).group("variant").upper()==variant.upper()]
    return sorted(dirs, key=lambda d: tp_days(SCAN_RE.match(d.name).group("timepoint")))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Reference scan dir (e.g. .../PAT01_D00_A)")
    p.add_argument("--scan",    default=None, help="Single scan to compare")
    p.add_argument("--dataset", default=None)
    p.add_argument("--patient", default=None)
    p.add_argument("--scans",   nargs="+", default=None)
    p.add_argument("--variant", default=None, help="Compare all scans of this variant letter")
    args = p.parse_args()

    if args.scan:
        process_pair(args.ref, args.scan)
    elif args.dataset and args.patient:
        dirs = find_scans(args.dataset, args.patient, args.variant, args.scans)
        print(f"Reference: {Path(args.ref).name}")
        print(f"Comparing {len(dirs)} scan(s)\n")
        for d in dirs:
            try: process_pair(args.ref, d)
            except Exception as e: print(f"  ✗ {d.name}: {e}")
    else:
        p.error("Provide --scan, or --dataset+--patient (with optional --scans/--variant)")

if __name__=="__main__":
    main()