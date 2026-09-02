#!/usr/bin/env python
"""Milestone 1 - Registration QC: align each Spectralis IR localizer to the advRPE 6x6
SubRPE en-face and prove (visually + by metric) the anatomy matches BEFORE any GA measurement.

Coordinate frame: a single fovea-centred, isotropic-mm 6x6 frame (512x512 at 6/512 mm/px).
The advRPE en-face already lives there; the IR (8.77x7.31 mm over 768x768, anisotropic pixels)
is resampled to the same mm scale and centre-cropped to the 6x6. Because scale is fixed by the
mm frame and both scans are fovea-centred, the residual IR<->advRPE transform is a small
EUCLIDEAN (rotation + translation, NO scale) -- recovered by ECC on contrast-normalised
intensity. Polarity matches across modalities (GA bright, vessels dark in both), so intensity
registration locks onto the GA blob (large lesions) and vessels (no-GA eyes) alike.

Per ok eye it writes (into cohort/<subj>/<eye>/):
  spectralis_ir_6x6.png   fovea-centred 6x6 IR crop, GEOMETRY only (no registration)   [reusable]
  spectralis_ir_reg.png   IR warped into the advRPE 6x6 frame (registered)             [reusable]
  reg_panel.png           [IR context | advRPE SubRPE | checker | red/green] QC
  sixbysix_panel.png      [Spect 6x6 geom | Spect 6x6 reg | advRPE 6x6 | advRPE+GA]
And at the project root:
  registration.csv        transform params + quality (ecc rho, NMI, rot/shift) + reg_flag
  _MONTAGE_registration.png / _MONTAGE_sixbysix.png   one tile/eye, worst-quality-first

Read-only inputs; only new PNG/CSV artifacts are written.
Run: oct_env\\Scripts\\python.exe register_qc.py
"""
import csv
import json
import math
import os

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.util import img_as_float

import qcviz as qv

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")

FRAME = 512
ADV_MMPP = 6.0 / FRAME          # advRPE 6x6 mm en-face pixel size (isotropic) = working scale
MAX_ROT_DEG = 20.0              # plausible residual rotation
MAX_SHIFT_MM = 1.75            # plausible residual fovea offset
MAX_SCALE_DEV = 0.12           # ORB similarity scale must stay within +-12%
VESSEL_PCT = 82                # vesselness percentile -> binary vessel map for the overlap metric
# reg_flag = review (eyeball it) if ANY: vessel overlap weak, or the transform is larger than a
# fovea-centred acquisition plausibly needs (likely a mis-centre or a bad lock).
REVIEW_VDICE = 0.185
REVIEW_SHIFT_MM = 1.2
REVIEW_ROT_DEG = 8.0


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if img is None else img


def resample(img, src_mmpp_xy, out=FRAME, interp=cv2.INTER_LINEAR):
    """Resample into the isotropic-mm 6x6 frame, centred on the image centre (fovea prior)."""
    h, w = img.shape[:2]
    sx, sy = src_mmpp_xy[0] / ADV_MMPP, src_mmpp_xy[1] / ADV_MMPP
    cxs, cys = (w - 1) / 2.0, (h - 1) / 2.0
    cxd = cyd = (out - 1) / 2.0
    M = np.array([[sx, 0, cxd - sx * cxs], [0, sy, cyd - sy * cys]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=interp, borderValue=0)


def clahe(g):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def vmap(g):
    """Frangi vesselness as uint8 (dark vessels in both modalities -> black_ridges)."""
    return qv.norm8(frangi(img_as_float(g), sigmas=range(1, 5), black_ridges=True))


def vbin(vm):
    """Binary vessel map (top percentile of a vesselness image)."""
    return vm >= np.percentile(vm, VESSEL_PCT)


def dice(a, b):
    a, b = a > 0, b > 0
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 0.0


def nmi(a, b, bins=64):
    """Normalised mutual information in [1,2]; 1 = independent, higher = more aligned."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    p = h / max(h.sum(), 1)
    pa, pb = p.sum(1), p.sum(0)
    hab = -(p[p > 0] * np.log(p[p > 0])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float((ha + hb) / hab) if hab > 0 else 1.0


def decompose(M):
    """Return (rotation_deg, shift_mm, scale) of a forward 2x3 affine."""
    a, c = M[0, 0], M[1, 0]
    rot = math.degrees(math.atan2(c, a))
    shift = math.hypot(M[0, 2], M[1, 2]) * ADV_MMPP
    return rot, shift, math.hypot(a, c)


def plausible(M):
    rot, shift, scale = decompose(M)
    return abs(rot) <= MAX_ROT_DEG and shift <= MAX_SHIFT_MM and abs(scale - 1.0) <= MAX_SCALE_DEV


def warp(img, Mfwd, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, Mfwd, (FRAME, FRAME), flags=interp, borderValue=0)


def ecc_core(ref, mov):
    """Coarse-to-fine ECC euclidean. ref=template(adv), mov=input(ir), both uint8.
    Returns forward (ir->adv) 2x3 + rho, or (None,-1)."""
    w = np.eye(2, 3, dtype=np.float32)
    cc = -1.0
    for scale in (0.5, 1.0):
        r = cv2.resize(ref, None, fx=scale, fy=scale)
        m = cv2.resize(mov, None, fx=scale, fy=scale)
        w[0, 2] *= scale
        w[1, 2] *= scale
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        try:
            cc, w = cv2.findTransformECC(r.astype(np.float32) / 255.0, m.astype(np.float32) / 255.0,
                                         w, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        except cv2.error:
            return None, -1.0
        w[0, 2] /= scale
        w[1, 2] /= scale
    return cv2.invertAffineTransform(w), float(cc)


def orb_core(mov, ref, n=3000):
    """ORB + RANSAC partial-affine. mov=ir, ref=adv (uint8). Returns forward (ir->adv) 2x3 + inliers."""
    orb = cv2.ORB_create(n)
    k1, d1 = orb.detectAndCompute(mov, None)
    k2, d2 = orb.detectAndCompute(ref, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 8:
        return None, 0
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return (M.astype(np.float32), int(inl.sum())) if M is not None else (None, 0)


def candidates(ir6, adv6, flip):
    """All plausible (method, Mfwd, rho, inliers, flip) from intensity + vesselness, ECC + ORB.

    Intensity locks onto the GA blob (large lesions); vesselness onto vessels (no-GA eyes).
    """
    ir_i, adv_i = clahe(ir6), clahe(adv6)
    ir_v, adv_v = vmap(ir6), vmap(adv6)
    ir_vb, adv_vb = cv2.GaussianBlur(ir_v, (0, 0), 2.0), cv2.GaussianBlur(adv_v, (0, 0), 2.0)
    out = []
    Mie, r = ecc_core(cv2.GaussianBlur(adv_i, (0, 0), 2.0), cv2.GaussianBlur(ir_i, (0, 0), 2.0))
    if Mie is not None:
        out.append(("int-ecc", Mie, r, -1, flip))
    Mve, r = ecc_core(adv_vb, ir_vb)
    if Mve is not None:
        out.append(("ves-ecc", Mve, r, -1, flip))
    Mio, n = orb_core(ir_i, adv_i)
    if Mio is not None:
        out.append(("int-orb", Mio, -1.0, n, flip))
    Mvo, n = orb_core(ir_v, adv_v)
    if Mvo is not None:
        out.append(("ves-orb", Mvo, -1.0, n, flip))
    return [c for c in out if plausible(c[1])]


def register(ir6, adv6):
    """Build candidates (incl. a flipped-IR set), pick the one with the best vessel-overlap Dice."""
    adv_vb = vbin(vmap(adv6))

    def quality(src, M):
        reg = warp(src, M)
        return dice(vbin(vmap(reg)), adv_vb), nmi(reg, adv6)

    pool = [(c, ir6) for c in candidates(ir6, adv6, False)]
    irf = np.fliplr(ir6)
    pool += [(c, irf) for c in candidates(irf, adv6, True)]
    ident = (("none", np.eye(2, 3, dtype=np.float32), -1.0, 0, False), ir6)

    best, best_src, best_q = ident[0], ir6, quality(ir6, ident[0][1])
    for cand, src in pool:
        q = quality(src, cand[1])
        if q[0] > best_q[0] + 1e-4:          # higher vessel Dice wins (NMI breaks near-ties)
            best, best_src, best_q = cand, src, q
        elif abs(q[0] - best_q[0]) <= 1e-4 and q[1] > best_q[1]:
            best, best_src, best_q = cand, src, q

    method, M, rho, inliers, flip = best
    rot, shift, _ = decompose(M)
    return dict(method=method, M=M, rho=rho, inliers=inliers, vdice=best_q[0],
                nmi=best_q[1], flip=flip, rot=rot, shift=shift)


def ok_eyes():
    with open(PAIRING, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("qc_status") == "ok"]


def eye_meta(subject, eye):
    try:
        with open(os.path.join(COH, subject, "meta.json")) as f:
            e = json.load(f).get("eyes", {}).get(eye, {})
        fov = [float(x) for x in e.get("fov_mm", [8.77, 7.31])]
        return fov, float(e.get("advRPE_area_mm2") or 0.0)
    except Exception:
        return [8.77, 7.31], 0.0


def process(subject, eye):
    ed = os.path.join(COH, subject, eye)
    ir = load_gray(os.path.join(ed, "spectralis_ir.png"))
    adv = load_gray(os.path.join(ed, "advrpe_subrpe_enface.png"))
    if ir is None or adv is None:
        return None
    mask = load_gray(os.path.join(ed, "ga_mask.png"))
    fov, area = eye_meta(subject, eye)

    ir6 = resample(ir, (fov[0] / ir.shape[1], fov[1] / ir.shape[0]))   # fovea-centred 6x6, geometry
    adv6 = resample(adv, (ADV_MMPP, ADV_MMPP))
    mask6 = resample(mask, (ADV_MMPP, ADV_MMPP), interp=cv2.INTER_NEAREST) if mask is not None \
        else np.zeros((FRAME, FRAME), np.uint8)

    reg = register(ir6, adv6)
    src = np.fliplr(ir6) if reg["flip"] else ir6
    ir_reg = warp(src, reg["M"])

    qv.save_rgb(os.path.join(ed, "spectralis_ir_6x6.png"), ir6)
    qv.save_rgb(os.path.join(ed, "spectralis_ir_reg.png"), ir_reg)

    flag = "review" if (reg["vdice"] < REVIEW_VDICE or reg["shift"] > REVIEW_SHIFT_MM
                        or abs(reg["rot"]) > REVIEW_ROT_DEG) else "ok"
    hdr = (f"{subject} {eye}  GA={area:.2f} mm2  {reg['method']}"
           f"{'(flip)' if reg['flip'] else ''}  vDice={reg['vdice']:.3f}  NMI={reg['nmi']:.3f}  "
           f"rot={reg['rot']:+.1f}deg shift={reg['shift']:.2f}mm  [{flag}]")

    ctx = cv2.resize(ir, (FRAME, FRAME))                 # full-field context (squished, for anatomy ID)
    reg_panel = qv.panel(
        [ctx, qv.draw_contour(adv6, mask6), qv.checkerboard(ir_reg, adv6),
         qv.redgreen(ir_reg, adv6)],
        ["Spectralis IR (full field)", "advRPE SubRPE 6x6 +GA", "checker (registered)",
         "red=Spectralis  green=advRPE"],
        header=hdr, mm_per_px=ADV_MMPP, scalebar_mm=1.0, bar_on=[False, True, True, True])
    qv.save_rgb(os.path.join(ed, "reg_panel.png"), reg_panel)

    six_panel = qv.panel(
        [ir6, ir_reg, adv6, qv.draw_contour(adv6, mask6)],
        ["Spect 6x6 (geom)", "Spect 6x6 (registered)", "advRPE 6x6", "advRPE 6x6 + GA"],
        header=hdr, mm_per_px=ADV_MMPP, scalebar_mm=1.0)
    qv.save_rgb(os.path.join(ed, "sixbysix_panel.png"), six_panel)

    return dict(subject=subject, eye=eye, area_mm2=area, method=reg["method"],
                flip=int(reg["flip"]), vdice=reg["vdice"], nmi=reg["nmi"], rho=reg["rho"],
                rot=reg["rot"], shift=reg["shift"], reg_flag=flag, M=reg["M"],
                _tiles=(ir_reg, adv6, mask6))


def main():
    results, reg_tiles, six_tiles = [], [], []
    for r in ok_eyes():
        out = process(r["subject"], r["eye"])
        if out is None:
            print(f"  SKIP {r['subject']} {r['eye']} (missing image)", flush=True)
            continue
        print(f"  {out['subject']} {out['eye']:2}  {out['method']:7} "
              f"vDice={out['vdice']:.3f} NMI={out['nmi']:.3f} rot={out['rot']:+5.1f} "
              f"shift={out['shift']:.2f}mm -> {out['reg_flag']}", flush=True)
        ir_reg, adv6, mask6 = out.pop("_tiles")
        cap = f"{out['subject'].replace('NHAMD-003-', '')} {out['eye']} D{out['vdice']:.2f} {out['reg_flag']}"
        key = out["vdice"]
        reg_tiles.append((key, qv.label_tile(qv.redgreen(ir_reg, adv6), cap)))
        six = qv.panel([ir_reg, qv.draw_contour(adv6, mask6)], ["Spect reg", "advRPE+GA"],
                       mm_per_px=ADV_MMPP)
        six_tiles.append((key, qv.label_tile(six, cap)))
        results.append(out)

    reg_tiles.sort(key=lambda t: t[0])      # worst (lowest vessel Dice) first
    six_tiles.sort(key=lambda t: t[0])
    qv.save_rgb(os.path.join(OUT_DIR, "_MONTAGE_registration.png"),
                qv.montage([t for _, t in reg_tiles], cols=6))
    qv.save_rgb(os.path.join(OUT_DIR, "_MONTAGE_sixbysix.png"),
                qv.montage([t for _, t in six_tiles], cols=4))

    cols = ["subject", "eye", "area_mm2", "method", "flip", "vdice", "nmi", "rho", "rot", "shift",
            "reg_flag", "m00", "m01", "m02", "m10", "m11", "m12"]
    with open(os.path.join(RESULTS_DIR, "registration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            M = r.pop("M").ravel()
            row = {**{k: r[k] for k in ("subject", "eye", "method", "flip", "reg_flag")},
                   "area_mm2": f"{r['area_mm2']:.4f}", "vdice": f"{r['vdice']:.4f}",
                   "nmi": f"{r['nmi']:.4f}", "rho": f"{r['rho']:.4f}", "rot": f"{r['rot']:.2f}",
                   "shift": f"{r['shift']:.3f}"}
            row.update({f"m{i//3}{i%3}": f"{M[i]:.5f}" for i in range(6)})
            w.writerow(row)

    n_rev = sum(1 for r in results if r["reg_flag"] == "review")
    print(f"\nDONE  {len(results)} eyes  ({n_rev} flagged review)")
    print("  -> registration.csv, _MONTAGE_registration.png, _MONTAGE_sixbysix.png")


if __name__ == "__main__":
    main()
