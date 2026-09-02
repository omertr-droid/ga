#!/usr/bin/env python
"""PLEX -> projection registration + GA LABEL TRANSFER (the canonical module for segmentation labels).

Goal: put the advRPE (PLEX) GA annotation onto our Spectralis projection, so it can train a segmenter
that runs on the projection alone. Two-map method (one OCT volume -> two en-faces in the same
fovea-centered 6x6 frame, both via to_6mm => pixel-aligned):
  Map V = sub-RPE MEAN-intensity SHADOWGRAM   (registration image: keeps GA + sub-RPE structure)
  Map G = cached f_trans                       (the projection / payload)
Register Map V -> advRPE SubRPE (a like-to-like sub-RPE lock; GA + structure drive it), then carry the
SAME transform to Map G. For the LABEL, apply the INVERSE transform to the advRPE GA mask so it lands in
the projection's native frame -> `galabel` aligned to the cached features = the training target.
NB: the lock is GA-influenced -> that is correct for LABEL TRANSFER (we want the GA aligned), but it is
NOT an independent registration, so do not use it as a spatial-Dice validation metric. Works where GA is
present (the eyes that need labels); controls get empty labels (negatives).

Run: oct_env\\Scripts\\python.exe src\\register_projection.py                    (all good-BM eyes)
     oct_env\\Scripts\\python.exe src\\register_projection.py NHAMD-003-003-V3 OD   (one eye)
Out: outputs/register_projection/{subject}_{eye}.png        QC panel
     outputs/register_projection/labels/{s}_{e}_galabel.png GA label in the projection frame (target)
     outputs/register_projection/_MONTAGE.png               projection + transferred label, all eyes
     results/projection_registration.csv                    transform + metrics per eye
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import m2_bm
import m3_projections as mp
import qcviz as qv
import register_qc as rq
from paths import COHORT_DIR, OUT_DIR, RESULTS_DIR

CY = (0, 220, 255)
YEL = (255, 255, 0)
TR_LO, TR_HI = 0.18, 0.62
OUT = os.path.join(OUT_DIR, "register_projection")
LBL = os.path.join(OUT, "labels")
FEAT = os.path.join(OUT_DIR, "features")


def to512(a, interp=cv2.INTER_LINEAR):
    return cv2.resize(np.asarray(a), (512, 512), interpolation=interp)


def good_bm_eyes():
    with open(os.path.join(RESULTS_DIR, "bm_good.csv")) as f:
        return [(r["subject"], r["eye"]) for r in csv.DictReader(f)]


def process(subject, eye, loaded, writer):
    fp = os.path.join(FEAT, f"{subject}_{eye}.npz")
    if not os.path.exists(fp) or eye not in loaded:
        print(f"  {subject} {eye}: skip (no cache or no volume)", flush=True)
        return None
    vol, ilm, dev_bm = loaded[eye]
    if dev_bm is None:
        print(f"  {subject} {eye}: skip (no device BM)", flush=True)
        return None
    d = np.load(fp, allow_pickle=True)
    f_trans, fov = d["f_trans"].astype(np.float32), [float(v) for v in d["fov"]]
    area = float(d["area"]) if "area" in d else 0.0
    gview = (np.clip((gaussian_filter(f_trans, 1.0) - TR_LO) / (TR_HI - TR_LO), 0, 1) * 255).astype(np.uint8)

    bm = m2_bm.fill_bm(dev_bm)
    shadow6 = qv.norm8(mp.to_6mm(mp.destripe2d(mp.band(vol, bm, 10, 340, "mean")), fov))   # Map V

    ed = os.path.join(COHORT_DIR, subject, eye)
    adv = cv2.imread(os.path.join(ed, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
    if adv is None:
        print(f"  {subject} {eye}: skip (no advRPE)", flush=True)
        return None
    adv6 = to512(adv)
    m = cv2.imread(os.path.join(ed, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
    mask6 = to512(m, cv2.INTER_NEAREST) > 127 if m is not None else np.zeros((512, 512), bool)

    reg = rq.register(shadow6, adv6)                       # Map V -> advRPE
    M = reg["M"]
    flipf = np.fliplr if reg["flip"] else (lambda a: a)
    proj_reg = rq.warp(flipf(gview), M)                    # f_trans in advRPE frame (overlay QC)
    # GA LABEL: advRPE GA -> projection's native frame via the INVERSE transform (forward = M o flip)
    Minv = cv2.invertAffineTransform(M)
    galabel = flipf(rq.warp((mask6 * 255).astype(np.uint8), Minv, cv2.INTER_NEAREST)) > 127
    cv2.imwrite(os.path.join(LBL, f"{subject}_{eye}_galabel.png"), (galabel * 255).astype(np.uint8))

    writer.writerow({"subject": subject, "eye": eye, "method": reg["method"], "flip": int(reg["flip"]),
                     "rot": round(reg["rot"], 2), "shift": round(reg["shift"], 3),
                     "advRPE_area_mm2": round(area, 3), "label_px": int(galabel.sum()),
                     **{f"m{i // 3}{i % 3}": round(float(M.ravel()[i]), 5) for i in range(6)}})
    print(f"  {subject} {eye}: {reg['method']:8s} rot={reg['rot']:+5.1f} shift={reg['shift']:.2f}mm "
          f"flip={reg['flip']}  GA={area:.2f}mm2  label={int(galabel.sum())}px", flush=True)

    # zoom on the GA (in the projection frame)
    ys, xs = np.where(galabel if galabel.any() else mask6)
    cyc, cxc = (int(ys.mean()), int(xs.mean())) if len(ys) else (256, 256)
    r0 = int(max(80, 1.4 * max(ys.max() - ys.min(), xs.max() - xs.min()) / 2)) if len(ys) else 130
    y0, y1, x0, x1 = max(0, cyc - r0), min(512, cyc + r0), max(0, cxc - r0), min(512, cxc + r0)
    crop = lambda im: cv2.resize(qv.ensure_rgb(im)[y0:y1, x0:x1], (440, 440), interpolation=cv2.INTER_NEAREST)

    target = qv.draw_contour(gview, galabel, CY)           # f_trans + transferred GA label
    rowA = qv.panel(
        [qv.draw_contour(adv6, mask6, YEL), shadow6, target, qv.draw_contour(proj_reg, mask6, CY)],
        ["advRPE SubRPE + PLEX GA", "Map V: sub-RPE shadowgram (reg image)",
         "f_trans + GA LABEL (registered)  <-- training target", "QC: f_trans->advRPE frame + GA"],
        mm_per_px=rq.ADV_MMPP, scalebar_mm=1.0)
    rowB = qv.panel(
        [crop(qv.draw_contour(adv6, mask6, YEL, 2)), crop(qv.draw_contour(gview, galabel, CY, 2)),
         crop(qv.draw_contour(gview, mask6, (255, 120, 0), 2)), crop(qv.redgreen(galabel, mask6))],
        ["advRPE + GA (zoom)", "f_trans + GA LABEL registered (zoom)",
         "f_trans + GA mask GEOMETRY-only (zoom)", "registered red / geom green (zoom)"])
    W = max(rowA.shape[1], rowB.shape[1])
    rows = [np.pad(x, ((0, 0), (0, W - x.shape[1]), (0, 0))) for x in (rowA, rowB)]
    hdr = (f"{subject} {eye}  PLEX->projection label transfer (two-map).  {reg['method']} "
           f"rot={reg['rot']:+.1f} shift={reg['shift']:.2f}mm  GA={area:.2f}mm2")
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}.png"),
                qv.add_header(np.vstack([rows[0], np.full((6, W, 3), 80, np.uint8), rows[1]]), hdr))

    cap = f"{subject.replace('NHAMD-003-', '')} {eye} GA={area:.2f}"
    return area, qv.label_tile(target, cap)


def main():
    os.makedirs(LBL, exist_ok=True)
    args = sys.argv[1:]
    eyes = [(args[0], args[1])] if len(args) >= 2 else good_bm_eyes()
    by_sub = {}
    for s, e in eyes:
        by_sub.setdefault(s, []).append(e)
    print(f"PLEX->projection label transfer on {len(eyes)} eyes -> {OUT}\n", flush=True)

    cols = ["subject", "eye", "method", "flip", "rot", "shift", "advRPE_area_mm2", "label_px",
            "m00", "m01", "m02", "m10", "m11", "m12"]
    csvf = open(os.path.join(RESULTS_DIR, "projection_registration.csv"), "w", newline="")
    writer = csv.DictWriter(csvf, fieldnames=cols)
    writer.writeheader()

    tiles = []
    for subject in sorted(by_sub):
        try:
            loaded = m2_bm.load_subject_layers(subject)
        except Exception as ex:
            print(f"  {subject}: LOAD ERROR {type(ex).__name__}: {ex}", flush=True)
            continue
        for eye in by_sub[subject]:
            out = process(subject, eye, loaded, writer)
            if out:
                tiles.append(out)
    csvf.close()

    if tiles:
        tiles.sort(key=lambda t: t[0])
        qv.save_rgb(os.path.join(OUT, "_MONTAGE.png"), qv.montage([t for _, t in tiles], cols=4))
    print(f"\nDONE  {len(tiles)} eyes -> {OUT}/  (panels + labels/ + _MONTAGE.png), "
          f"results/projection_registration.csv", flush=True)


if __name__ == "__main__":
    main()
