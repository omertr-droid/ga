#!/usr/bin/env python
"""Cohort-wide view of the HARDENED RPE-loss channel, from cache (fast, no E2E). Per good-BM eye, the
PLEX advRPE GA annotation (cyan outline) is overlaid on EVERY channel so you can see whether the channel
fires INSIDE the GA and stays empty on controls:
  [ f_trans+GA | f_rpe(hi=lost)+GA | trans-AND-rpe+GA | advRPE GA (ref) | FAF(GA=dark) ]
advRPE outline is geometry-approx (NOT pixel-registered) -> a visual reference, not a score. Sorted
controls->large GA, 5 eyes/image.

Run: oct_env\\Scripts\\python.exe rpe_review.py
"""
import csv
import glob
import json
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import qcviz as qv
import register_qc as reg

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
FEAT = os.path.join(OUT_DIR, "features")
DISP_LO, DISP_HI = 0.18, 0.62
SZ = 300
CY = (0, 220, 255)


def thr(a, k):
    m = np.median(a); s = np.median(np.abs(a - m)) + 1e-6
    return a > m + k * s


def sh(a, lo, hi):
    return qv.ensure_rgb(cv2.resize(qv.norm8(np.clip(a, lo, hi)), (SZ, SZ)))


def main():
    good = set()
    with open(os.path.join(RESULTS_DIR, "bm_good.csv")) as f:
        for r in csv.DictReader(f):
            good.add((r["subject"], r["eye"]))
    data = []
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        key = (str(d["subject"]), str(d["eye"]))
        if key in good:
            data.append((key, gaussian_filter(d["f_trans"], 1.0), gaussian_filter(d["f_rpe"], 1.0), float(d["area"])))
    data.sort(key=lambda t: t[3])

    rows = []
    for (subject, eye), ft, fr, area in data:
        ed = os.path.join(COH, subject, eye)
        with open(os.path.join(COH, subject, "meta.json")) as f:
            fov = [float(v) for v in json.load(f)["eyes"][eye]["fov_mm"]]
        both = thr(ft, 2.0) & thr(fr, 1.0)
        advm = cv2.imread(os.path.join(ed, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
        advm = cv2.resize((advm > 127).astype(np.uint8), (SZ, SZ)) > 0 if advm is not None else np.zeros((SZ, SZ), bool)
        sub = cv2.imread(os.path.join(ed, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
        sub = sub if sub is not None else np.zeros((SZ, SZ), np.uint8)
        faf = cv2.imread(os.path.join(ed, "spectralis_baf.png"), cv2.IMREAD_GRAYSCALE)
        ftile = sh(reg.resample(faf.astype(np.float32), (fov[0] / faf.shape[1], fov[1] / faf.shape[0])),
                   *np.percentile(faf, [2, 98])) if faf is not None else np.zeros((SZ, SZ, 3), np.uint8)

        t_tr = qv.draw_contour(sh(ft, DISP_LO, DISP_HI), advm, CY, 2)
        t_rp = qv.draw_contour(sh(fr, np.percentile(fr, 5), np.percentile(fr, 99)), advm, CY, 2)
        t_and = qv.draw_contour(qv.ensure_rgb(cv2.resize((both * 255).astype(np.uint8), (SZ, SZ))), advm, CY, 2)
        t_ref = qv.draw_contour(cv2.resize(qv.ensure_rgb(sub), (SZ, SZ)), advm, CY, 2)
        tag = subject.replace("NHAMD-003-", "") + " " + eye + (f"  GA={area:.2f}mm2" if area >= 0.05 else "  CTRL (no GA)")
        rows.append(qv.panel([t_tr, t_rp, t_and, t_ref, ftile],
                             [f"{tag}  TRANS+GA", "RPE-loss+GA", "trans AND rpe +GA",
                              "advRPE GA (ref)", "FAF (GA=dark)"]))

    PER = 5
    for k in range(0, len(rows), PER):
        batch = rows[k:k + PER]
        W = max(b.shape[1] for b in batch)
        batch = [np.pad(b, ((0, 0), (0, W - b.shape[1]), (0, 0))) for b in batch]
        grid = batch[0]
        for b in batch[1:]:
            grid = np.vstack([grid, np.full((6, W, 3), 80, np.uint8), b])
        qv.save_rgb(os.path.join(OUT_DIR, f"rpe_review_p{k // PER + 1}.png"), grid)
    print(f"wrote {(len(rows) + PER - 1) // PER} images (rpe_review_p*.png), {len(rows)} eyes")


if __name__ == "__main__":
    main()
