#!/usr/bin/env python
"""Review the current projection (from cached features) on the good-BM eyes: phone-friendly grid
sorted controls->large GA, plus the MASK-FREE scorecard. Fast (reads features/, no E2E).

Scorecard:
  control uniformity  -> for advRPE=0 eyes, the false-GA area at a per-eye-adaptive threshold
                         (med+4*MAD) + cRORA. Lower = flatter control = fewer false positives.
The advRPE GA outline is drawn on GA eyes only as an APPROXIMATE reference view (NOT registered).

Run: oct_env\\Scripts\\python.exe perfect_projection.py
"""
import csv
import glob
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology

import qcviz as qv

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
FEAT = os.path.join(OUT_DIR, "features")
MMPP = 6.0 / 512.0
MIN_DIAM_PX = 0.250 / MMPP


def crora_area(binimg):
    binimg = morphology.remove_small_holes(binimg, area_threshold=int(MIN_DIAM_PX ** 2))
    lbl = measure.label(binimg)
    a = sum(r.area for r in measure.regionprops(lbl) if r.axis_major_length >= MIN_DIAM_PX)
    return a * MMPP ** 2


def main():
    good = set()
    with open(os.path.join(RESULTS_DIR, "bm_good.csv")) as f:
        for r in csv.DictReader(f):
            good.add((r["subject"], r["eye"]))
    data = []
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        if (str(d["subject"]), str(d["eye"])) not in good:      # good-BM working set only
            continue
        data.append(dict(subject=str(d["subject"]), eye=str(d["eye"]),
                         ft=d["f_trans"], area=float(d["area"])))
    data.sort(key=lambda d: d["area"])
    tiles, ctrl = [], []
    for d in data:
        ft, area = d["ft"], d["area"]
        sm = gaussian_filter(ft, 1.0)
        disp = qv.norm8(np.clip(sm, np.percentile(sm, 2), np.percentile(sm, 99)))
        med = np.median(ft)
        mad = np.median(np.abs(ft - med)) + 1e-6
        fp = crora_area(ft > med + 4 * mad)
        tag = d["subject"].replace("NHAMD-003-", "")
        if area < 0.05:
            ctrl.append((tag, d["eye"], fp))
            t = disp; lab = f"{tag} {d['eye']} CTRL  FP={fp:.1f}mm2"
        else:
            advm = cv2.imread(os.path.join(COH, d["subject"], d["eye"], "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
            advm = advm > 127 if advm is not None else np.zeros((512, 512), bool)
            t = qv.draw_contour(disp, advm, (0, 220, 255), 2)       # approx reference outline
            lab = f"{tag} {d['eye']}  GA={area:.2f}mm2"
        tiles.append(qv.label_tile(cv2.resize(qv.ensure_rgb(t), (360, 360)), lab))

    # 2 tiles per row
    rows = []
    for i in range(0, len(tiles), 2):
        chunk = tiles[i:i + 2]
        while len(chunk) < 2:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack([chunk[0], np.zeros((chunk[0].shape[0], 6, 3), np.uint8), chunk[1]]))
    grid = rows[0]
    for r in rows[1:]:
        grid = np.vstack([grid, np.full((6, grid.shape[1], 3), 70, np.uint8), r])
    qv.save_rgb(os.path.join(OUT_DIR, "projection_review.png"), grid)
    print("control false-GA area (lower=better, more uniform):")
    for tag, eye, fp in sorted(ctrl, key=lambda c: -c[2]):
        print(f"  {tag} {eye}: {fp:.2f} mm2")
    print(f"\nwrote projection_review.png ({len(tiles)} eyes)")


if __name__ == "__main__":
    main()
