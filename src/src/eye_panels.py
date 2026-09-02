#!/usr/bin/env python
"""One clearly-labeled panel PER good-BM eye that answers "is our projection better than the ORIGINAL
Spectralis en-face?" with the PLEX GA annotation visible on everything:
  [ Spectralis IR (ORIGINAL en-face) +GA | OUR PROJECTION +GA ]
  [ PLEX advRPE GA (REFERENCE map)       | FAF (GA=dark) +GA   ]
All in the same fovea-centred 6 mm frame. Cyan = PLEX advRPE GA outline (geometry-approx on the
Spectralis-derived tiles -- NOT pixel-registered -- a visual reference, exact only on the PLEX tile).
Reads the cached projection (features/) + cohort images. Fast.

Run: oct_env\\Scripts\\python.exe eye_panels.py
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
OUT = os.path.join(OUT_DIR, "eye_panels_out")
SZ = 380
DISP_LO, DISP_HI = 0.18, 0.62        # fixed cross-eye transmission window (see CLAUDE.md)
CY = (0, 220, 255)


def sq(img):
    return cv2.resize(qv.ensure_rgb(img), (SZ, SZ), interpolation=cv2.INTER_AREA)


def loc6(path, fov, p=(2, 98)):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g6 = reg.resample(g.astype(np.float32), (fov[0] / g.shape[1], fov[1] / g.shape[0]))
    return qv.norm8(np.clip(g6, np.percentile(g6, p[0]), np.percentile(g6, p[1])))


def main():
    os.makedirs(OUT, exist_ok=True)
    good = set()
    with open(os.path.join(RESULTS_DIR, "bm_good.csv")) as f:
        for r in csv.DictReader(f):
            good.add((r["subject"], r["eye"]))

    feats = {}
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        key = (str(d["subject"]), str(d["eye"]))
        if key in good:
            feats[key] = (d["f_trans"], float(d["area"]))

    for (subject, eye), (ft, area) in sorted(feats.items(), key=lambda kv: kv[1][1]):
        ed = os.path.join(COH, subject, eye)
        with open(os.path.join(COH, subject, "meta.json")) as f:
            fov = [float(v) for v in json.load(f)["eyes"][eye]["fov_mm"]]
        proj = qv.norm8(np.clip(gaussian_filter(ft, 1.0), DISP_LO, DISP_HI))

        advm = cv2.imread(os.path.join(ed, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
        advm = advm > 127 if advm is not None else np.zeros((512, 512), bool)
        sub = cv2.imread(os.path.join(ed, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
        sub = sub if sub is not None else np.zeros((512, 512), np.uint8)

        ir6 = loc6(os.path.join(ed, "spectralis_ir.png"), fov)            # ORIGINAL en-face
        ir_tile = sq(qv.draw_contour(ir6, advm, CY, 2)) if ir6 is not None else np.zeros((SZ, SZ, 3), np.uint8)
        ir_lab = "Spectralis IR  (ORIGINAL en-face) +GA" if ir6 is not None else "(no IR)"
        faf6 = loc6(os.path.join(ed, "spectralis_baf.png"), fov)
        faf_tile = sq(qv.draw_contour(faf6, advm, CY, 2)) if faf6 is not None else np.zeros((SZ, SZ, 3), np.uint8)
        faf_lab = "FAF  (GA = DARK) +GA" if faf6 is not None else "(no FAF)"

        # PLEX reference: subRPE en-face with GA outline AND a faint cyan fill so the "GA map" reads clearly
        ref = qv.ensure_rgb(sub).copy()
        fill = ref.copy(); fill[advm] = CY
        ref = cv2.addWeighted(ref, 0.7, fill, 0.3, 0)
        ref = qv.draw_contour(ref, advm, CY, 2)

        row1 = qv.panel([ir_tile, sq(qv.draw_contour(proj, advm, CY, 2))],
                        [ir_lab, "OUR PROJECTION  (GA = WHITE) +GA"])
        row2 = qv.panel([sq(ref), faf_tile],
                        ["PLEX advRPE GA  (REFERENCE map)", faf_lab])
        W = max(row1.shape[1], row2.shape[1])
        row1 = np.pad(row1, ((0, 0), (0, W - row1.shape[1]), (0, 0)))
        row2 = np.pad(row2, ((0, 0), (0, W - row2.shape[1]), (0, 0)))
        grid = qv.add_header(np.vstack([row1, row2]),
                             f"{subject} {eye}    advRPE GA = {area:.2f} mm2     (cyan = PLEX GA, approx on Spectralis tiles)")
        qv.save_rgb(os.path.join(OUT, f"GA{area:05.2f}_{subject}_{eye}.png"), grid)
        print(f"  {subject} {eye} GA={area:.2f} ir={'Y' if ir6 is not None else 'N'}", flush=True)
    print(f"\nwrote {len(feats)} per-eye panels -> eye_panels_out/")


if __name__ == "__main__":
    main()
