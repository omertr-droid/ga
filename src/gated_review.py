#!/usr/bin/env python
"""Does gating transmission by RPE-integrity sharpen GA-vs-rest contrast? Per good-BM eye, from cache:
  [ f_trans +GA | f_gated = trans x RPE-gone +GA | advRPE GA (ref) | FAF (GA=dark) +GA ]
Cyan = PLEX advRPE GA outline on EVERY tile (geometry-approx on the Spectralis tiles -- a visual
reference, not a score). Both feature tiles use FIXED cross-eye windows (never per-eye percentile).
Sorted controls -> large GA, 5 eyes/image. Reads features/ (fast, no E2E).

Run: oct_env\\Scripts\\python.exe gated_review.py
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
SZ = 300
CY = (0, 220, 255)
TR_LO, TR_HI = 0.18, 0.62        # fixed transmission window (the existing feature)
GT_LO, GT_HI = 0.02, 0.40        # fixed gated window (gating lowers the magnitude)


def sh(a, lo, hi):
    return qv.ensure_rgb(cv2.resize(qv.norm8(np.clip(gaussian_filter(a, 1.0), lo, hi)), (SZ, SZ)))


def loc6(path, fov):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return np.zeros((SZ, SZ), np.uint8)
    g6 = reg.resample(g.astype(np.float32), (fov[0] / g.shape[1], fov[1] / g.shape[0]))
    return cv2.resize(qv.norm8(np.clip(g6, *np.percentile(g6, [2, 98]))), (SZ, SZ))


def main():
    good = set()
    with open(os.path.join(RESULTS_DIR, "bm_good.csv")) as f:
        for r in csv.DictReader(f):
            good.add((r["subject"], r["eye"]))
    data = []
    for p in sorted(glob.glob(os.path.join(FEAT, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        key = (str(d["subject"]), str(d["eye"]))
        if key in good and "f_gated" in d:
            data.append((key, d["f_trans"], d["f_gated"], float(d["area"])))
    data.sort(key=lambda t: t[3])

    rows = []
    for (subject, eye), ft, fg, area in data:
        ed = os.path.join(COH, subject, eye)
        with open(os.path.join(COH, subject, "meta.json")) as f:
            fov = [float(v) for v in json.load(f)["eyes"][eye]["fov_mm"]]
        advm = cv2.imread(os.path.join(ed, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
        advm = cv2.resize((advm > 127).astype(np.uint8), (SZ, SZ)) > 0 if advm is not None else np.zeros((SZ, SZ), bool)
        sub = cv2.imread(os.path.join(ed, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
        sub = sub if sub is not None else np.zeros((SZ, SZ), np.uint8)

        ref = qv.ensure_rgb(cv2.resize(sub, (SZ, SZ)))
        fill = ref.copy(); fill[advm] = CY
        ref = qv.draw_contour(cv2.addWeighted(ref, 0.7, fill, 0.3, 0), advm, CY, 2)
        faf = qv.draw_contour(qv.ensure_rgb(loc6(os.path.join(ed, "spectralis_baf.png"), fov)), advm, CY, 2)

        tag = subject.replace("NHAMD-003-", "") + " " + eye + (f"  GA={area:.2f}mm2" if area >= 0.05 else "  CTRL (no GA)")
        rows.append(qv.panel(
            [qv.draw_contour(sh(ft, TR_LO, TR_HI), advm, CY, 2),
             qv.draw_contour(sh(fg, GT_LO, GT_HI), advm, CY, 2), ref, faf],
            [f"{tag}  f_trans +GA", "f_GATED (trans x RPE-gone) +GA", "advRPE GA (ref)", "FAF (GA=dark) +GA"]))

    PER = 5
    for k in range(0, len(rows), PER):
        batch = rows[k:k + PER]
        W = max(b.shape[1] for b in batch)
        batch = [np.pad(b, ((0, 0), (0, W - b.shape[1]), (0, 0))) for b in batch]
        grid = batch[0]
        for b in batch[1:]:
            grid = np.vstack([grid, np.full((6, W, 3), 80, np.uint8), b])
        qv.save_rgb(os.path.join(OUT_DIR, f"gated_review_p{k // PER + 1}.png"), grid)
    print(f"wrote {(len(rows) + PER - 1) // PER} images (gated_review_p*.png), {len(rows)} eyes")


if __name__ == "__main__":
    main()
