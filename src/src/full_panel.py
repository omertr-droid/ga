#!/usr/bin/env python
"""Unified per-eye comparison, ALL views in one row (cache-only, fast):
  [ Spectralis IR | f_trans | f_gated | f_rpe (RPE-loss) | PLEX advRPE GA | FAF ]
The GA annotation appears ONLY on the PLEX tile, and ONLY as an outline (no fill) -- every other tile
is shown raw so the projections can be judged uncluttered. Fixed cross-eye display windows. Good-BM
eyes, sorted controls -> large GA, 5 eyes/image.

Run: oct_env\\Scripts\\python.exe full_panel.py
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
TR = (0.18, 0.62)        # f_trans window
GT = (0.02, 0.40)        # f_gated window
RP = (-0.85, 0.05)       # f_rpe window (high = RPE lost = GA)


def feat(a, lo, hi):
    return qv.ensure_rgb(cv2.resize(qv.norm8(np.clip(gaussian_filter(a, 1.0), lo, hi)), (SZ, SZ)))


def loc6(path, fov):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return np.zeros((SZ, SZ, 3), np.uint8)
    g6 = reg.resample(g.astype(np.float32), (fov[0] / g.shape[1], fov[1] / g.shape[0]))
    return qv.ensure_rgb(cv2.resize(qv.norm8(np.clip(g6, *np.percentile(g6, [2, 98]))), (SZ, SZ)))


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
            data.append((key, d["f_trans"], d["f_gated"], d["f_rpe"], float(d["area"])))
    data.sort(key=lambda t: t[4])

    rows = []
    for (subject, eye), ft, fg, fr, area in data:
        ed = os.path.join(COH, subject, eye)
        with open(os.path.join(COH, subject, "meta.json")) as f:
            fov = [float(v) for v in json.load(f)["eyes"][eye]["fov_mm"]]

        ir = loc6(os.path.join(ed, "spectralis_ir.png"), fov)
        faf = loc6(os.path.join(ed, "spectralis_baf.png"), fov)

        # PLEX tile: subRPE en-face with the GA OUTLINE only (no fill), drawn ONLY here
        sub = cv2.imread(os.path.join(ed, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
        sub = qv.ensure_rgb(cv2.resize(sub if sub is not None else np.zeros((SZ, SZ), np.uint8), (SZ, SZ)))
        advm = cv2.imread(os.path.join(ed, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
        advm = cv2.resize((advm > 127).astype(np.uint8), (SZ, SZ)) > 0 if advm is not None else np.zeros((SZ, SZ), bool)
        plex = qv.draw_contour(sub, advm, CY, 2)

        tag = subject.replace("NHAMD-003-", "") + " " + eye + (f"  GA={area:.2f}mm2" if area >= 0.05 else "  CTRL")
        rows.append(qv.panel(
            [ir, feat(ft, *TR), feat(fg, *GT), feat(fr, *RP), plex, faf],
            [f"{tag}  Spectralis IR", "f_trans", "f_gated", "f_rpe (RPE-loss)",
             "PLEX advRPE GA (outline)", "FAF"]))

    PER = 5
    for k in range(0, len(rows), PER):
        batch = rows[k:k + PER]
        W = max(b.shape[1] for b in batch)
        batch = [np.pad(b, ((0, 0), (0, W - b.shape[1]), (0, 0))) for b in batch]
        grid = batch[0]
        for b in batch[1:]:
            grid = np.vstack([grid, np.full((6, W, 3), 80, np.uint8), b])
        qv.save_rgb(os.path.join(OUT_DIR, f"full_panel_p{k // PER + 1}.png"),
                    qv.add_header(grid, "IR | f_trans | f_gated | f_rpe | PLEX advRPE GA (outline only) | FAF"))
    print(f"wrote {(len(rows) + PER - 1) // PER} images (full_panel_p*.png), {len(rows)} eyes")


if __name__ == "__main__":
    main()
