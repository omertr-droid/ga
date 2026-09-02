#!/usr/bin/env python
"""Diagnostic for the RPE-veto idea: render B-scans of 016 OD (FP) and 005 OD (gold GA) with the BM,
the rpe_surface peak row, and the per-column prom value, restricted to the GA-called columns -- to SEE
whether prom 'sees' a present RPE in 016 that it does not see in real GA.

Writes outputs/rpe_veto/<key>_<eye>_b####.png for a few representative B-scans.
No edits to oac_ga / m3_projections.
"""
import csv
import os
import sys

os.environ["OCT_BM_DL"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

import bm_dl
import m3_projections as mp
from paths import DATA_DIR, RESULTS_DIR, OUT_DIR
from reader.core import e2e_source, oac_ga
from viewer.core import ga_native

OUT = os.path.join(OUT_DIR, "rpe_veto")
os.makedirs(OUT, exist_ok=True)


def resolve(key, eye):
    for fn in ("bm_worklist.csv", "spectralis_ga_pairing.csv"):
        with open(os.path.join(RESULTS_DIR, fn), newline="") as f:
            for r in csv.DictReader(f):
                if r["subject"] == key and r["eye"].upper() == eye.upper() and r.get("qc_status", "ok") == "ok":
                    return os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return None


def render(key, eye, bscans):
    e2e = resolve(key, eye)
    raw = e2e_source.open_e2e(e2e)
    ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
    bm = bm_dl.segment_volume(ov.vol)
    p = oac_ga.prep(ov, bm, baseline="radial2")
    mask, area = oac_ga.footprint(p, 0.5)
    ga_nat = ga_native.enface_to_native(mask, ov.fov_mm, ov.n_bscans, ov.W).astype(bool)
    row_s, prom = mp.rpe_surface(ov.vol, bm)
    print(f"{key} {eye}: area={area:.3f} n={ov.n_bscans} -- GA cols per chosen B-scan:")
    for bi in bscans:
        bs = ov.vol[bi]
        img = cv2.cvtColor((np.clip(bs / (bs.max() + 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        H, W = bs.shape
        for x in range(W):
            yb = int(np.clip(bm[bi, x], 0, H - 1)); img[yb, x] = (0, 180, 255)          # BM = orange
            yr = int(np.clip(row_s[bi, x], 0, H - 1))
            if ga_nat[bi, x]:
                img[max(0, yr - 1):yr + 2, x] = (0, 0, 255)                              # RPE-peak in GA = red
                img[0:3, x] = (0, 255, 0)                                                 # green tick = GA-called col
        ncol = int(ga_nat[bi].sum())
        if ncol:
            pm = prom[bi, ga_nat[bi]]
            txt = f"b{bi} GAcols={ncol} prom(med={np.median(pm):.2f} p10={np.percentile(pm,10):.2f} p90={np.percentile(pm,90):.2f})"
        else:
            txt = f"b{bi} GAcols=0"
        cv2.putText(img, txt, (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        fn = os.path.join(OUT, f"{key}_{eye}_b{bi:04d}.png")
        cv2.imwrite(fn, img)
        print(f"   {txt} -> {fn}")


if __name__ == "__main__":
    # pick B-scans near the lesion centre (middle of the stack) for each eye
    render("NHAMD-003-016-V2", "OD", [40, 48, 56])
    render("NHAMD-003-005-V3", "OD", [40, 48, 56])
