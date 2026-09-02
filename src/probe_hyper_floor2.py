#!/usr/bin/env python
"""ITEM B PROBE part 2 — floor sweep on the DECISIVE eyes (the 3 firing controls + the small/medium GA
that an absolute floor could damage). Skips 008 OD/OS + 011 OS (huge, safely above any plausible floor).

Caches each eye's prep dict so the floor sweep is instant after the (slow) DL+prep pass.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["OCT_BM_DL"] = "1"

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

import bm_dl
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj

MMPP2 = oac_ga.MMPP2

EYES = [  # (subject, eye, label)  -- decisive subset
    ("NHAMD-003-016-V2", "OD", "control"),   # fires 1.31, hyper6 HIGH (med .425)
    ("NHAMD-003-009-V2", "OS", "control"),   # fires 2.27, hyper6 med .257
    ("NHAMD-003-002-V2", "OD", "control"),   # fires 1.06, hyper6 LOW  (med .070)
    ("NHAMD-003-012-V3", "OD", "control"),   # fires 0.00 already
    ("NHAMD-003-006-V3", "OS", "control"),   # fires 0.05 already
    ("NHAMD-003-005-V3", "OD", "ga"),        # PLEX 1.08 ref 1.055
    ("NHAMD-003-005-V3", "OS", "ga"),        # PLEX 0.57 ref 0.526
    ("NHAMD-003-015-V3", "OD", "ga"),        # PLEX 1.99 ref 1.498
    ("NHAMD-003-003-V3", "OD", "ga"),        # PLEX 2.78 ref 6.65 (already over-calls; floor must not worsen GA core)
]


def footprint_floor(p, frac, hyper_floor, hyper_frac=0.7, hyper_keep=0.4, close_mm=0.15, min_diam_um=250.0):
    b = (p["loss6"] < frac * p["base"]) & p["core"]
    if "hyper6" in p and b.any():
        h = p["hyper6"]
        keep_thr = max(hyper_keep * float(np.percentile(h[p["core"]], 75)), hyper_floor)
        b = b & (h > keep_thr)
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            fill_thr = max(hyper_frac * float(np.percentile(h[b], 60)), hyper_floor)
            b = b | (holes & (h > fill_thr))
    mask = fp.crora(binary_fill_holes(b), min_diam_um)
    return mask, float(mask.sum()) * MMPP2


def e2e_lookup():
    out = {}
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("qc_status") != "ok":
                continue
            out[(r["subject"], r["eye"].upper())] = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return out


def main():
    print(f"bm_dl.active()={bm_dl.active()} backend={bm_dl.backend()}", flush=True)
    e2e = e2e_lookup()
    recs = []
    for subject, eye, label in EYES:
        path = e2e[(subject, eye)]
        raw = e2e_source.open_e2e(path)
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol)
        p = oac_ga.prep(ov, bm, baseline="radial2")
        ref_mask, ref_area = oac_ga.footprint(p, 0.5)
        recs.append((subject, eye, label, p, ref_area, ref_mask))
        print(f"  loaded {subject} {eye} [{label}] ref_area={ref_area:.3f}", flush=True)

    floors = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    print("\n==== FLOOR SWEEP (area mm2) ====")
    print("eye".ljust(22) + "label".ljust(9) + "".join(f"{f:>8.2f}" for f in floors))
    for subject, eye, label, p, ref_area, ref_mask in recs:
        row = []
        for fl in floors:
            _, a = footprint_floor(p, 0.5, fl)
            row.append(a)
        print(f"{subject[-7:]+'_'+eye:22}{label:9}" + "".join(f"{a:8.3f}" for a in row))

    # hyper6 inside the firing region (where the relative gate currently passes) — the real discriminator
    print("\n==== hyper6 INSIDE the reference firing region (loss6<0.5*base & core, pre-hyper-gate) ====")
    for subject, eye, label, p, ref_area, ref_mask in recs:
        b = (p["loss6"] < 0.5 * p["base"]) & p["core"]
        if b.any():
            hin = p["hyper6"][b]
            print(f"{subject[-7:]+'_'+eye:22}{label:9} n={b.sum():6d}  "
                  f"hyper6: med={np.median(hin):.4f} p25={np.percentile(hin,25):.4f} "
                  f"p50={np.percentile(hin,50):.4f} p75={np.percentile(hin,75):.4f}")
        else:
            print(f"{subject[-7:]+'_'+eye:22}{label:9} n=0 (no RPE-loss)")


if __name__ == "__main__":
    main()
