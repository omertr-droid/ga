#!/usr/bin/env python
"""ITEM B probe — ABSOLUTE hypertransmission floor for control specificity.

STANDALONE experiment (does NOT edit oac_ga.py / footprint.py). Imports the detector and reuses
oac_ga.prep(baseline='radial2') with the DL BM, then:
  1. prints the hyper6 distribution over `core` (median, p75, p90, max) for CONTROLS vs GA eyes,
  2. reports, per eye, the relative-gate thresholds the reference footprint uses
     (hyper_keep*p75(h[core]), and after step-1 the hyper_frac*p60(h[b])),
  3. reimplements the footprint gate with an ABSOLUTE FLOOR added and sweeps the floor to find the
     value that zeroes the 3 control FPs (016 OD, 009 OS, 002 OD) WITHOUT regressing the GA eyes.

Run (from repo root):
  oct_env\\Scripts\\python.exe src\\exp_hyper_floor.py
"""
import csv
import os
import sys

os.environ["OCT_BM_DL"] = "1"          # DL BM (must be set before bm_dl import / first use)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

import bm_dl
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj

MMPP2 = oac_ga.MMPP2

# focused subset: (subject, visit, eye, role, plex_mm2)
CONTROLS = [
    ("NHAMD-003-016-V2", "V2", "OD", 0.0),
    ("NHAMD-003-009-V2", "V2", "OS", 0.0),
    ("NHAMD-003-002-V2", "V2", "OD", 0.0),
    ("NHAMD-003-012-V3", "V3", "OD", 0.0),
    ("NHAMD-003-006-V3", "V3", "OS", 0.0),
]
GA = [
    ("NHAMD-003-005-V3", "V3", "OD", 1.0829),
    ("NHAMD-003-005-V3", "V3", "OS", 0.5741),
    ("NHAMD-003-008-V1", "V1", "OD", 13.7795),
    ("NHAMD-003-008-V1", "V1", "OS", 15.0612),
    ("NHAMD-003-015-V3", "V3", "OD", 1.9918),
    ("NHAMD-003-003-V3", "V3", "OD", 2.7755),
    ("NHAMD-003-011-V3", "V3", "OS", 2.0958),
]


def e2e_path_for(subject, eye):
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["subject"] == subject and r["eye"].upper() == eye.upper():
                return os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    raise SystemExit(f"{subject} {eye} not in pairing csv")


def load_prep(subject, eye):
    path = e2e_path_for(subject, eye)
    raw = e2e_source.open_e2e(path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    bm = bm_dl.segment_volume(ov.vol)
    p = oac_ga.prep(ov, bm, baseline="radial2")
    return p


# ---- reference footprint gate (copied EXACTLY from oac_ga.footprint, defaults) ----
def ref_footprint(p, frac=0.50, min_diam_um=250.0, close_mm=0.15, hyper_frac=0.7, hyper_keep=0.4):
    b = (p["loss6"] < frac * p["base"]) & p["core"]
    if "hyper6" in p and b.any():
        h = p["hyper6"]
        b = b & (h > hyper_keep * float(np.percentile(h[p["core"]], 75)))
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            b = b | (holes & (h > hyper_frac * float(np.percentile(h[b], 60))))
    mask = fp.crora(binary_fill_holes(b), min_diam_um)
    return mask, float(mask.sum()) * MMPP2


# ---- candidate footprint gate WITH an absolute floor on hyper6 ----
def floor_footprint(p, floor, frac=0.50, min_diam_um=250.0, close_mm=0.15, hyper_frac=0.7,
                    hyper_keep=0.4):
    """Identical to ref_footprint but the criterion-1 transmission gate becomes
       h > MAX(hyper_keep*p75(h[core]), floor)   and the hole-fill uses MAX(..., floor) too.
    The floor is an ABSOLUTE physical minimum in hyper6 units (sub-BM slab / median-RPE ratio,
    destriped+smoothed), so a flat control whose relative p75 is tiny can no longer pass on noise."""
    b = (p["loss6"] < frac * p["base"]) & p["core"]
    if "hyper6" in p and b.any():
        h = p["hyper6"]
        rel1 = hyper_keep * float(np.percentile(h[p["core"]], 75))
        b = b & (h > max(rel1, floor))
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            rel2 = hyper_frac * float(np.percentile(h[b], 60))
            b = b | (holes & (h > max(rel2, floor)))
    mask = fp.crora(binary_fill_holes(b), min_diam_um)
    return mask, float(mask.sum()) * MMPP2


def hstats(p):
    h = p["hyper6"]; c = p["core"]
    hc = h[c]
    rel1 = 0.4 * float(np.percentile(hc, 75))   # the criterion-1 relative threshold actually used
    return dict(med=float(np.median(hc)), p75=float(np.percentile(hc, 75)),
                p90=float(np.percentile(hc, 90)), p99=float(np.percentile(hc, 99)),
                mx=float(hc.max()), rel1=rel1)


def main():
    print(f"bm_dl.active() = {bm_dl.active()}")
    rows = []
    for grp, lst in (("CONTROL", CONTROLS), ("GA", GA)):
        for subject, visit, eye, plex in lst:
            p = load_prep(subject, eye)
            s = hstats(p)
            ref_mask, ref_area = ref_footprint(p)
            rows.append((grp, subject, eye, plex, p, s, ref_area))
            print(f"[{grp:7}] {subject} {eye}  PLEX={plex:5.2f}  ref_area={ref_area:6.2f}  "
                  f"hyper6[core] med={s['med']:.3f} p75={s['p75']:.3f} p90={s['p90']:.3f} "
                  f"p99={s['p99']:.3f} max={s['mx']:.3f}  | crit1_relthr(0.4*p75)={s['rel1']:.4f}",
                  flush=True)

    print("\n==== hyper6[core] summary (the floor must sit ABOVE control noise, BELOW real GA transmission) ====")
    for grp in ("CONTROL", "GA"):
        sub = [r for r in rows if r[0] == grp]
        print(f"  {grp}:")
        for _, subject, eye, plex, p, s, ref_area in sub:
            # the hyper6 over the pixels the RPE-loss criterion actually selects (pre transmission gate)
            b0 = (p["loss6"] < 0.5 * p["base"]) & p["core"]
            hb = p["hyper6"][b0] if b0.any() else np.array([0.0])
            print(f"    {subject} {eye}: core p90={s['p90']:.3f} max={s['mx']:.3f} | "
                  f"hyper6 over loss-selected px: med={float(np.median(hb)):.3f} "
                  f"p75={float(np.percentile(hb,75)):.3f} p90={float(np.percentile(hb,90)):.3f}")

    print("\n==== floor sweep: ref vs floored area per eye (mm2) ====")
    floors = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    hdr = "  {:22} {:3} {:>6} {:>7}".format("eye", "", "PLEX", "ref") + \
          "".join(f"{f:>7.2f}" for f in floors)
    print(hdr)
    for grp in ("CONTROL", "GA"):
        for _, subject, eye, plex, p, s, ref_area in [r for r in rows if r[0] == grp]:
            line = "  {:22} {:3} {:6.2f} {:7.2f}".format(subject, eye, plex, ref_area)
            for f in floors:
                _, a = floor_footprint(p, f)
                line += f"{a:7.2f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
