#!/usr/bin/env python
"""GATE (A) EXPERIMENT — RPE-ELEVATION / DRUSEN exclusion on the OAC footprint gate.

HYPOTHESIS (the clinician's): 016 OD's false GA firing is DRUSEN (RPE present-but-LIFTED, not lost).
If so, dropping firing columns whose RPE->BM elevation exceeds a threshold X should kill 016 while
sparing real GA (where the RPE is GONE -> elevation is LOW).

This reimplements oac_ga.footprint's gate logic BYTE-FOR-BYTE (radial2 baseline, DL BM), then ANDs the
firing seed b = (loss6 < frac*base) & core with an EXTRA elevation gate (elev6 < X) -- i.e. a drusen-
lifted column may not SEED GA. (It also applies the same elev mask to the criterion-2 fills so a filled
hole can't reintroduce a drusen column.) X=inf reproduces oac_ga.footprint(p,0.5) byte-for-byte.

elev6 is built per the workflow recipe:
    oac      = mp.oac_volume(ov.vol)
    elev_nat = clip((bm - band_argmax_row(oac, bm, *OAC_RPE_UM)) * AX, 0, None)   # (n,W) um, RPE->BM lift
    elev6    = to_enface(destripe2d(elev_nat, signed=False), fov)                  # aligns to rpe6

We sweep X over a sensible um range, report area vs X for 016 OD (FP) + the must-hold GA eyes
(005 OD gold, 005 OS faint, 008 OD large, 015 OD) + the stay-clean controls, find the X that drives
016 < 0.25 mm2, and report the worst GA %-change THERE. We are explicitly honest if NO X clears 016
without eroding GA (which is the predicted outcome if 016 fires at LOW elevation, like real GA).

Run (repo root):  oct_env\\Scripts\\python.exe src\\gate_elev_experiment.py
Writes results/gate_elev.csv (eye, label, plex, ref_area, area@each X) + prints the sweep + decision.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["OCT_BM_DL"] = "1"   # MUST be set before importing bm_dl

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

import bm_dl
import m3_projections as mp
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj

MMPP2 = oac_ga.MMPP2

# (subject, eye, label). FP = the lone remaining false-positive control we want to kill. must-hold =
# true GA we MUST preserve (005 OS is the FAINT erase-risk). stay-clean = controls that must remain ~0.
EYES = [
    ("NHAMD-003-016-V2", "OD", "FP"),        # PLEX 0, radial2 reads ~1.31 -> the target
    ("NHAMD-003-005-V3", "OD", "GA"),        # gold Dice 0.940, PLEX 1.08
    ("NHAMD-003-005-V3", "OS", "GA-faint"),  # small/faint, the erase-risk stress; PLEX 0.57
    ("NHAMD-003-008-V1", "OD", "GA-large"),  # large eccentric; PLEX 13.78
    ("NHAMD-003-015-V3", "OD", "GA"),        # PLEX 1.99
    ("NHAMD-003-002-V2", "OS", "control"),   # stay-clean
    ("NHAMD-003-006-V3", "OS", "control"),   # stay-clean
    ("NHAMD-003-012-V3", "OD", "control"),   # stay-clean
]

# Elevation cut sweep (um). inf = no gate (= reference). Drusen are typically >~30-50um lift; we go down
# to 15um to see if ANY cut bites 016 -- and to expose where it erases real GA.
XS = [float("inf"), 60.0, 50.0, 40.0, 35.0, 30.0, 25.0, 20.0, 18.0, 15.0]


def footprint_elev(p, elev6, frac, X, min_diam_um=250.0, close_mm=0.15, hyper_frac=0.7,
                   hyper_keep=0.4, fill_all_holes=True):
    """EXACT copy of oac_ga.footprint's gate logic, with an EXTRA elevation gate (elev6 < X) ANDed into
    the firing seed AND into the criterion-2 hole fills. X=inf -> byte-identical to oac_ga.footprint."""
    keep_elev = elev6 < X                                                    # the drusen-exclusion gate
    b = (p["loss6"] < frac * p["base"]) & p["core"] & keep_elev
    if "hyper6" in p and b.any():
        h = p["hyper6"]
        keep_thr = max(hyper_keep * float(np.percentile(h[p["core"]], 75)), 0.0)
        b = b & (h > keep_thr)                                              # criterion 1: require transmission
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            fill_thr = max(hyper_frac * float(np.percentile(h[b], 60)), 0.0)
            b = b | (holes & (h > fill_thr) & keep_elev)                    # criterion 2 (+ same elev gate)
    mask = fp.crora(binary_fill_holes(b) if fill_all_holes else b, min_diam_um)
    return mask, float(mask.sum()) * MMPP2


def e2e_lookup():
    out = {}
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("qc_status") != "ok":
                continue
            out[(r["subject"], r["eye"].upper())] = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return out


def plex_lookup():
    out = {}
    path = os.path.join(RESULTS_DIR, "plex_compare.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[(r["subject"], r["eye"].upper())] = float(r["plex_mm2"])
    return out


def fmtx(x):
    return "inf" if x == float("inf") else f"{x:g}"


def main():
    print(f"bm_dl.active()={bm_dl.active()} backend={bm_dl.backend()}", flush=True)
    e2e = e2e_lookup()
    plex = plex_lookup()

    recs = []   # (subject, eye, label, plex, ref_area, areas_by_X dict)
    for subject, eye, label in EYES:
        path = e2e.get((subject, eye))
        if path is None or not os.path.exists(path):
            print(f"  SKIP {subject} {eye}: e2e missing ({path})", flush=True)
            continue
        raw = e2e_source.open_e2e(path)
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol)
        p = oac_ga.prep(ov, bm, baseline="radial2")

        # REFERENCE = current production gate (radial2, DL BM). Must match X=inf byte-for-byte.
        ref_mask, ref_area = oac_ga.footprint(p, 0.50)

        # ---- RPE->BM elevation en-face (um), aligned to rpe6, per the recipe ----
        oac = mp.oac_volume(ov.vol)
        elev_nat = np.clip((bm - mp.band_argmax_row(oac, bm, *mp.OAC_RPE_UM)) * mp.AX, 0.0, None)  # (n,W) um
        elev6 = proj.to_enface(mp.destripe2d(elev_nat, signed=False), ov.fov_mm)

        # sanity: X=inf must reproduce the reference exactly (the elev gate is all-True).
        _, a_inf = footprint_elev(p, elev6, 0.50, float("inf"))
        assert abs(a_inf - ref_area) < 1e-9, \
            f"X=inf copy mismatch {subject} {eye}: {a_inf} vs {ref_area}"

        areas = {X: footprint_elev(p, elev6, 0.50, X)[1] for X in XS}
        recs.append((subject, eye, label, plex.get((subject, eye), float("nan")), ref_area, areas))
        print(f"  {subject[-7:]}_{eye} [{label}] plex={plex.get((subject, eye), float('nan')):.3f} "
              f"ref={ref_area:.3f}", flush=True)

    # ---- write results/gate_elev.csv ----
    out_csv = os.path.join(RESULTS_DIR, "gate_elev.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "eye", "label", "plex_mm2", "ref_area"]
                   + [f"elev_lt_{fmtx(X)}" for X in XS])
        for subject, eye, label, pl, ref_area, areas in recs:
            w.writerow([subject, eye, label, f"{pl:.4f}", f"{ref_area:.4f}"]
                       + [f"{areas[X]:.4f}" for X in XS])
    print(f"\nwrote {out_csv}", flush=True)

    # ---- full sweep table (area mm2 vs elevation cut X) ----
    print("\n==== ELEVATION-GATE SWEEP  (area mm2 ; gate = drop firing columns with RPE->BM elev >= X um) ====")
    print("eye".ljust(20) + "label".ljust(10) + "plex".rjust(6)
          + "".join(f"{fmtx(X):>8}" for X in XS))
    for subject, eye, label, pl, ref_area, areas in recs:
        print(f"{subject[-7:]+'_'+eye:20}{label:10}{pl:6.2f}"
              + "".join(f"{areas[X]:8.3f}" for X in XS))

    # ---- DECISION: find the LARGEST X (loosest cut) that drives 016 OD below 0.25, and the worst GA
    #      %-change there. Also the smallest. Report honestly if NONE clears 016. ----
    fp_rec = next((r for r in recs if r[2] == "FP"), None)
    ga_recs = [r for r in recs if r[2] in ("GA", "GA-faint", "GA-large")]
    ctrl_recs = [r for r in recs if r[2] == "control"]

    print("\n==== DECISION ====")
    if fp_rec is None:
        print("  016 OD (FP) not loaded -- cannot decide.")
        return

    _, _, _, _, fp_ref, fp_areas = fp_rec
    print(f"  016 OD reference area (no gate) = {fp_ref:.3f} mm2 ; target < 0.25 mm2")

    clearing_Xs = [X for X in XS if X != float("inf") and fp_areas[X] < 0.25]
    if not clearing_Xs:
        best_X = min((X for X in XS if X != float("inf")), key=lambda X: fp_areas[X])
        print(f"  *** NO elevation cut in the swept range clears 016 OD below 0.25. ***")
        print(f"      Tightest tried (X={fmtx(best_X)}um) only brings 016 to {fp_areas[best_X]:.3f} mm2.")
        print("      => 016 fires at LOW elevation (it is NOT drusen) -- this gate cannot work.")
    else:
        # 'best' = the LOOSEST (largest X) cut that still clears 016 -> least collateral GA damage.
        best_X = max(clearing_Xs)
        print(f"  X values that clear 016 (<0.25): {[fmtx(X) for X in clearing_Xs]}")
        print(f"  Loosest clearing cut: X={fmtx(best_X)}um -> 016 = {fp_areas[best_X]:.3f} mm2")

    # GA / control impact AT best_X (the cut we'd actually pick).
    print(f"\n  --- impact at the chosen cut X={fmtx(best_X)}um ---")
    print("  " + "eye".ljust(18) + "label".ljust(10) + "before".rjust(9) + "after".rjust(9)
          + "%change".rjust(9))
    worst_ga = 0.0
    worst_ga_eye = ""
    for subject, eye, label, pl, ref_area, areas in recs:
        before = areas[float("inf")]
        after = areas[best_X]
        if before > 1e-6:
            chg = 100.0 * (after - before) / before
        else:
            chg = 0.0
        tag = ""
        if label in ("GA", "GA-faint", "GA-large") and before > 1e-6:
            if abs(chg) > abs(worst_ga):
                worst_ga, worst_ga_eye = chg, f"{subject[-7:]}_{eye}"
        print(f"  {subject[-7:]+'_'+eye:18}{label:10}{before:9.3f}{after:9.3f}{chg:8.1f}%{tag}")

    print(f"\n  WORST must-hold-GA change at X={fmtx(best_X)}um: {worst_ga:+.1f}%  ({worst_ga_eye})")
    if clearing_Xs:
        cleared = fp_areas[best_X] < 0.25
        ga_safe = abs(worst_ga) <= 5.0
        print(f"  016 cleared (<0.25): {cleared} | worst GA within +-5%: {ga_safe}")
        if cleared and ga_safe:
            print("  => VIABLE elevation gate found.")
        elif cleared and not ga_safe:
            print("  => clears 016 but ERODES GA beyond 5% -- NOT viable without regression.")
    print("\n  (See the full sweep above for the GA erosion vs X trade-off across ALL cuts.)")


if __name__ == "__main__":
    main()
