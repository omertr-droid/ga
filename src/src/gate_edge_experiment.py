#!/usr/bin/env python
"""PHASE-2 GATE (B) — EDGE / SNR / VIGNETTE gate head-to-head on the OAC footprint.

The lone remaining false-positive control is 016 OD (PLEX 0, radial2 reads ~1.31 mm2). Phase-1
(src/firing_feature_probe.py) established that 016's firing is LOW-elevation (not drusen), GENUINELY
bright sub-BM (so hyper_abs cannot touch it), and the only NAMED axis where it differs from a focal GA
eye is field-edge proximity -- but that axis COLLIDES with the faint/eccentric must-hold GA (005 OS,
015 OD). This script tests gate (B) RIGOROUSLY by rebuilding the production footprint boolean and ANDing
two extra gates, swept head-to-head:

  (Y) whole-column SNR / vignette gate  -- tighten the existing sig_frac:  keep only sig6 > Y*pctl50(sig6[core])
  (M) field-edge margin gate            -- drop firing within M mm of the in-field (core) edge.

For 016 OD (the FP) and each must-hold GA eye (005 OD gold, 005 OS faint-erase-risk, 008 OD large
eccentric, 015 OD) + the stay-clean controls, we report area vs (Y, M). We look for ANY (Y, M) that
drives 016 < 0.25 WITHOUT eroding the GA eyes (esp. 005 OS faint + 008 OD whose lesion may approach the
field edge). The script is HONEST: it flags when clearing 016 also trims a real eccentric lesion.

HARD RULE: does NOT edit reader/core/oac_ga.py -- it reimplements the footprint gate byte-identically
(at Y=0, M=0 it reproduces oac_ga.footprint(p,0.5) exactly, asserted) and folds the two extra gates in
on the boolean `b` BEFORE crora, then re-runs crora + area exactly as production.

Run (from repo root):
  oct_env\\Scripts\\python.exe src\\gate_edge_experiment.py
Writes results/gate_edge.csv (eye, label, plex, ref_area, area@each (Y,M)) + prints the sweep tables and
the pass/fail verdict.
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
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes,
                           distance_transform_edt, gaussian_filter)

import bm_dl
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj

MMPP = oac_ga.MMPP
MMPP2 = oac_ga.MMPP2

# (subject, eye, label). FP = the target; ga = MUST-HOLD true GA; control = stay-clean.
EYES = [
    ("NHAMD-003-016-V2", "OD", "FP"),        # PLEX 0, radial2 ~1.31 -> drive < 0.25
    ("NHAMD-003-005-V3", "OD", "ga"),        # gold Dice 0.940, PLEX 1.08 -- robust central
    ("NHAMD-003-005-V3", "OS", "ga-faint"),  # faint erase-risk, near-edge; PLEX 0.57
    ("NHAMD-003-008-V1", "OD", "ga-large"),  # large eccentric, lesion may approach edge; PLEX 13.78
    ("NHAMD-003-015-V3", "OD", "ga"),        # PLEX 1.99, eccentric
    ("NHAMD-003-002-V2", "OS", "control"),   # stay-clean
    ("NHAMD-003-006-V3", "OS", "control"),   # stay-clean (the TRUE edge-artifact control)
    ("NHAMD-003-012-V3", "OD", "control"),   # stay-clean (reads 0)
]

# Y = SNR multiple of the in-field median whole-col intensity (sig_frac is 0.5 in production prep, but
# that gate already ran INSIDE core; here Y is an ADDITIONAL tightening on the FIRING pixels). Y=0 -> off.
Y_SWEEP = [0.0, 0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.20]
# M = field-edge margin in mm: drop firing whose distance to the core (in-field) edge is < M. M=0 -> off.
M_SWEEP = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]


def footprint_edge(p, sig6, edist6, frac=0.50, Y=0.0, M=0.0,
                   min_diam_um=250.0, close_mm=0.15, hyper_frac=0.7, hyper_keep=0.4,
                   fill_all_holes=True):
    """EXACT copy of oac_ga.footprint's gate logic, with two EXTRA gates ANDed onto the firing boolean
    `b` BEFORE the cRORA morphology:
      (Y) sig6 > Y * pctl50(sig6[core])   -- whole-column SNR / vignette gate (Y=0 -> no-op)
      (M) edist6 >= M                     -- field-edge margin (drop firing within M mm of core edge; M=0 -> no-op)
    Y=0 and M=0 -> byte-identical to oac_ga.footprint(p, frac).

    NOTE ON ORDER (faithful to how a real gate would act): the extra gates are applied to the SAME `b`
    that the hyper criteria build on, so the hole-fill (criterion 2) operates on the gated seed -- i.e.
    we gate the whole detection, not just trim the final mask. This is the honest, production-shaped test
    (a gate added inside prep/footprint would behave this way)."""
    core = p["core"]
    b = (p["loss6"] < frac * p["base"]) & core
    # ---- the two extra gates (applied to the firing seed) ----
    if Y > 0.0:
        thr = Y * float(np.nanpercentile(sig6[core], 50))
        b = b & (sig6 > thr)
    if M > 0.0:
        b = b & (edist6 >= M)
    # ---- the production hyper criteria, verbatim from oac_ga.footprint ----
    if "hyper6" in p and b.any():
        h = p["hyper6"]
        keep_thr = hyper_keep * float(np.percentile(h[core], 75))
        b = b & (h > keep_thr)
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            fill_thr = hyper_frac * float(np.percentile(h[b], 60))
            filled = holes & (h > fill_thr)
            # the extra gates must also constrain the hole-fills (else M/Y leak back in via the fill)
            if Y > 0.0:
                filled = filled & (sig6 > Y * float(np.nanpercentile(sig6[core], 50)))
            if M > 0.0:
                filled = filled & (edist6 >= M)
            b = b | filled
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


def main():
    print(f"bm_dl.active()={bm_dl.active()} backend={bm_dl.backend()}", flush=True)
    e2e = e2e_lookup()
    plex = plex_lookup()

    recs = []   # (subject, eye, label, plex, ref_area, area[(Y,M)] dict)
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

        # whole-column SNR en-face (smoothed exactly like prep's vignette gate) + field-edge distance (mm).
        sig6 = gaussian_filter(proj.to_enface(ov.vol.mean(axis=1).astype(np.float32), ov.fov_mm), 2.0)
        edist6 = distance_transform_edt(p["core"]) * MMPP    # mm from the in-field (core) edge

        # REFERENCE = production gate. Our Y=0,M=0 copy MUST reproduce it byte-for-byte.
        ref_mask, ref_area = oac_ga.footprint(p, 0.5)
        _, a00 = footprint_edge(p, sig6, edist6, frac=0.5, Y=0.0, M=0.0)
        assert abs(a00 - ref_area) < 1e-9, f"Y=0,M=0 copy mismatch {subject} {eye}: {a00} vs {ref_area}"

        areas = {}
        for Y in Y_SWEEP:
            for M in M_SWEEP:
                areas[(Y, M)] = footprint_edge(p, sig6, edist6, frac=0.5, Y=Y, M=M)[1]
        recs.append((subject, eye, label, plex.get((subject, eye), float("nan")), ref_area, areas))
        print(f"  {subject[-7:]}_{eye} [{label}] plex={plex.get((subject, eye), float('nan')):.3f} "
              f"ref={ref_area:.3f}  field_reach={edist6[p['core']].max():.2f}mm", flush=True)

    # ---- write results/gate_edge.csv (long form: one row per eye x (Y,M)) ----
    out_csv = os.path.join(RESULTS_DIR, "gate_edge.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "eye", "label", "plex_mm2", "ref_area", "Y_snr", "M_edge_mm", "area_mm2"])
        for subject, eye, label, pl, ref_area, areas in recs:
            for (Y, M), a in sorted(areas.items()):
                w.writerow([subject, eye, label, f"{pl:.4f}", f"{ref_area:.4f}",
                            f"{Y:g}", f"{M:g}", f"{a:.4f}"])
    print(f"\nwrote {out_csv}", flush=True)

    # ============================ TABLE 1: SNR-only sweep (M=0) ============================
    print("\n==== SNR-ONLY SWEEP (M=0): area mm2 vs Y ====")
    print("eye".ljust(20) + "label".ljust(10) + "plex".rjust(6) + "".join(f"Y={y:g}".rjust(8) for y in Y_SWEEP))
    for subject, eye, label, pl, ref_area, areas in recs:
        print(f"{subject[-7:]+'_'+eye:20}{label:10}{pl:6.2f}"
              + "".join(f"{areas[(y, 0.0)]:8.3f}" for y in Y_SWEEP))

    # ============================ TABLE 2: EDGE-only sweep (Y=0) ============================
    print("\n==== EDGE-ONLY SWEEP (Y=0): area mm2 vs M (mm) ====")
    print("eye".ljust(20) + "label".ljust(10) + "plex".rjust(6) + "".join(f"M={m:g}".rjust(8) for m in M_SWEEP))
    for subject, eye, label, pl, ref_area, areas in recs:
        print(f"{subject[-7:]+'_'+eye:20}{label:10}{pl:6.2f}"
              + "".join(f"{areas[(0.0, m)]:8.3f}" for m in M_SWEEP))

    # ============================ TABLE 3: full (Y,M) grid for the FP + the danger GA eyes =====
    danger = {("NHAMD-003-016-V2", "OD"), ("NHAMD-003-005-V3", "OS"),
              ("NHAMD-003-008-V1", "OD"), ("NHAMD-003-015-V3", "OD"),
              ("NHAMD-003-005-V3", "OD")}
    for subject, eye, label, pl, ref_area, areas in recs:
        if (subject, eye) not in danger:
            continue
        print(f"\n==== (Y,M) GRID  {subject[-7:]}_{eye} [{label}] plex={pl:.2f} ref={ref_area:.3f} ====")
        print("M\\Y".ljust(7) + "".join(f"{y:g}".rjust(8) for y in Y_SWEEP))
        for M in M_SWEEP:
            print(f"{M:<7g}" + "".join(f"{areas[(y, M)]:8.3f}" for y in Y_SWEEP))

    # ============================ VERDICT: best (Y,M) clearing 016 without GA regression =======
    fp_key = ("NHAMD-003-016-V2", "OD")
    fp_rec = next((r for r in recs if (r[0], r[1]) == fp_key), None)
    ga_recs = [r for r in recs if r[2].startswith("ga")]
    ctrl_recs = [r for r in recs if r[2] == "control"]

    print("\n==== VERDICT: search (Y,M) for 016<0.25 with GA held & controls clean ====")
    print("A GA eye is 'eroded' if its area drops >10% vs ref (the must-hold tolerance).")
    best = None    # (max_ga_regression, Y, M, fp_area)
    survivable = []
    for Y in Y_SWEEP:
        for M in M_SWEEP:
            if Y == 0.0 and M == 0.0:
                continue   # the no-op reference
            fp_area = fp_rec[5][(Y, M)]
            if fp_area >= 0.25:
                continue   # 016 not cleared
            # worst GA regression (relative drop) and worst control area at this setting
            worst_ga = 0.0
            worst_ga_eye = ""
            for s, e, lab, pl, ref, areas in ga_recs:
                if ref > 1e-6:
                    drop = (ref - areas[(Y, M)]) / ref
                    if drop > worst_ga:
                        worst_ga, worst_ga_eye = drop, f"{s[-7:]}_{e}"
            worst_ctrl = max((areas[(Y, M)] for s, e, lab, pl, ref, areas in ctrl_recs), default=0.0)
            survivable.append((Y, M, fp_area, worst_ga, worst_ga_eye, worst_ctrl))
            if best is None or worst_ga < best[0]:
                best = (worst_ga, Y, M, fp_area, worst_ga_eye, worst_ctrl)

    if not survivable:
        print("  NO (Y,M) setting drives 016 OD below 0.25. The edge/SNR/vignette gate CANNOT clear it.")
        # show how close the best-case gets + what it costs the worst GA eye
        approx = min(((fp_rec[5][(Y, M)], Y, M) for Y in Y_SWEEP for M in M_SWEEP
                      if not (Y == 0 and M == 0)), key=lambda t: t[0])
        a, Y, M = approx
        worst_ga, worst_eye = 0.0, ""
        for s, e, lab, pl, ref, areas in ga_recs:
            if ref > 1e-6:
                drop = (ref - areas[(Y, M)]) / ref
                if drop > worst_ga:
                    worst_ga, worst_eye = drop, f"{s[-7:]}_{e}"
        print(f"  closest: Y={Y:g} M={M:g} -> 016={a:.3f} (still >=0.25); "
              f"at that setting worst GA drop = {100*worst_ga:.0f}% ({worst_eye})")
    else:
        print(f"  {len(survivable)} (Y,M) setting(s) drive 016 < 0.25. Ranked by SMALLEST GA regression:")
        survivable.sort(key=lambda t: t[3])
        print("  Y     M     016_area   worstGA_drop  (eye)            worst_ctrl")
        for Y, M, fp_area, worst_ga, worst_eye, worst_ctrl in survivable[:12]:
            flag = "  <-- CLEAN" if (worst_ga <= 0.10 and worst_ctrl < 0.25) else ""
            print(f"  {Y:<5g} {M:<5g} {fp_area:8.3f}   {100*worst_ga:9.0f}%   {worst_eye:14}  "
                  f"{worst_ctrl:7.3f}{flag}")
        wga, Y, M, fa, weye, wctrl = best
        clean = wga <= 0.10 and wctrl < 0.25
        print(f"\n  BEST: Y={Y:g} M={M:g} -> 016={fa:.3f}; worst GA drop {100*wga:.0f}% ({weye}); "
              f"worst control {wctrl:.3f}. "
              + ("CLEAN (GA held <=10%, controls clean)." if clean
                 else "NOT CLEAN -- clearing 016 erodes a real GA eye / lifts a control."))


if __name__ == "__main__":
    main()
