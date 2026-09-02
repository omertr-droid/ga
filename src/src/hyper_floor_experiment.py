#!/usr/bin/env python
"""ITEM B EXPERIMENT — absolute hypertransmission FLOOR on the OAC footprint gate.

oac_ga.footprint requires sub-BM hypertransmission via a RELATIVE gate:
    criterion 1 (keep):  h > hyper_keep * percentile(h[core], 75)          (hyper_keep=0.4)
    criterion 2 (fill):  holes & (h > hyper_frac * percentile(h[b], 60))   (hyper_frac=0.7)
On a FLAT control eye (no GA, no real transmission) the percentile is itself tiny -> the relative
threshold is tiny -> NOISE passes it -> false-positive GA (016 OD, 009 OS, 002 OD).

This script adds an ABSOLUTE physical floor on the per-eye scalar-normalised sub-BM intensity channel
hyper6 (= m3_slab.hyper_enface destriped + to_enface + gaussian, returned by oac_ga.prep). It does NOT
edit oac_ga.py / footprint.py — it reimplements the gate with the floor and compares to the reference
oac_ga.footprint(p, 0.5)[1] (byte-for-byte the current relative gate).

The floor enters BOTH criteria as a max():
    keep_thr = max(hyper_keep * pctl(h[core],75), hyper_floor)
    fill_thr = max(hyper_frac * pctl(h[b],   60), hyper_floor)

hyper_floor = 0.0 reproduces the reference exactly (the max() is a no-op).

Run (from repo root):
  oct_env\\Scripts\\python.exe src\\hyper_floor_experiment.py
Writes results/hyper_floor.csv (eye, label, plex, ref_area, new_area@each floor) + prints the
before/after tables (3 control FPs + the GA eyes).
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
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj

MMPP2 = oac_ga.MMPP2

# Focused subset from the workflow. (subject, eye, label, expected_ref) — controls must go <0.25,
# GA must hold within ~5%.
EYES = [
    # ---- controls (PLEX 0; should read ~0) ----
    ("NHAMD-003-016-V2", "OD", "control"),
    ("NHAMD-003-009-V2", "OS", "control"),
    ("NHAMD-003-002-V2", "OD", "control"),
    ("NHAMD-003-012-V3", "OD", "control"),
    ("NHAMD-003-006-V3", "OS", "control"),
    # ---- GA (must hold) ----
    ("NHAMD-003-005-V3", "OD", "ga"),     # gold-Dice eye; PLEX 1.08
    ("NHAMD-003-005-V3", "OS", "ga"),     # PLEX 0.57
    ("NHAMD-003-008-V1", "OD", "ga"),     # PLEX 13.78
    ("NHAMD-003-008-V1", "OS", "ga"),     # PLEX 15.06
    ("NHAMD-003-015-V3", "OD", "ga"),     # PLEX 1.99
    ("NHAMD-003-003-V3", "OD", "ga"),     # PLEX 2.78 (already over-calls)
    ("NHAMD-003-011-V3", "OS", "ga"),     # PLEX 2.10
]

FLOORS = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]


def footprint_floor(p, frac, hyper_floor, hyper_frac=0.7, hyper_keep=0.4, close_mm=0.15,
                    min_diam_um=250.0, fill_all_holes=True):
    """EXACT copy of oac_ga.footprint's gate logic, with an ABSOLUTE floor `hyper_floor` folded into
    both hyper thresholds via max(). hyper_floor=0.0 -> byte-identical to oac_ga.footprint(p, frac)."""
    b = (p["loss6"] < frac * p["base"]) & p["core"]
    if "hyper6" in p and b.any():
        h = p["hyper6"]
        keep_thr = max(hyper_keep * float(np.percentile(h[p["core"]], 75)), hyper_floor)   # crit 1
        b = b & (h > keep_thr)
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            fill_thr = max(hyper_frac * float(np.percentile(h[b], 60)), hyper_floor)         # crit 2
            b = b | (holes & (h > fill_thr))
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

    recs = []   # (subject, eye, label, plex_mm2, p, ref_area, areas_by_floor dict)
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

        # REFERENCE = current relative gate (oac_ga.footprint) -- the ground truth we must match at floor 0.
        ref_mask, ref_area = oac_ga.footprint(p, 0.5)
        # sanity: our floor=0 copy must reproduce the reference byte-for-byte.
        _, a0 = footprint_floor(p, 0.5, 0.0)
        assert abs(a0 - ref_area) < 1e-9, f"floor=0 copy mismatch {subject} {eye}: {a0} vs {ref_area}"

        areas = {fl: footprint_floor(p, 0.5, fl)[1] for fl in FLOORS}
        recs.append((subject, eye, label, plex.get((subject, eye), float("nan")), p, ref_area, areas))
        print(f"  {subject[-7:]}_{eye} [{label}] plex={plex.get((subject, eye), float('nan')):.3f} "
              f"ref={ref_area:.3f}", flush=True)

    # ---- write results/hyper_floor.csv ----
    out_csv = os.path.join(RESULTS_DIR, "hyper_floor.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "eye", "label", "plex_mm2", "ref_area"] + [f"floor_{fl:g}" for fl in FLOORS])
        for subject, eye, label, pl, p, ref_area, areas in recs:
            w.writerow([subject, eye, label, f"{pl:.4f}", f"{ref_area:.4f}"]
                       + [f"{areas[fl]:.4f}" for fl in FLOORS])
    print(f"\nwrote {out_csv}", flush=True)

    # ---- full sweep table ----
    print("\n==== FULL FLOOR SWEEP (area mm2) ====")
    print("eye".ljust(20) + "label".ljust(9) + "plex".rjust(7) + "".join(f"{fl:>8.2f}" for fl in FLOORS))
    for subject, eye, label, pl, p, ref_area, areas in recs:
        print(f"{subject[-7:]+'_'+eye:20}{label:9}{pl:7.2f}"
              + "".join(f"{areas[fl]:8.3f}" for fl in FLOORS))

    # ---- the decision tables (BEFORE = floor 0 = ref ; AFTER at a few candidate floors) ----
    cand = [0.06, 0.08, 0.10, 0.12]

    def block(title, want):
        print(f"\n==== {title} ====")
        hdr = "eye".ljust(20) + "plex".rjust(6) + "before".rjust(9) + "".join(f"  @{c:g}".rjust(9) for c in cand)
        print(hdr)
        for subject, eye, label, pl, p, ref_area, areas in recs:
            if label != want:
                continue
            print(f"{subject[-7:]+'_'+eye:20}{pl:6.2f}{areas[0.0]:9.3f}"
                  + "".join(f"{areas[c]:9.3f}" for c in cand))

    block("CONTROLS (target: area < 0.25 at the chosen floor)", "control")
    block("GA (target: hold within ~5% of before)", "ga")

    # ---- pass/fail at each candidate floor ----
    print("\n==== PASS/FAIL per candidate floor (controls<0.25 AND GA within 5%) ====")
    for c in cand:
        ctrl_ok = all(areas[c] < 0.25 for s, e, lab, pl, p, ra, areas in recs if lab == "control")
        ga_ok, ga_worst = True, 0.0
        for s, e, lab, pl, p, ra, areas in recs:
            if lab != "ga":
                continue
            before = areas[0.0]
            if before > 1e-6:
                rel = abs(areas[c] - before) / before
                ga_worst = max(ga_worst, rel)
                if rel > 0.05:
                    ga_ok = False
        n_ctrl_fp = sum(1 for s, e, lab, pl, p, ra, areas in recs if lab == "control" and areas[c] >= 0.25)
        print(f"  floor={c:.2f}: controls_all<0.25={ctrl_ok} (#FP>=0.25: {n_ctrl_fp}) | "
              f"GA_all_within5%={ga_ok} (worst GA rel-change={100*ga_worst:.1f}%)")


if __name__ == "__main__":
    main()
