#!/usr/bin/env python
"""ITEM B PROBE — absolute hypertransmission floor for the OAC GA footprint gate.

Standalone experiment (does NOT edit reader/core/oac_ga.py or footprint.py). Imports the detector,
runs oac_ga.prep(baseline='radial2') with DL BM on the focused subset, and reports the hyper6
distribution over `core` for CONTROLS vs GA. Then it reimplements the footprint gate (start from the
EXACT hyper_keep/hyper_frac logic in oac_ga.footprint) with an added ABSOLUTE floor and compares the
resulting area to the reference oac_ga.footprint(p, 0.5)[1].

Run (from repo root):
  oct_env\\Scripts\\python.exe src\\probe_hyper_floor.py
"""
import csv
import os
import re
import sys

# --- imports (mirror src/oac_area.py): repo root + src/ on the path ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# BM = the DL model (per the workflow brief)
os.environ["OCT_BM_DL"] = "1"

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

import bm_dl
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import footprint as fp
from reader.core import projection as proj

MMPP2 = oac_ga.MMPP2

# ---- focused subset: (subject, visit, eye, label) ----
CONTROLS = [
    ("NHAMD-003-016-V2", "V2", "OD"),
    ("NHAMD-003-009-V2", "V2", "OS"),
    ("NHAMD-003-002-V2", "V2", "OD"),
    ("NHAMD-003-012-V3", "V3", "OD"),
    ("NHAMD-003-006-V3", "V3", "OS"),
]
GA = [
    ("NHAMD-003-005-V3", "V3", "OD"),
    ("NHAMD-003-005-V3", "V3", "OS"),
    ("NHAMD-003-008-V1", "V1", "OD"),
    ("NHAMD-003-008-V1", "V1", "OS"),
    ("NHAMD-003-015-V3", "V3", "OD"),
    ("NHAMD-003-003-V3", "V3", "OD"),
    ("NHAMD-003-011-V3", "V3", "OS"),
]


def plex_lookup():
    """subject,visit,eye -> plex_mm2 from results/plex_compare.csv."""
    out = {}
    with open(os.path.join(RESULTS_DIR, "plex_compare.csv"), newline="") as f:
        for r in csv.DictReader(f):
            out[(r["subject"], r["eye"].upper())] = float(r["plex_mm2"])
    return out


def e2e_lookup():
    """subject,eye -> absolute E2E path from the master pairing CSV (qc_status==ok)."""
    out = {}
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("qc_status") != "ok":
                continue
            out[(r["subject"], r["eye"].upper())] = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
    return out


def open_eye(e2e_path, eye):
    raw = e2e_source.open_e2e(e2e_path)
    idx = e2e_source.default_volume_index(raw, eye)
    ov = e2e_source.load_volume(raw, idx)
    return ov


# ----------------------------------------------------------------------------
# Reimplemented footprint gate, EXACTLY mirroring oac_ga.footprint's hyper_keep/hyper_frac lines,
# with an added ABSOLUTE FLOOR. `hyper_floor` (in hyper6 units) raises BOTH the criterion-1 keep
# threshold and the criterion-2 hole-fill threshold to be at least this absolute value, so a flat
# control whose relative percentile is tiny can no longer pass on noise.
def footprint_floor(p, frac=0.50, min_diam_um=250.0, hyper_fill=True, close_mm=0.15,
                    hyper_frac=0.7, hyper_keep=0.4, fill_all_holes=True, hyper_floor=0.0):
    b = (p["loss6"] < frac * p["base"]) & p["core"]
    if hyper_fill and "hyper6" in p and b.any():
        h = p["hyper6"]
        keep_thr = max(hyper_keep * float(np.percentile(h[p["core"]], 75)), hyper_floor)
        b = b & (h > keep_thr)                                  # criterion 1: require transmission
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            fill_thr = max(hyper_frac * float(np.percentile(h[b], 60)), hyper_floor)
            b = b | (holes & (h > fill_thr))                    # criterion 2: fill GA-like holes
    mask = fp.crora(binary_fill_holes(b) if fill_all_holes else b, min_diam_um)
    return mask, float(mask.sum()) * MMPP2


def stats_of(vals):
    return {
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def main():
    print(f"bm_dl.active() = {bm_dl.active()}  backend={bm_dl.backend()}  model={bm_dl.model_path()}")
    plex = plex_lookup()
    e2e = e2e_lookup()

    records = []
    for label, group in (("control", CONTROLS), ("ga", GA)):
        for subject, visit, eye in group:
            key = (subject, eye)
            e2e_path = e2e.get(key)
            if e2e_path is None or not os.path.exists(e2e_path):
                print(f"  SKIP {subject} {eye}: e2e missing ({e2e_path})")
                continue
            ov = open_eye(e2e_path, eye)
            bm = bm_dl.segment_volume(ov.vol)
            p = oac_ga.prep(ov, bm, baseline="radial2")
            h = p["hyper6"]
            core = p["core"]
            hc = h[core]
            st = stats_of(hc)
            # the within-eye relative keep threshold the reference gate actually uses:
            rel_keep = 0.4 * float(np.percentile(hc, 75))
            ref_mask, ref_area = oac_ga.footprint(p, 0.5)
            records.append(dict(subject=subject, eye=eye, label=label,
                                plex=plex.get(key, float("nan")),
                                ref_area=ref_area, rel_keep=rel_keep, p=p, **st))
            print(f"[{label:7}] {subject} {eye}  PLEX={plex.get(key, float('nan')):6.2f}  "
                  f"ref_area(DL,radial2,0.5)={ref_area:6.3f}  | hyper6 over core: "
                  f"med={st['median']:.4f} p75={st['p75']:.4f} p90={st['p90']:.4f} "
                  f"p95={st['p95']:.4f} max={st['max']:.4f}  | rel_keep(0.4*p75)={rel_keep:.4f}",
                  flush=True)

    # ---- separation analysis ----
    print("\n================ SEPARATION ANALYSIS (hyper6 over core) ================")
    ctrl = [r for r in records if r["label"] == "control"]
    ga = [r for r in records if r["label"] == "ga"]
    print("\nControls (PLEX=0, must NOT fire):")
    for r in ctrl:
        print(f"  {r['subject']} {r['eye']}: ref_area={r['ref_area']:6.3f}  "
              f"med={r['median']:.4f} p90={r['p90']:.4f} p95={r['p95']:.4f} max={r['max']:.4f}")
    print("\nGA eyes (must hold):")
    for r in ga:
        print(f"  {r['subject']} {r['eye']}: PLEX={r['plex']:.2f} ref_area={r['ref_area']:6.3f}  "
              f"med={r['median']:.4f} p90={r['p90']:.4f} p95={r['p95']:.4f} max={r['max']:.4f}")

    # The key question: does an absolute hyper6 value separate control-noise from real GA transmission?
    # Compare the hyper6 level INSIDE the eye's GA footprint (where the gate fires) across the two groups.
    print("\n---- hyper6 over the REFERENCE GA footprint (where the relative gate currently fires) ----")
    for r in records:
        p = r["p"]
        mask, _ = oac_ga.footprint(p, 0.5)
        if mask.any():
            hin = p["hyper6"][mask]
            print(f"  [{r['label']:7}] {r['subject']} {r['eye']}: fires {mask.sum():5d}px  "
                  f"hyper6@fire med={np.median(hin):.4f} p25={np.percentile(hin,25):.4f} "
                  f"p10={np.percentile(hin,10):.4f} min={hin.min():.4f}")
        else:
            print(f"  [{r['label']:7}] {r['subject']} {r['eye']}: fires 0px (no GA called)")

    # ---- floor sweep: apply the absolute floor, report control + GA areas ----
    print("\n================ FLOOR SWEEP (absolute hyper6 floor on the gate) ================")
    floors = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    hdr = "floor   " + "  ".join(f"{r['subject'][-7:]}_{r['eye']}" for r in records)
    print(hdr)
    print("labels: " + "  ".join(f"{r['label'][:4]:>11}" for r in records))
    print("PLEX:   " + "  ".join(f"{r['plex']:>11.2f}" for r in records))
    for fl in floors:
        areas = []
        for r in records:
            _, a = footprint_floor(r["p"], 0.5, hyper_floor=fl)
            areas.append(a)
        print(f"{fl:5.2f}   " + "  ".join(f"{a:11.3f}" for a in areas))

    # ---- pass/fail summary at candidate floors ----
    print("\n================ PASS/FAIL at candidate floors ================")
    # control target: area < 0.25; GA target: hold ref area (within ~10% or > 0.5*ref)
    ref = {(r["subject"], r["eye"]): r["ref_area"] for r in records}
    for fl in floors:
        ctrl_ok, ga_ok, lines = 0, 0, []
        for r in records:
            _, a = footprint_floor(r["p"], 0.5, hyper_floor=fl)
            if r["label"] == "control":
                ok = a < 0.25
                ctrl_ok += ok
            else:
                rf = ref[(r["subject"], r["eye"])]
                ok = (rf < 0.25) or (a >= 0.85 * rf)   # GA must not collapse
                ga_ok += ok
            lines.append(f"{r['subject'][-7:]}_{r['eye']}={a:.2f}{'' if ok else '!'}")
        print(f"floor={fl:.2f}  controls_ok={ctrl_ok}/{len(ctrl)}  ga_ok={ga_ok}/{len(ga)}   " +
              " ".join(lines))


if __name__ == "__main__":
    main()
