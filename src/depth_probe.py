#!/usr/bin/env python
"""Is RPE-loss DEPTH a usable discriminator for the 016 FP, or does it collide with faint real GA (005 OS)?

A column is called GA where loss6 < frac*base (frac 0.5), i.e. loss/base < 0.5. The DEPTH of the loss =
how far below 0.5 it reaches. The 016 deep-think claimed 016 fires only SHALLOWLY (never <0.43) while true
GA reaches <0.31 -- but that it 'collides with faint 005 OS'. This probes, within each eye's production
footprint, the distribution of loss/base and the fraction of the lesion reaching DEEP loss, so we can see
whether a per-lesion depth requirement separates 016 (FP) from the faint-but-REAL 005 OS without killing it.

Run (repo root):  oct_env\\Scripts\\python.exe src\\depth_probe.py
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OCT_BM_DL", "1")

import numpy as np  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
# (subject, visit, eye, label) — GA truth vs the FP/controls
EYES = [("NHAMD-003-005", "V3", "OD", "GA gold"), ("NHAMD-003-005", "V3", "OS", "GA faint"),
        ("NHAMD-003-008", "V1", "OS", "GA large"), ("NHAMD-003-015", "V3", "OD", "GA"),
        ("NHAMD-003-011", "V3", "OD", "GA multifocal"),
        ("NHAMD-003-016", "V2", "OD", "FP control"), ("NHAMD-003-006", "V3", "OS", "ctrl"),
        ("NHAMD-003-012", "V3", "OD", "ctrl")]


def row_for(subject, visit, eye):
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("qc_status") or "").strip() == "ok" and subject in r["subject"] \
                    and r["visit"] == visit and r["eye"].upper() == eye:
                return r
    return None


def main():
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})", flush=True)
    print(f"\n{'eye':16} {'label':14} {'PLEX':>5} {'area':>5} | within-footprint loss/base depth")
    print(f"{'':16} {'':14} {'':5} {'':5} | {'min':>5} {'p5':>5} {'p25':>5} {'med':>5} "
          f"{'%<.40':>6} {'%<.35':>6} {'%<.31':>6}")
    raw_cache = {}
    rows_out = []
    for subject, visit, eye, label in EYES:
        r = row_for(subject, visit, eye)
        if r is None:
            print(f"  {subject[-6:]} {eye}: not qc_ok"); continue
        try:
            plex = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            plex = float("nan")
        e2e = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if e2e not in raw_cache:
            raw_cache.clear(); raw_cache[e2e] = e2e_source.open_e2e(e2e)
        ov = e2e_source.load_volume(raw_cache[e2e], e2e_source.default_volume_index(raw_cache[e2e], eye))
        bm = bm_dl.segment_volume(ov.vol)
        P = oac_ga.prep(ov, bm)
        mask, area = oac_ga.footprint(P)
        ratio = (P["loss6"] / np.maximum(P["base"], 1e-6))
        if mask.sum() < 5:
            print(f"  {subject[-6:]} {eye:2} {label:14} {plex:5.2f} {area:5.2f} | (no footprint)")
            continue
        v = ratio[mask]
        stats = (float(v.min()), float(np.percentile(v, 5)), float(np.percentile(v, 25)),
                 float(np.median(v)),
                 float((v < 0.40).mean() * 100), float((v < 0.35).mean() * 100),
                 float((v < 0.31).mean() * 100))
        rows_out.append((f"{subject[-6:]} {eye}", label, plex, area, stats))
        print(f"  {subject[-6:]} {eye:2} {label:14} {plex:5.2f} {area:5.2f} | "
              f"{stats[0]:5.2f} {stats[1]:5.2f} {stats[2]:5.2f} {stats[3]:5.2f} "
              f"{stats[4]:5.0f}% {stats[5]:5.0f}% {stats[6]:5.0f}%", flush=True)

    # decisive check: is there a depth rule (min loss/base, or % of lesion reaching deep) that puts
    # the FP 016 on the 'reject' side while keeping the faint real GA 005 OS on the 'keep' side?
    fp = next((s for n, l, p, a, s in rows_out if "016" in n), None)
    faint = next((s for n, l, p, a, s in rows_out if "005 OS" in n), None)
    if fp and faint:
        print("\n=== 016 (FP) vs 005 OS (faint REAL GA) ===")
        print(f"  016    min loss/base {fp[0]:.2f}   %<0.35 {fp[5]:.0f}%   %<0.31 {fp[6]:.0f}%")
        print(f"  005 OS min loss/base {faint[0]:.2f}   %<0.35 {faint[5]:.0f}%   %<0.31 {faint[6]:.0f}%")
        sep_min = faint[0] < fp[0]
        sep_frac = faint[5] > fp[5]
        print(f"  -> min-depth separates (005OS deeper than 016)?  {sep_min}")
        print(f"  -> %deep separates (005OS more deep-cols than 016)?  {sep_frac}")


if __name__ == "__main__":
    main()
