#!/usr/bin/env python
"""Quantify the hypertransmission-locator candidates: can an ABSOLUTE threshold separate true GA from the
flat-control firing? (The production relative gate leaks on flat controls -> 016 OD FP.)

For each eye it computes the production measurement `core` (oac_ga.prep) + each hyper variant's en-face, and
reports the hyper LEVEL over the GA region (005 OD: the in-frame gold mask) vs the control field (016/006).
A variant is 'separable' if 005's GA-median exceeds every control field's p90 -> one fixed threshold works.

Run (repo root):  oct_env\\Scripts\\python.exe src\\hyper_probe.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("OCT_BM_DL", "1")

import csv  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, OUT_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402
from reader.core import projection as proj  # noqa: E402
from hyper_locator import to6, variants  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
GOLD = os.path.join(OUT_DIR, "ga_bscan_dataset", "labels")
# eye -> role; GA eyes use their in-frame gold mask if present, controls use the whole core field.
EYES = [("NHAMD-003-005", "V3", "OD", "GA"), ("NHAMD-003-008", "V1", "OS", "GA"),
        ("NHAMD-003-016", "V2", "OD", "CTRL"), ("NHAMD-003-006", "V3", "OS", "CTRL"),
        ("NHAMD-003-012", "V3", "OD", "CTRL")]


def gold_enface(subject, visit, eye, ov):
    rows, found, W = [], False, None
    for s in (f"{subject}-{visit}", subject):
        rows, found, W = [], False, None
        for i in range(ov.n_bscans):
            p = os.path.join(GOLD, f"{s}_{eye}_b{i:04d}.png")
            if os.path.exists(p):
                found = True
                col = (np.array(Image.open(p)) == 1).any(axis=0)
                W = len(col)
                rows.append(col)
            else:
                rows.append(None)
        if found:
            nat = np.array([(r if r is not None else np.zeros(W, bool)) for r in rows], np.float32)
            return proj.to_enface(nat, ov.fov_mm) > 0.5
    return None


def row_for(subject, visit, eye):
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("qc_status") or "").strip() == "ok" and subject in r["subject"] \
                    and r["visit"] == visit and r["eye"].upper() == eye:
                return r
    return None


def main():
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})", flush=True)
    names = ["v0_scalar", "v1_local", "v2_mich"]
    # store per-variant: ga medians (list) and control p90s (list)
    ga_med = {n: [] for n in names}
    ctrl_p90 = {n: [] for n in names}
    raw_cache = {}
    print(f"\n{'eye':16} {'role':5} " + " ".join(f"{n:>22}" for n in names))
    print(f"{'':16} {'':5} " + " ".join(f"{'GAmed/fieldp50  p90':>22}" for _ in names))
    for subject, visit, eye, role in EYES:
        r = row_for(subject, visit, eye)
        if r is None:
            print(f"  {subject[-6:]} {eye}: not qc_ok, skip"); continue
        e2e = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if e2e not in raw_cache:
            raw_cache.clear(); raw_cache[e2e] = e2e_source.open_e2e(e2e)
        ov = e2e_source.load_volume(raw_cache[e2e], e2e_source.default_volume_index(raw_cache[e2e], eye))
        bm = bm_dl.segment_volume(ov.vol)
        core = oac_ga.prep(ov, bm)["core"]
        gold = gold_enface(subject, visit, eye, ov) if role == "GA" else None
        vs = variants(ov.vol, bm)
        cells = []
        for n in names:
            e6 = proj.to_enface(to6(vs[n]), ov.fov_mm)
            if gold is not None and (core & gold).sum() > 20:
                sig = float(np.median(e6[core & gold]))                  # GA signal
                ga_med[n].append(sig)
                cells.append(f"{sig:7.3f} (GA)        ")
            else:
                p50 = float(np.median(e6[core])); p90 = float(np.percentile(e6[core], 90))
                if role == "CTRL":
                    ctrl_p90[n].append(p90)
                cells.append(f"{p50:6.3f} {p90:6.3f}      ")
        tag = f"{subject[-6:]} {eye}"
        print(f"  {tag:14} {role:5} " + " ".join(cells), flush=True)

    print("\n=== separability: can ONE absolute threshold put GA above EVERY control field p90? ===")
    for n in names:
        if not ga_med[n] or not ctrl_p90[n]:
            print(f"  {n:12}: insufficient data"); continue
        ga_lo = min(ga_med[n])             # weakest GA signal
        ctrl_hi = max(ctrl_p90[n])         # brightest control field tail
        gap = ga_lo - ctrl_hi
        margin = ga_lo / (ctrl_hi + 1e-9)
        verdict = "SEPARABLE" if gap > 0 else "OVERLAP"
        print(f"  {n:12}: min GA-median {ga_lo:.3f}  vs  max control-p90 {ctrl_hi:.3f}  "
              f"-> gap {gap:+.3f} (x{margin:.2f})  {verdict}")


if __name__ == "__main__":
    main()
