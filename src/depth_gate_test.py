#!/usr/bin/env python
"""Tune the COMPLETE-loss depth gate (footprint min_depth): sweep thresholds in ONE pass.

For every qc_ok eye (DL BM, radial2 + hyper_abs defaults) computes the area at several min_depth values and
reports PLEX agreement + control specificity for each, plus the SENSITIVE eyes (016 = the FP to kill; 005 OS
+ 011 OS = faint/near-perfect real GA to preserve). Goal: the GENTLEST threshold that still zeroes 016 while
shaving the least real GA. prep() is computed once per eye; only the cheap footprint() varies.

Run (repo root):  oct_env\\Scripts\\python.exe src\\depth_gate_test.py
Output -> results/depth_gate.csv + report.
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
from summarize_plex import CALL_THR, CONTROL_THR, ccc, stat_block  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "depth_gate.csv")
THRESH = [None, 0.33, 0.35, 0.36, 0.37]


def col(t):
    return "now" if t is None else f"d{t:.2f}"


def main():
    if not bm_dl.available():
        print("FATAL: DL BM unavailable"); return
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})  thresholds={THRESH}", flush=True)
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]
    print(f"{len(rows)} qc_ok eyes", flush=True)

    out, raw_cache = [], {}
    for r in rows:
        subj, eye = r["subject"], r["eye"].upper()
        e2e = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e):
            print(f"  SKIP {subj} {eye}: E2E missing"); continue
        try:
            plex = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            continue
        if e2e not in raw_cache:
            raw_cache.clear(); raw_cache[e2e] = e2e_source.open_e2e(e2e)
        ov = e2e_source.load_volume(raw_cache[e2e], e2e_source.default_volume_index(raw_cache[e2e], eye))
        bm = bm_dl.segment_volume(ov.vol)
        P = oac_ga.prep(ov, bm)
        rec = {"subject": subj, "eye": eye, "plex": plex}
        for t in THRESH:
            rec[col(t)] = oac_ga.footprint(P, min_depth=t)[1]
        out.append(rec)
        print(f"  {subj[-6:]} {eye:2} PLEX={plex:6.2f}  " +
              " ".join(f"{col(t)}={rec[col(t)]:5.2f}" for t in THRESH), flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "eye", "plex", *[col(t) for t in THRESH]])
        w.writeheader()
        for rec in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
    print(f"\nwrote {OUT_CSV}", flush=True)

    plex = np.array([rc["plex"] for rc in out])
    ctrl = plex < CONTROL_THR

    def spec(c):
        a = np.array([rc[c] for rc in out])
        return f"{int(np.sum(a[ctrl] < CALL_THR))}/{int(ctrl.sum())}"

    def get(subj_sub, eye, c):
        for rc in out:
            if subj_sub in rc["subject"] and rc["eye"] == eye:
                return rc[c]
        return float("nan")

    print(f"\n=== PLEX agreement, {len(out)} eyes (controls {int(ctrl.sum())}) ===")
    print(f"{'config':16} {'bias':>7} {'MAE':>6} {'r':>6} {'CCC':>6} {'±1mm²':>7} {'spec':>6} "
          f"| {'016(FP)':>8} {'005OS':>6} {'011OS':>6}")
    for t in THRESH:
        c = col(t)
        ours = np.array([rc[c] for rc in out])
        b = stat_block(plex, ours)
        name = "no gate" if t is None else f"min_depth {t:.2f}"
        print(f"{name:16} {b['bias']:+7.2f} {b['mae']:6.2f} {b['pearson_r']:6.3f} {ccc(plex, ours):6.3f} "
              f"{b['within_1p0']:6.0f}% {spec(c):>6} | {get('016', 'OD', c):8.2f} "
              f"{get('005', 'OS', c):6.2f} {get('011', 'OS', c):6.2f}")
    print("\n(016 must drop < 0.25; 005 OS PLEX 0.57, 011 OS PLEX 2.10 -> keep high.)")


if __name__ == "__main__":
    main()
