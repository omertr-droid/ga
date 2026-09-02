#!/usr/bin/env python
"""Before/after PLEX agreement for the NEW production defaults (radial2 baseline + hyper_abs floor).

Runs the detector on every qc_ok cohort eye (DL BM, annotation-free; discarded scans excluded automatically
by the qc_status==ok filter) under four configs to ATTRIBUTE the change, and reports full agreement stats vs
the PLEX advRPE reference, reusing summarize_plex's stat block (bias/MAE/RMSE/r/CCC/Bland-Altman LoA/within±1):

  old        = trend/quadratic baseline, no hyper floor    (the PREVIOUS production default)
  quad+floor = quadratic baseline + hyper_abs=0.10          (floor effect, holding baseline)
  radial2    = radial2 baseline, no floor                   (baseline effect, holding gate)
  NEW        = radial2 baseline + hyper_abs=0.10            (the NEW production default)

Output -> results/default_compare.csv + a printed before/after report.
Run (repo root):  oct_env\\Scripts\\python.exe src\\compare_default.py
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
OUT_CSV = os.path.join(RESULTS_DIR, "default_compare.csv")

# Production prep/footprint params held fixed; only baseline + hyper_abs vary across configs.
PREP = dict(reducer="mean", smooth_px=2.0, margin_mm=0.30, rpe_hi_pct=95.0, sig_frac=0.5, base_cap=1.15)
FP = dict(frac=0.50, min_diam_um=250.0, hyper_fill=True, close_mm=0.15, hyper_frac=0.7,
          hyper_keep=0.4, fill_all_holes=True)
CONFIGS = ["old", "quad_floor", "radial2", "new"]      # column order


def main():
    if not bm_dl.available():
        print("FATAL: DL BM model unavailable", flush=True)
        return
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})", flush=True)
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]
    print(f"{len(rows)} qc_ok eyes (discarded scans excluded)", flush=True)

    out, raw_cache = [], {}
    for r in rows:
        subj, eye = r["subject"], r["eye"].upper()
        e2e = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e):
            print(f"  SKIP {subj} {eye}: E2E missing", flush=True)
            continue
        try:
            plex = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            continue
        if e2e not in raw_cache:
            raw_cache.clear()
            raw_cache[e2e] = e2e_source.open_e2e(e2e)
        raw = raw_cache[e2e]
        ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
        bm = bm_dl.segment_volume(ov.vol)
        # 2 preps (trend / radial2), each footprinted with floor off / on -> 4 areas
        P_t = oac_ga.prep(ov, bm, baseline="trend", trend_order=2, **PREP)
        P_r = oac_ga.prep(ov, bm, baseline="radial2", **PREP)
        a_old = oac_ga.footprint(P_t, hyper_abs=0.0, **FP)[1]
        a_qf = oac_ga.footprint(P_t, hyper_abs=0.10, **FP)[1]
        a_r2 = oac_ga.footprint(P_r, hyper_abs=0.0, **FP)[1]
        a_new = oac_ga.footprint(P_r, hyper_abs=0.10, **FP)[1]
        rec = {"subject": subj, "eye": eye, "plex": plex,
               "old": a_old, "quad_floor": a_qf, "radial2": a_r2, "new": a_new}
        out.append(rec)
        print(f"  {subj[-6:]} {eye:2} PLEX={plex:6.2f}  old={a_old:6.2f} quad+fl={a_qf:6.2f} "
              f"r2={a_r2:6.2f} NEW={a_new:6.2f}  (old->new {a_new - a_old:+.2f})", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "eye", "plex", *CONFIGS])
        w.writeheader()
        for rec in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
    print(f"\nwrote {OUT_CSV}  ({len(out)} eyes)", flush=True)

    plex = np.array([rec["plex"] for rec in out])
    ctrl = plex < CONTROL_THR
    ga = ~ctrl

    def spec(col):
        a = np.array([rec[col] for rec in out])
        return f"{int(np.sum(a[ctrl] < CALL_THR))}/{int(ctrl.sum())}"

    labels = {"old": "OLD (quad, no floor)", "quad_floor": "quad + floor",
              "radial2": "radial2, no floor", "new": "NEW (radial2 + floor)"}
    print(f"\n================ PLEX agreement, all {len(out)} eyes "
          f"(GA {int(ga.sum())}, controls {int(ctrl.sum())}) ================")
    hdr = f"{'config':24} {'bias':>7} {'MAE':>6} {'RMSE':>6} {'r':>6} {'CCC':>6} {'±1mm²':>7} {'95% LoA':>16} {'spec':>6}"
    print(hdr)
    for col in CONFIGS:
        ours = np.array([rec[col] for rec in out])
        b = stat_block(plex, ours)
        print(f"{labels[col]:24} {b['bias']:+7.2f} {b['mae']:6.2f} {b['rmse']:6.2f} {b['pearson_r']:6.3f} "
              f"{ccc(plex, ours):6.3f} {b['within_1p0']:6.0f}% [{b['loa_lo']:+5.2f},{b['loa_hi']:+5.2f}] "
              f"{spec(col):>6}")

    print(f"\n---- GA-present only ({int(ga.sum())} eyes) ----")
    for col in CONFIGS:
        ours = np.array([rec[col] for rec in out])[ga]
        b = stat_block(plex[ga], ours)
        print(f"{labels[col]:24} bias {b['bias']:+.2f}  MAE {b['mae']:.2f}  r {b['pearson_r']:.3f}  "
              f"within±1 {b['within_1p0']:.0f}%")

    # per-eye movers (old -> new), sorted by |delta|
    print("\n---- biggest per-eye changes (old -> NEW) ----")
    movers = sorted(out, key=lambda rc: -abs(rc["new"] - rc["old"]))[:10]
    print(f"{'eye':12} {'PLEX':>6} {'old':>6} {'NEW':>6} {'d(n-o)':>7} {'|old-PLEX|':>10} {'|new-PLEX|':>10}")
    for rc in movers:
        print(f"{rc['subject'][-6:] + ' ' + rc['eye']:12} {rc['plex']:6.2f} {rc['old']:6.2f} {rc['new']:6.2f} "
              f"{rc['new'] - rc['old']:+7.2f} {abs(rc['old'] - rc['plex']):10.2f} {abs(rc['new'] - rc['plex']):10.2f}")

    # controls
    print("\n---- control eyes (PLEX < 0.05) : old -> NEW ----")
    for rc in out:
        if rc["plex"] < CONTROL_THR:
            print(f"  {rc['subject'][-6:]} {rc['eye']:2} PLEX={rc['plex']:.2f}  old={rc['old']:.2f} "
                  f"NEW={rc['new']:.2f}  {'OK' if rc['new'] < CALL_THR else 'FP'}")


if __name__ == "__main__":
    main()
