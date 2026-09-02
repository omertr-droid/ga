#!/usr/bin/env python
"""Full-cohort test of the new oac_ga `baseline='radial2'` (PCR) option vs linear and quadratic.

Runs the PRODUCTION detector (reader.core.oac_ga.detect, the radial2 option now wired into prep) on every
qc_ok cohort eye with the DL BM model (annotation-free, BM-consistent), and reports cohort agreement vs the
PLEX advRPE reference for all three baselines. Output -> results/cohort_radial2.csv + a printed summary.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\cohort_radial2.py
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
OUT_CSV = os.path.join(RESULTS_DIR, "cohort_radial2.csv")
CONTROL_THR = 0.05
CALL_THR = 0.25


def agg(plex, ours):
    plex, ours = np.asarray(plex, float), np.asarray(ours, float)
    d = ours - plex
    ctrl = plex < CONTROL_THR
    spec = int(np.sum(ours[ctrl] < CALL_THR))
    return {
        "MAE": float(np.abs(d).mean()), "bias": float(d.mean()),
        "within1": float(np.mean(np.abs(d) <= 1.0) * 100),
        "maxAE": float(np.abs(d).max()),
        "spec": f"{spec}/{int(ctrl.sum())}",
    }


def main():
    if not bm_dl.available():
        print("FATAL: DL BM model unavailable", flush=True)
        return
    print(f"DL BM: {bm_dl.model_path()} ({bm_dl.backend()})", flush=True)
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]

    out, raw_cache = [], {}
    for r in rows:
        subj, eye = r["subject"], r["eye"].upper()
        e2e = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e):
            print(f"  SKIP {subj} {eye}: E2E missing", flush=True); continue
        try:
            plex = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            continue
        if e2e not in raw_cache:
            raw_cache.clear(); raw_cache[e2e] = e2e_source.open_e2e(e2e)
        raw = raw_cache[e2e]
        ov = e2e_source.load_volume(raw, e2e_source.default_volume_index(raw, eye))
        bm = bm_dl.segment_volume(ov.vol)
        a_lin = oac_ga.detect(ov, bm, trend_order=1)[2]
        a_quad = oac_ga.detect(ov, bm, trend_order=2)[2]
        a_r2 = oac_ga.detect(ov, bm, baseline="radial2")[2]
        out.append({"subject": subj, "eye": eye, "plex": plex,
                    "linear": a_lin, "quad": a_quad, "radial2": a_r2})
        print(f"  {subj[-6:]} {eye:2} PLEX={plex:6.2f}  lin={a_lin:6.2f} quad={a_quad:6.2f} "
              f"radial2={a_r2:6.2f}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "eye", "plex", "linear", "quad", "radial2"])
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {OUT_CSV}  ({len(out)} eyes)", flush=True)

    plex = np.array([r["plex"] for r in out])
    ga = plex >= CONTROL_THR
    print(f"\n=== ALL {len(out)} eyes ===")
    print(f"{'method':8} {'MAE':>6} {'bias':>7} {'within1':>8} {'maxAE':>7} {'ctrl_spec':>10}")
    for m in ("linear", "quad", "radial2"):
        a = agg(plex, [r[m] for r in out])
        print(f"{m:8} {a['MAE']:6.3f} {a['bias']:+7.3f} {a['within1']:7.1f}% {a['maxAE']:7.2f} {a['spec']:>10}")
    print(f"\n=== GA-present only ({int(ga.sum())} eyes) ===")
    for m in ("linear", "quad", "radial2"):
        a = agg(plex[ga], np.array([r[m] for r in out])[ga])
        print(f"{m:8} {a['MAE']:6.3f} {a['bias']:+7.3f} {a['within1']:7.1f}% {a['maxAE']:7.2f}")


if __name__ == "__main__":
    main()
