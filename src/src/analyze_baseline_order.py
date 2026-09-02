#!/usr/bin/env python
"""ANALYSIS-ONLY (does not change the pipeline): for every qc_ok eye, recompute the QUADRATIC-baseline
OAC GA mask on the DL BM and extract OCT-INTRINSIC shape descriptors (no PLEX) — area-fraction of the
in-core field, largest-component eccentricity, centroid offset from the field centre, component count —
to test whether a per-eye descriptor computable WITHOUT PLEX could pick linear-vs-quadratic per eye.

Mirrors src/compare_plex.py's loading exactly (DL BM, oac_ga.prep/footprint defaults) so the areas it
reports match plex_compare.csv's dl_quad/dl_lin. Output -> results/baseline_order_descriptors.csv.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from skimage import measure  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "baseline_order_descriptors.csv")


def shape_descriptors(mask, core):
    """OCT-intrinsic descriptors of a detected GA footprint within the measurement field `core`."""
    m = np.asarray(mask, bool)
    H, W = m.shape
    cf = np.asarray(core, bool)
    n_core = int(cf.sum())
    out = dict(area_px=int(m.sum()), core_px=n_core,
               area_frac=(m.sum() / n_core) if n_core else 0.0,
               ecc=0.0, centroid_off=0.0, n_comp=0, solidity=0.0, frac_largest=0.0,
               cx=0.0, cy=0.0)
    if not m.any():
        return out
    lbl = measure.label(m)
    props = measure.regionprops(lbl)
    props.sort(key=lambda r: -r.area)
    out["n_comp"] = len(props)
    big = props[0]
    out["ecc"] = float(big.eccentricity)            # 0 = round, ->1 = elongated
    out["solidity"] = float(big.solidity)
    out["frac_largest"] = float(big.area) / float(m.sum())   # confluence: 1 = one solid blob
    # centroid offset of the WHOLE footprint from the field centre, normalised by field half-width
    ys, xs = np.where(m)
    cy0, cx0 = np.array(np.where(cf)).mean(axis=1)   # field centre
    cyg, cxg = ys.mean(), xs.mean()
    out["cy"], out["cx"] = float(cyg), float(cxg)
    half = 0.5 * np.sqrt(n_core) if n_core else 1.0
    out["centroid_off"] = float(np.hypot(cyg - cy0, cxg - cx0) / (half + 1e-6))
    return out


def main():
    print(f"DL BM: {bm_dl.model_path()}  backend={bm_dl.backend()}", flush=True)
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("qc_status") or "").strip() == "ok"]
    print(f"{len(rows)} qc_ok eyes", flush=True)

    raw_cache = {}
    out_rows = []
    for r in rows:
        subj, visit, eye = r["subject"], r["visit"], r["eye"].upper()
        e2e_path = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e_path):
            print(f"  SKIP {subj} {eye}: E2E missing", flush=True)
            continue
        try:
            plex = float(r["advRPE_area_mm2"])
        except (TypeError, ValueError):
            plex = float("nan")
        if e2e_path not in raw_cache:
            raw_cache.clear()
            raw_cache[e2e_path] = e2e_source.open_e2e(e2e_path)
        raw = raw_cache[e2e_path]
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol, bs=2)

        pq = oac_ga.prep(ov, bm, trend_order=2)
        mq, aq = oac_ga.footprint(pq)
        pl = oac_ga.prep(ov, bm, trend_order=1)
        ml, al = oac_ga.footprint(pl)

        # descriptors from the QUAD mask (the proposed default order) — the signal a per-eye picker sees
        d = shape_descriptors(mq, pq["core"])
        rec = dict(eye=subj[-7:] + " " + eye, plex=round(plex, 4),
                   dl_quad=round(aq, 4), dl_lin=round(al, 4), **d)
        out_rows.append(rec)
        print(f"  {rec['eye']:12s} PLEX={plex:6.2f} q={aq:6.2f} l={al:6.2f}  "
              f"afrac={d['area_frac']:.3f} ecc={d['ecc']:.2f} off={d['centroid_off']:.2f} "
              f"ncomp={d['n_comp']} fbig={d['frac_largest']:.2f}", flush=True)

    cols = ["eye", "plex", "dl_quad", "dl_lin", "area_px", "core_px", "area_frac",
            "ecc", "solidity", "frac_largest", "centroid_off", "n_comp", "cx", "cy"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in out_rows:
            w.writerow({k: (round(x[k], 4) if isinstance(x.get(k), float) else x.get(k)) for k in cols})
    print(f"\nwrote {OUT_CSV}  ({len(out_rows)} eyes)", flush=True)


if __name__ == "__main__":
    main()
