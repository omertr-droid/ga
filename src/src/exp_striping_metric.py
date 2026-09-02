#!/usr/bin/env python
"""ITEM A probe — per-eye STRIPING / low-SNR quality metric for the OAC RPE-loss en-face.

STANDALONE (does NOT edit oac_ga.py). Computes a cheap per-eye QUALITY metric from the OAC RPE-loss
NATIVE map (rpe_nat, the same object oac_ga.prep destripes) over the whole qc_ok cohort, then tests
whether the high-striping eyes coincide with the worst |ours-PLEX| agreement eyes.

The metric does NOT need DL detection: striping is an acquisition property, so we anchor the OAC band
with the eye's own (device/auto) BM via oac_ga.prep(baseline='radial2'). To stay fast we sweep ALL
qc_ok eyes with the DEFAULT BM (ov.bm), not the DL model. (The striping is in the en-face regardless
of which BM anchors the band; a wrong BM only adds a constant column offset, which destripe handles.)

Metric definition (two parts, both order-invariant to the baseline):
  STRIPE = residual slow-axis banding amplitude that survives destripe2d, measured on the en-face core,
           as a fraction of the en-face spatial std (the destripe2d 'banding_score' idea, but measured
           on the DESTRIPED rpe_nat = the residual the detector actually sees).
  SNR    = the en-face contrast-to-noise: median(rpe6[core]) / (residual high-freq noise std). Low SNR
           = the RPE-loss signal is buried in speckle -> the threshold call is unreliable.

Run: oct_env\\Scripts\\python.exe src\\exp_striping_metric.py
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import m3_projections as mp
from paths import DATA_DIR, RESULTS_DIR
from reader.core import e2e_source, oac_ga
from reader.core import projection as proj


def plex_table():
    """subject,eye -> (plex_mm2, dl_quad area) from plex_compare.csv (the all-DL areas)."""
    out = {}
    with open(os.path.join(RESULTS_DIR, "plex_compare.csv"), newline="") as f:
        for r in csv.DictReader(f):
            out[(r["subject"], r["eye"].upper())] = (
                float(r["plex_mm2"]),
                float(r.get("dl_lin", "nan") or "nan"),
                float(r.get("dl_quad", "nan") or "nan"),
            )
    return out


def qc_ok_eyes():
    eyes = []
    with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["qc_status"] == "ok":
                eyes.append((r["subject"], r["eye"].upper(),
                             os.path.join(DATA_DIR, *r["e2e_file"].split("/"))))
    return eyes


def stripe_snr(p):
    """Striping + SNR metrics from a prep dict (uses rpe_nat = the destriped OAC native map, and
    rpe6/core = the resampled en-face + measurement core)."""
    nat = np.asarray(p["rpe_nat"], np.float32)          # destriped OAC RPE-loss, native (n, W)
    # RESIDUAL slow-axis banding that survived destripe2d: per-B-scan row-mean high-freq component.
    rowmean = nat.mean(axis=1)
    band_resid = rowmean - gaussian_filter1d(rowmean, 4)
    stripe = float(band_resid.std() / (nat.std() + 1e-6))
    # A second, en-face striping read: horizontal-band power on the en-face core (rows are slow axis
    # after to_enface row-resample) relative to spatial std -> robust to native sampling.
    rpe6, core = p["rpe6"], p["core"]
    cm = rpe6.copy(); cm[~core] = np.nan
    rowmean6 = np.nanmean(cm, axis=1)
    rowmean6 = np.where(np.isfinite(rowmean6), rowmean6, np.nanmedian(rowmean6))
    band6 = rowmean6 - gaussian_filter1d(rowmean6, 4)
    stripe6 = float(np.nanstd(band6) / (np.nanstd(rpe6[core]) + 1e-6))
    # SNR: signal level vs high-frequency noise on the core (the speckle the threshold must beat).
    sig = float(np.nanmedian(rpe6[core]))
    hi = rpe6 - gaussian_filter(rpe6, 2.0)
    noise = float(np.nanstd(hi[core]) + 1e-6)
    snr = sig / noise
    return stripe, stripe6, snr


def main():
    plex = plex_table()
    rows = []
    for subject, eye, path in qc_ok_eyes():
        if not os.path.exists(path):
            print(f"  MISSING {subject} {eye}: {path}", flush=True)
            continue
        try:
            raw = e2e_source.open_e2e(path)
            idx = e2e_source.default_volume_index(raw, eye)
            ov = e2e_source.load_volume(raw, idx)
            p = oac_ga.prep(ov, ov.bm, baseline="radial2")     # DEFAULT BM (fast, no DL)
        except Exception as e:
            print(f"  FAIL {subject} {eye}: {e}", flush=True)
            continue
        stripe, stripe6, snr = stripe_snr(p)
        pm, dl_lin, dl_quad = plex.get((subject, eye), (float("nan"),) * 3)
        err = abs(dl_lin - pm) if np.isfinite(dl_lin) and np.isfinite(pm) else float("nan")
        rows.append((subject, eye, stripe, stripe6, snr, pm, dl_lin, err))
        print(f"  {subject} {eye}  stripe={stripe:.4f} stripe6={stripe6:.4f} snr={snr:5.2f}  "
              f"PLEX={pm:6.2f} dl_lin={dl_lin:6.2f} |err|={err:6.2f}", flush=True)

    print("\n==== ranked by STRIPE (native residual banding), with |dl_lin - PLEX| ====")
    rows_s = sorted(rows, key=lambda r: -r[2])
    print("  {:22} {:3} {:>8} {:>8} {:>6} {:>7} {:>7} {:>6}".format(
        "subject", "eye", "stripe", "stripe6", "snr", "PLEX", "dl_lin", "|err|"))
    for subject, eye, stripe, stripe6, snr, pm, dl_lin, err in rows_s:
        print("  {:22} {:3} {:8.4f} {:8.4f} {:6.2f} {:7.2f} {:7.2f} {:6.2f}".format(
            subject, eye, stripe, stripe6, snr, pm, dl_lin, err))

    # correlation: does striping / low SNR predict the large agreement errors?
    fin = [r for r in rows if np.isfinite(r[7])]
    if len(fin) >= 4:
        st = np.array([r[2] for r in fin]); st6 = np.array([r[3] for r in fin])
        sn = np.array([r[4] for r in fin]); er = np.array([r[7] for r in fin])
        def rs(a, b):
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])
        print(f"\n  Spearman(stripe , |err|) = {rs(st, er):+.3f}")
        print(f"  Spearman(stripe6, |err|) = {rs(st6, er):+.3f}")
        print(f"  Spearman(snr    , |err|) = {rs(sn, er):+.3f}  (expect NEGATIVE: low SNR -> big err)")
        print(f"  Pearson (stripe , |err|) = {float(np.corrcoef(st, er)[0,1]):+.3f}")


if __name__ == "__main__":
    main()
