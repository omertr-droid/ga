#!/usr/bin/env python
"""ITEM A probe (STANDALONE, does NOT edit oac_ga / footprint): characterize slow-axis striping +
low-SNR in the OAC RPE-loss en-face, per eye, over ALL qc_ok cohort eyes.

The OAC RPE-loss en-face has B-scan-to-B-scan (SLOW-AXIS) striping. `m3_projections.destripe2d`
removes the per-B-scan LEVEL shift (additive per-row median of the slow-axis high-freq, then a
multiplicative per-row gain). The LESION is low-frequency along the slow axis (spans many B-scans);
striping is high-frequency (row-to-row jitter). So a striping-severity metric = the slow-axis
HIGH-FREQUENCY energy of the per-B-scan row-mean profile, normalized by the en-face's own spread.

We do NOT need DL GA detection for this -- striping is a property of the acquisition + en-face, not of
the GA call -- so we use the BM that load_volume already provides (device where present, self-seg else),
keeping the probe cheap (no per-eye DL). We reproduce the OAC RPE-loss native en-face up through
`rpe_raw` exactly as oac_ga.prep does, so we can measure striping PRE-destripe (rpe_raw) and what
SURVIVES destripe (rpe_nat = destripe2d(rpe_raw)).

Metrics per eye (all on the NATIVE (n,W) en-face; n = #B-scans = slow axis):
  stripe_pre   slow-axis HF energy fraction of the row-mean, BEFORE destripe (how striped the raw is)
  stripe_post  same, AFTER destripe -> RESIDUAL striping that the detector actually sees   << the flag
  stripe_drop  stripe_pre - stripe_post (how much destripe removed; small drop + high post = stuck)
  snr          median row-mean / robust slow-axis row-mean noise  (low = low-SNR / vignette-heavy)
  rowcv        coeff-of-variation of the residual row-mean (an alt amplitude view, mm-free)

Run: oct_env\\Scripts\\python.exe src\\probe_striping.py
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
from reader.core import e2e_source

# --- cheap BM proxy -------------------------------------------------------------------------------
# The striping metric needs only an APPROXIMATE BM: striping is a per-B-scan ROW-LEVEL fluctuation that
# survives in the row-mean of the OAC band regardless of exact band depth. Self-seg BM (graph search) is
# ~60s/eye -> too slow over 29 eyes. Instead derive BM per A-scan as the DEEPEST bright row (the
# RPE/BM complex is the deepest strong reflector) and lightly smooth it. Validated below: on the
# device-BM eyes the cheap-BM metric matches the real-BM run (001-005) to ~0.01.


def cheap_bm(vol):
    """Per-A-scan BM proxy (n,W): deepest bright row in the lower half, smoothed across the field."""
    v = np.asarray(vol, np.float32)
    n, H, W = v.shape
    lo = H // 3                                   # search the lower 2/3 (RPE/BM complex is deep)
    sub = v[:, lo:, :]
    # weight deeper rows so we latch the DEEPEST strong reflector (BM), not the inner retina
    depth_w = np.linspace(1.0, 1.6, sub.shape[1])[None, :, None]
    bm = lo + np.argmax(sub * depth_w, axis=1).astype(np.float32)     # (n,W)
    bm = gaussian_filter(bm, (1.0, 3.0))          # denoise within + across B-scans
    return np.clip(bm, lo, H - 2)

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
PLEX = os.path.join(RESULTS_DIR, "plex_compare.csv")


AX = mp.AX                                                    # axial um/px


def band_mean_vec(vol, bm, lo_um, hi_um):
    """Vectorized per-A-scan MEAN of `vol` over the BM-relative band [lo_um,hi_um] um. Identical to
    mp.band(...,'mean') but via a depth cumsum (no Python triple loop) -> ~1s/eye instead of ~40s."""
    n, H, W = vol.shape
    lo = np.clip(np.round(bm + lo_um / AX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bm + hi_um / AX), 1, H).astype(int)
    cs = np.zeros((n, H + 1, W), np.float32)
    np.cumsum(vol.astype(np.float32), axis=1, out=cs[:, 1:, :])    # cs[:,k,:] = sum of rows [0:k)
    ii = np.arange(n)[:, None]; xx = np.arange(W)[None, :]
    s = cs[ii, hi, xx] - cs[ii, lo, xx]
    cnt = np.maximum(hi - lo, 1)
    return (s / cnt).astype(np.float32)


def rpe_native(ov, bm):
    """Reproduce oac_ga.prep's OAC RPE-loss native en-face up through rpe_raw + rpe_nat (destriped)."""
    oac = mp.oac_volume(ov.vol)
    rpe_raw = band_mean_vec(oac, bm, *mp.OAC_RPE_UM)             # (n,W) RPE-loss, low = GA
    rpe_nat = mp.destripe2d(rpe_raw, signed=False)
    return rpe_raw, rpe_nat


def slow_hf_fraction(nat, valid_rows, slow_sigma=2.0):
    """Slow-axis high-frequency energy fraction of the per-B-scan row-mean profile.

    rowmean[i] = mean over A-scans of en-face row i (one number per B-scan). Striping is the part of
    rowmean that the slow-axis smoother (same sigma destripe2d uses) does NOT capture -- i.e. the
    high-frequency, row-to-row jitter. The lesion lives in the low-frequency (smoothed) part.
      frac = var(rowmean - smooth(rowmean)) / var(rowmean)
    Restricted to in-field rows (rows whose en-face mean is non-trivially above the off-field pad), so
    empty padding rows don't dilute the metric. Returns frac in [0,1] (higher = more residual striping)."""
    rm = nat[valid_rows].mean(axis=1) if valid_rows.any() else nat.mean(axis=1)
    lo = gaussian_filter1d(rm, slow_sigma, mode="nearest")
    hf = rm - lo
    return float(hf.var() / (rm.var() + 1e-12))


def metrics(rpe_raw, rpe_nat):
    # in-field rows: a B-scan whose row-mean is above 5% of the field's max row-mean (drops pure pad)
    rm_all = rpe_nat.mean(axis=1)
    thr = 0.05 * float(np.nanmax(rm_all)) if np.isfinite(np.nanmax(rm_all)) else -np.inf
    valid_rows = rm_all > thr
    pre = slow_hf_fraction(rpe_raw, valid_rows)
    post = slow_hf_fraction(rpe_nat, valid_rows)
    # SNR: median in-field row level / robust slow-axis noise (MAD of the HF residual of the row-mean)
    rm = rpe_nat[valid_rows].mean(axis=1) if valid_rows.any() else rm_all
    lo = gaussian_filter1d(rm, 2.0, mode="nearest")
    noise = 1.4826 * float(np.median(np.abs(rm - lo))) + 1e-9
    snr = float(np.median(rm) / noise)
    rowcv = float((rm - lo).std() / (abs(np.median(rm)) + 1e-9))
    return dict(stripe_pre=pre, stripe_post=post, stripe_drop=pre - post, snr=snr, rowcv=rowcv,
                n_rows=int(valid_rows.sum()))


def qc_ok_eyes():
    out = []
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if r["qc_status"] == "ok":
                out.append((r["subject"], r["eye"], os.path.join(DATA_DIR, *r["e2e_file"].split("/"))))
    return out


def plex_map():
    m = {}
    if os.path.exists(PLEX):
        with open(PLEX, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    m[(r["subject"], r["eye"])] = float(r["plex_mm2"])
                    m[(r["subject"], r["eye"], "A_quad")] = float(r.get("A_quad") or "nan")
                except (TypeError, ValueError):
                    pass
    return m


def main():
    pm = plex_map()
    rows = []
    for subject, eye, e2e in qc_ok_eyes():
        if not os.path.exists(e2e):
            print(f"[{subject} {eye}] no E2E", flush=True)
            continue
        try:
            raw = e2e_source.open_e2e(e2e)
            idx = e2e_source.default_volume_index(raw, eye)
            v = raw.vols[idx]
            vol = np.asarray(v.volume, float)
            # cheap BM proxy -> NO slow self-seg graph search; ref = raw.refs[idx].fov_mm
            bm = cheap_bm(vol)

            class _OV:                                          # minimal stand-in for the OAC band call
                pass
            ov = _OV(); ov.vol = vol
            rpe_raw, rpe_nat = rpe_native(ov, bm)
            mt = metrics(rpe_raw, rpe_nat)
            mt.update(subject=subject, eye=eye, plex=pm.get((subject, eye), float("nan")),
                      ours=pm.get((subject, eye, "A_quad"), float("nan")), bm_src="cheap")
            rows.append(mt)
            print(f"[{subject} {eye}] pre={mt['stripe_pre']:.3f} post={mt['stripe_post']:.3f} "
                  f"drop={mt['stripe_drop']:.3f} snr={mt['snr']:.1f}", flush=True)
        except Exception as e:
            print(f"[{subject} {eye}] FAIL {type(e).__name__}: {e}", flush=True)

    if not rows:
        return
    out = os.path.join(RESULTS_DIR, "striping_metric.csv")
    cols = ["subject", "eye", "bm_src", "n_rows", "stripe_pre", "stripe_post", "stripe_drop",
            "snr", "rowcv", "plex", "ours"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # report: worst residual striping + |ours-plex| coincidence
    print("\n=== RESIDUAL STRIPING (stripe_post), worst first ===")
    for r in sorted(rows, key=lambda d: -d["stripe_post"]):
        err = abs(r["ours"] - r["plex"]) if np.isfinite(r["ours"]) and np.isfinite(r["plex"]) else float("nan")
        print(f"  {r['subject']} {r['eye']:>2}  post={r['stripe_post']:.3f}  pre={r['stripe_pre']:.3f}  "
              f"snr={r['snr']:5.1f}  plex={r['plex']:6.2f}  ours={r['ours']:6.2f}  |err|={err:5.2f}")

    # does the metric PREDICT the worst-agreement eyes? rank-correlate stripe_post vs |ours-plex|
    paired = [(r["stripe_post"], abs(r["ours"] - r["plex"]))
              for r in rows if np.isfinite(r["ours"]) and np.isfinite(r["plex"])]
    if len(paired) >= 4:
        a = np.array([p[0] for p in paired]); b = np.array([p[1] for p in paired])
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
        print(f"\nSpearman(stripe_post, |ours-plex|) = {float((ra*rb).mean()):.3f}  (n={len(paired)})")
        # also pre and snr
        for name, vals in [("stripe_pre", [r["stripe_pre"] for r in rows if np.isfinite(r["ours"]) and np.isfinite(r["plex"])]),
                           ("snr", [r["snr"] for r in rows if np.isfinite(r["ours"]) and np.isfinite(r["plex"])])]:
            v = np.array(vals); rv = v.argsort().argsort().astype(float)
            rv = (rv - rv.mean()) / (rv.std() + 1e-9)
            print(f"Spearman({name:12s}, |ours-plex|) = {float((rv*rb).mean()):.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
