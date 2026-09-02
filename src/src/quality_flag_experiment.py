#!/usr/bin/env python
"""ITEM A BUILD (STANDALONE; does NOT edit reader/core/oac_ga.py or footprint.py): the slow-axis
striping / low-SNR QUALITY FLAG, formalized from the PHASE-1 diagnosis (src/probe_striping3.py).

The flag is a NON-GATING per-eye CONFIDENCE label, never an exclusion. It is computed from the destriped
native OAC RPE-loss map (rpe_nat = mp.destripe2d(rpe_raw, signed=False)) -- the same array oac_ga.prep
builds at line 174 -- as the residual slow-axis striping energy that SURVIVED destripe2d:

  stripe_pwr = sqrt( mean[ (slow-axis diff of HF-residual)^2 ] ) / dynamic_range
  anis_hf    = mean[(slow-axis diff)^2] / mean[(fast-axis diff)^2]        (the residual 2D stripe ratio)

where HF-residual = rpe_nat - gaussian_filter1d(rpe_nat, 2.0, axis=0) removes the smooth lesion/falloff so
a genuinely smooth steep anatomical gradient is NOT mistaken for striping.

quality_flag(rpe_nat) reproduces EXACTLY the logic of the recommended oac_ga.prep insertion (see the
result diff). It is BM-row-insensitive (striping is an acquisition artifact; a few-px BM shift does not
move B-scan-to-B-scan level jitter -- proven in probe v1/v3), so this script computes it over EVERY
qc_ok eye via the CHEAP-BM path (~1s/eye, no DL detection needed). DL is NOT required for the metric.

Then it tests the PAYOFF: Spearman(metric, |ours - PLEX|) using the all-DL quad column (dl_quad) from
results/plex_compare.csv, and reports which eyes the flag fires on.

Output -> results/quality_flag.csv (subject, eye, stripe_pwr, anis_hf, low_confidence flag, plex,
dl_quad, abs_err).

Run: oct_env\\Scripts\\python.exe src\\quality_flag_experiment.py
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # noqa: E402

import m3_projections as mp  # noqa: E402
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
PLEX = os.path.join(RESULTS_DIR, "plex_compare.csv")
OUT = os.path.join(RESULTS_DIR, "quality_flag.csv")
AX = mp.AX

# Calibrated threshold from the diagnosis: stripe_pwr distribution over 29 qc_ok eyes is median 0.328,
# p90 0.371, MAX 0.452 (015 OS, a z=2.64 outlier). 0.40 sits BETWEEN 015 OS (0.452) and the next eye
# (026 OS 0.396) -> flags ONLY the one genuinely striping-corrupted eye, leaving every validated/good eye
# clean (005 OD 0.299, 005 OS 0.327, 008 OD 0.285, 008 OS 0.280, 015 OD 0.309).
STRIPE_PWR_FLAG = 0.40


def quality_flag(rpe_nat, stripe_pwr_thr=STRIPE_PWR_FLAG):
    """NON-GATING slow-axis striping / low-SNR confidence flag from the destriped native OAC RPE-loss map.

    This is the EXACT logic of the recommended oac_ga.prep insertion (see the result diff): in-field rows
    only, kill the low-frequency lesion/falloff with a slow-axis gaussian, then measure residual slow-axis
    HF energy (absolute, normalized by dynamic range = stripe_pwr) and its slow/fast ratio (anis_hf).

    Returns {stripe_pwr, anis_hf, low_confidence}. low_confidence True == 'striping -- area unreliable,
    low confidence' badge. NEVER drop the eye on it."""
    rm = rpe_nat.mean(axis=1)
    mx = float(np.nanmax(rm))
    vr = rm > 0.05 * mx if np.isfinite(mx) else np.ones(len(rm), bool)
    core = rpe_nat[vr] if vr.sum() >= 8 else rpe_nat                       # in-field rows only
    hf = core - gaussian_filter1d(core, 2.0, axis=0, mode="nearest")       # kill low-freq lesion/falloff
    dr = float(np.percentile(core, 95) - np.percentile(core, 5)) + 1e-9    # robust dynamic range
    eh_slow = float(np.mean(np.diff(hf, axis=0) ** 2))                     # row-to-row (slow / B-scan)
    eh_fast = float(np.mean(np.diff(hf, axis=1) ** 2))                     # col-to-col (fast / A-scan)
    stripe_pwr = float(np.sqrt(eh_slow) / dr)                             # abs residual slow-axis HF
    anis_hf = float(eh_slow / (eh_fast + 1e-12))                          # residual 2D stripe ratio
    return {"stripe_pwr": stripe_pwr, "anis_hf": anis_hf,
            "low_confidence": bool(stripe_pwr >= stripe_pwr_thr)}


# ---------------------------------------------------------------------------------------------------
# Cheap path: build rpe_nat exactly as oac_ga.prep does (oac_volume -> band over OAC_RPE_UM -> destripe2d)
# but with a cheap-BM proxy (the metric is BM-row-insensitive; see module docstring).
# ---------------------------------------------------------------------------------------------------
def cheap_bm(vol):
    """Per-A-scan BM proxy (n,W): deepest bright row in the lower 2/3, smoothed."""
    v = np.asarray(vol, np.float32)
    n, H, W = v.shape
    lo = H // 3
    sub = v[:, lo:, :]
    depth_w = np.linspace(1.0, 1.6, sub.shape[1])[None, :, None]
    bm = lo + np.argmax(sub * depth_w, axis=1).astype(np.float32)
    bm = gaussian_filter(bm, (1.0, 3.0))
    return np.clip(bm, lo, H - 2)


def band_mean_vec(vol, bm, lo_um, hi_um):
    """mp.band(...,'mean') over the BM-relative band, vectorized (matches mp.band's reducer='mean')."""
    n, H, W = vol.shape
    lo = np.clip(np.round(bm + lo_um / AX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bm + hi_um / AX), 1, H).astype(int)
    cs = np.zeros((n, H + 1, W), np.float32)
    np.cumsum(vol.astype(np.float32), axis=1, out=cs[:, 1:, :])
    ii = np.arange(n)[:, None]; xx = np.arange(W)[None, :]
    s = cs[ii, hi, xx] - cs[ii, lo, xx]
    return (s / np.maximum(hi - lo, 1)).astype(np.float32)


def rpe_nat_cheap(vol):
    """rpe_nat as oac_ga.prep builds it (line 169-174), cheap-BM: oac_volume -> band -> destripe2d."""
    oac = mp.oac_volume(vol)
    rpe_raw = band_mean_vec(oac, cheap_bm(vol), *mp.OAC_RPE_UM)
    return mp.destripe2d(rpe_raw, signed=False)


def load_plex():
    m = {}
    if os.path.exists(PLEX):
        with open(PLEX, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    m[(r["subject"], r["eye"].upper())] = (float(r["plex_mm2"]), float(r["dl_quad"]))
                except (KeyError, ValueError):
                    pass
    return m


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))


def main():
    pm = load_plex()
    with open(PAIRING, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["qc_status"] == "ok"]
    cache = {}
    out = []
    for r in rows:
        subj, eye = r["subject"], r["eye"].upper()
        e2e = os.path.join(DATA_DIR, *r["e2e_file"].split("/"))
        if not os.path.exists(e2e):
            print(f"[{subj} {eye}] no E2E -> skip", flush=True); continue
        try:
            if e2e not in cache:
                cache.clear(); cache[e2e] = e2e_source.open_e2e(e2e)
            raw = cache[e2e]
            idx = e2e_source.default_volume_index(raw, eye)
            vol = np.asarray(raw.vols[idx].volume, np.float32)
            qf = quality_flag(rpe_nat_cheap(vol))
            plex, dl = pm.get((subj, eye), (float("nan"), float("nan")))
            qf.update(subject=subj, eye=eye, plex=plex, dl_quad=dl,
                      abs_err=abs(dl - plex) if np.isfinite(dl) and np.isfinite(plex) else float("nan"))
            out.append(qf)
            print(f"[{subj} {eye:>2}] stripe_pwr={qf['stripe_pwr']:.3f} anis_hf={qf['anis_hf']:5.2f} "
                  f"flag={'LOW-CONF' if qf['low_confidence'] else 'ok':>8}  |err|={qf['abs_err']:.2f}",
                  flush=True)
        except Exception as e:
            print(f"[{subj} {eye}] FAIL {type(e).__name__}: {e}", flush=True)

    # --- write results/quality_flag.csv ---
    cols = ["subject", "eye", "stripe_pwr", "anis_hf", "low_confidence", "plex", "dl_quad", "abs_err"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for o in out:
            w.writerow({"subject": o["subject"], "eye": o["eye"],
                        "stripe_pwr": round(o["stripe_pwr"], 4), "anis_hf": round(o["anis_hf"], 4),
                        "low_confidence": int(o["low_confidence"]),
                        "plex": round(o["plex"], 4) if np.isfinite(o["plex"]) else "",
                        "dl_quad": round(o["dl_quad"], 4) if np.isfinite(o["dl_quad"]) else "",
                        "abs_err": round(o["abs_err"], 4) if np.isfinite(o["abs_err"]) else ""})
    print(f"\nwrote {OUT}  ({len(out)} eyes)", flush=True)

    # --- PAYOFF test: does the metric predict the per-eye |ours - PLEX| error? ---
    pr = [o for o in out if np.isfinite(o["abs_err"])]
    if len(pr) >= 5:
        er = [o["abs_err"] for o in pr]
        print(f"\n=== PAYOFF: Spearman(metric, |dl_quad - PLEX|), n={len(pr)} ===", flush=True)
        print(f"  stripe_pwr : rho = {spearman([o['stripe_pwr'] for o in pr], er):+.3f}", flush=True)
        print(f"  anis_hf    : rho = {spearman([o['anis_hf'] for o in pr], er):+.3f}", flush=True)

        flagged = [o for o in pr if o["low_confidence"]]
        print(f"\n=== FLAGGED (low_confidence, stripe_pwr >= {STRIPE_PWR_FLAG}): {len(flagged)} eye(s) ===",
              flush=True)
        for o in sorted(flagged, key=lambda d: -d["stripe_pwr"]):
            print(f"  {o['subject']} {o['eye']:>2}  stripe_pwr={o['stripe_pwr']:.3f}  "
                  f"plex={o['plex']:6.2f} ours={o['dl_quad']:6.2f} |err|={o['abs_err']:5.2f}", flush=True)

        print("\n=== worst |err| (worst first) vs their flag ===", flush=True)
        for o in sorted(pr, key=lambda d: -d["abs_err"])[:10]:
            tag = "  <-FLAGGED" if o["low_confidence"] else ""
            print(f"  {o['subject']} {o['eye']:>2}  |err|={o['abs_err']:5.2f}  "
                  f"stripe_pwr={o['stripe_pwr']:.3f}{tag}", flush=True)


if __name__ == "__main__":
    main()
