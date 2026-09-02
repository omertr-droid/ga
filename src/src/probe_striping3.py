#!/usr/bin/env python
"""ITEM A probe v3 (STANDALONE; does NOT edit oac_ga / footprint): the STRONGER metric, FAST.

v1 (probe_striping.py) proved the per-B-scan ROW-LEVEL striping metric does NOT discriminate -- destripe2d
removes that component (stripe_post ~0.01 everywhere; Spearman(post,|err|)=0.003). The striping that
CORRUPTS the area is the part destripe2d cannot touch: a stripe whose amplitude VARIES across A-scans
(within-row spatial texture) survives in rpe_nat as 2D SLOW-AXIS-ANISOTROPIC texture. So we measure the
DIRECTIONAL ANISOTROPY of the destriped native OAC RPE-loss map: residual HIGH-FREQUENCY gradient energy
ALONG the slow axis (row/B-scan-to-B-scan) vs the fast axis (column-to-column). Real anatomy is roughly
isotropic; residual striping injects slow-axis-direction texture -> anisotropy > 1.

Uses the SAME cheap-BM proxy as v1 (deepest bright reflector + smooth) -- striping is an ACQUISITION
artifact in the OAC en-face, ~independent of the exact BM row (a few-px BM shift doesn't move the
B-scan-to-B-scan level jitter; v1 already validated cheap-BM striping numbers). -> ~1s/eye, no DL, all
29 qc_ok eyes. Output -> results/striping_quality3.csv + Spearman vs |dl_quad - PLEX|.

Run: oct_env\\Scripts\\python.exe src\\probe_striping3.py
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
OUT = os.path.join(RESULTS_DIR, "striping_quality3.csv")
AX = mp.AX


def cheap_bm(vol):
    """Per-A-scan BM proxy (n,W): deepest bright row in the lower 2/3, smoothed (same as probe v1)."""
    v = np.asarray(vol, np.float32)
    n, H, W = v.shape
    lo = H // 3
    sub = v[:, lo:, :]
    depth_w = np.linspace(1.0, 1.6, sub.shape[1])[None, :, None]
    bm = lo + np.argmax(sub * depth_w, axis=1).astype(np.float32)
    bm = gaussian_filter(bm, (1.0, 3.0))
    return np.clip(bm, lo, H - 2)


def band_mean_vec(vol, bm, lo_um, hi_um):
    n, H, W = vol.shape
    lo = np.clip(np.round(bm + lo_um / AX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bm + hi_um / AX), 1, H).astype(int)
    cs = np.zeros((n, H + 1, W), np.float32)
    np.cumsum(vol.astype(np.float32), axis=1, out=cs[:, 1:, :])
    ii = np.arange(n)[:, None]; xx = np.arange(W)[None, :]
    s = cs[ii, hi, xx] - cs[ii, lo, xx]
    return (s / np.maximum(hi - lo, 1)).astype(np.float32)


def metrics(rpe_raw, rpe_nat):
    rm_all = rpe_nat.mean(axis=1)
    thr = 0.05 * float(np.nanmax(rm_all)) if np.isfinite(np.nanmax(rm_all)) else -np.inf
    vr = rm_all > thr
    core = rpe_nat[vr] if vr.sum() >= 8 else rpe_nat                      # (nv, W) in-field

    # 1. row-level striping (the destripe target) pre & post -- for the record (v1 result: ~0 post)
    def rowlevel_hf(nat):
        rm = nat[vr].mean(axis=1) if vr.sum() >= 8 else nat.mean(axis=1)
        hf = rm - gaussian_filter1d(rm, 2.0, mode="nearest")
        return float(hf.var() / (rm.var() + 1e-12))
    rl_pre, rl_post = rowlevel_hf(rpe_raw), rowlevel_hf(rpe_nat)

    # 2. DIRECTIONAL ANISOTROPY of the destriped map -- the residual 2D striping the area inherits.
    #    Remove the smooth anatomical falloff first (slow-axis gaussian) so a genuinely smooth steep
    #    gradient doesn't masquerade as striping; then compare slow- vs fast-axis HF gradient energy.
    hf_map = core - gaussian_filter1d(core, 2.0, axis=0, mode="nearest")  # kill low-freq lesion/falloff
    eh_slow = float(np.mean(np.diff(hf_map, axis=0) ** 2))                # row-to-row (slow / B-scan)
    eh_fast = float(np.mean(np.diff(hf_map, axis=1) ** 2))               # col-to-col (fast / A-scan)
    anis_hf = eh_slow / (eh_fast + 1e-12)

    # raw (no low-freq removal) anisotropy, for comparison
    anis = float(np.mean(np.diff(core, axis=0) ** 2) / (np.mean(np.diff(core, axis=1) ** 2) + 1e-12))

    # 3. ABSOLUTE residual slow-axis HF energy normalized by the map's robust dynamic range. Unlike the
    #    RATIO (anis_hf), this is a magnitude: how MUCH slow-axis texture remains relative to the signal.
    dr = float(np.percentile(core, 95) - np.percentile(core, 5)) + 1e-9
    stripe_pwr = float(np.sqrt(eh_slow) / dr)

    # 4. low-SNR proxy: within-row (fast-axis) HF noise / dynamic range. NOTE on a striped eye the stripes
    #    inflate dr, so a high cnr is NOT automatically "clean" -- report both, interpret jointly.
    col_hf = core - gaussian_filter1d(core, 2.0, axis=1, mode="nearest")
    noise = 1.4826 * float(np.median(np.abs(col_hf))) + 1e-9
    cnr = dr / noise

    return dict(rl_pre=rl_pre, rl_post=rl_post, anis=anis, anis_hf=anis_hf,
                stripe_pwr=stripe_pwr, cnr=cnr, dyn_range=dr, n_rows=int(vr.sum()))


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
    return float((rx * ry).sum() / (np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-12))


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
            print(f"[{subj} {eye}] no E2E", flush=True); continue
        try:
            if e2e not in cache:
                cache.clear(); cache[e2e] = e2e_source.open_e2e(e2e)
            raw = cache[e2e]
            idx = e2e_source.default_volume_index(raw, eye)
            v = raw.vols[idx]
            vol = np.asarray(v.volume, np.float32)
            bm = cheap_bm(vol)
            oac = mp.oac_volume(vol)
            rpe_raw = band_mean_vec(oac, bm, *mp.OAC_RPE_UM)
            rpe_nat = mp.destripe2d(rpe_raw, signed=False)
            mt = metrics(rpe_raw, rpe_nat)
            plex, dl = pm.get((subj, eye), (float("nan"), float("nan")))
            mt.update(subject=subj, eye=eye, plex=plex, dl_quad=dl,
                      abs_err=abs(dl - plex) if np.isfinite(dl) and np.isfinite(plex) else float("nan"))
            out.append(mt)
            print(f"[{subj} {eye:>2}] anis_hf={mt['anis_hf']:5.2f} anis={mt['anis']:5.2f} "
                  f"stripe_pwr={mt['stripe_pwr']:.3f} cnr={mt['cnr']:5.1f}  |err|={mt['abs_err']:.2f}", flush=True)
        except Exception as e:
            print(f"[{subj} {eye}] FAIL {type(e).__name__}: {e}", flush=True)

    cols = ["subject", "eye", "n_rows", "rl_pre", "rl_post", "anis", "anis_hf", "stripe_pwr",
            "cnr", "dyn_range", "plex", "dl_quad", "abs_err"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"\nwrote {OUT}  ({len(out)} eyes)", flush=True)

    pr = [o for o in out if np.isfinite(o["abs_err"])]
    if len(pr) >= 5:
        er = [o["abs_err"] for o in pr]
        print("\n=== Spearman(metric, |dl_quad - PLEX|) over ALL eyes ===", flush=True)
        for name in ["anis_hf", "anis", "stripe_pwr", "rl_post", "rl_pre", "cnr", "dyn_range"]:
            print(f"  {name:11s}: rho = {spearman([o[name] for o in pr], er):+.3f}", flush=True)

        # Striping is BIDIRECTIONAL (over- AND under-call) and several big errors are NON-striping
        # (008 OD margin trim, 009 OS drusenoid, model-eye). So also test: does the metric FLAG the
        # assessment's named striping eyes as distribution outliers? + does it predict |err| AMONG the
        # eyes the assessment attributes to striping?
        named = {("NHAMD-003-015-V3", "OS"), ("NHAMD-003-014-V1", "OD"), ("NHAMD-003-001-V1", "OD"),
                 ("NHAMD-003-010-V1", "OD"), ("NHAMD-003-011-V3", "OD"), ("NHAMD-003-011-V3", "OS"),
                 ("NHAMD-003-003-V3", "OD"), ("NHAMD-003-016-V2", "OD")}
        print("\n=== worst anis_hf (residual 2D striping), worst first ===", flush=True)
        for o in sorted(pr, key=lambda d: -d["anis_hf"]):
            tag = " <-NAMED" if (o["subject"], o["eye"]) in named else ""
            print(f"  {o['subject']} {o['eye']:>2}  anis_hf={o['anis_hf']:5.2f} stripe_pwr={o['stripe_pwr']:.3f} "
                  f"cnr={o['cnr']:5.1f}  plex={o['plex']:6.2f} ours={o['dl_quad']:6.2f} |err|={o['abs_err']:5.2f}{tag}", flush=True)
        print("\n=== worst |err|, worst first ===", flush=True)
        for o in sorted(pr, key=lambda d: -d["abs_err"]):
            tag = " <-NAMED" if (o["subject"], o["eye"]) in named else ""
            print(f"  {o['subject']} {o['eye']:>2}  |err|={o['abs_err']:5.2f}  anis_hf={o['anis_hf']:5.2f} "
                  f"stripe_pwr={o['stripe_pwr']:.3f} cnr={o['cnr']:5.1f}{tag}", flush=True)


if __name__ == "__main__":
    main()
