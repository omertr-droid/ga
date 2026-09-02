#!/usr/bin/env python
"""EMPIRICAL TEST: a foveal-centered monotone radial-profile + low-rank angular-residual healthy-RPE
baseline (PCR) vs the live LINEAR and QUADRATIC global-polynomial baselines in reader.core.oac_ga.

Reuses the production detector unchanged (oac_ga.prep gives rpe6/core/g_base; we swap ONLY the `base`
array, then oac_ga.footprint computes the cRORA area). BM = the DL model for every eye (no annotation,
fast, BM-consistent), matching src/sweep_oac.py and the dl_quad/dl_lin columns of results/plex_compare.csv.

PCR architecture (the proposal under test):
  1. fovea center = centroid of the top-20% brightest core pixels within the central 3mm, CLAMPED to
     +-1.0mm of the field center (geometric-center fallback if too few pixels).
  2. robust (p75) radial profile of rpe6 over `core` in 0.25mm annuli around that center.
  3. PAVA isotonic-DECREASING fit of that profile for r>=0.5mm, with a flat foveal plateau r<0.5mm
     (STRUCTURALLY cannot bow up at the corners -> the documented 008 over-clamp can't happen).
  4. + a 5-term {1, r*cos, r*sin, r*cos2, r*sin2} IRLS angular residual (k=1.5, n_iter=8) to recover
     the temporal/nasal asymmetry the radial mean smears away.
  5. lift the surface to g_base (the p95 healthy level), then a SOFT cap min(base, 1.10*g_base) only
     where the angular residual lifted it above 1.10*g_base.

Run (repo root):
  oct_env\\Scripts\\python.exe src\\baseline_experiment.py
Output -> results/baseline_experiment.csv (per-eye) + a printed aggregate table.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OCT_BM_DL", "1")          # use the DL BM model for every eye

import numpy as np  # noqa: E402

import bm_dl  # noqa: E402
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402
from reader.core import e2e_source, oac_ga  # noqa: E402
from reader.core import projection as proj  # noqa: E402

PAIRING = os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "baseline_experiment.csv")
MMPP = proj.ENFACE_MMPP                          # mm/px, isotropic (6/512)

CALL_THR = 0.25         # we "call no significant GA" below this mm^2 (cRORA ~250 um floor)
CONTROL_THR = 0.05      # PLEX < this mm^2 = a no-GA control eye

# The representative subset spanning the tradeoff (gold/validated + baseline-sensitive eyes).
TEST_EYES = [
    ("NHAMD-003-005", "V3", "OD"),   # focal gold, Dice eye
    ("NHAMD-003-005", "V3", "OS"),   # small focal, LINEAR over-calls (0.57 -> lin 1.78)
    ("NHAMD-003-008", "V1", "OD"),   # LARGE eccentric, QUAD under-calls (13.78 -> quad 7.63)
    ("NHAMD-003-008", "V1", "OS"),   # large confluent
    ("NHAMD-003-015", "V3", "OD"),   # validated small
    ("NHAMD-003-017", "V3", "OD"),   # validated, near-control, both over-call
    ("NHAMD-003-004", "V1", "OD"),   # large, linear helps
    ("NHAMD-003-003", "V3", "OD"),   # mid, both over-call
    ("NHAMD-003-003", "V3", "OS"),   # mid eccentric
    ("NHAMD-003-011", "V3", "OS"),   # mid, linear over-calls
    ("NHAMD-003-014", "V1", "OS"),   # near-control, both over-call
    ("NHAMD-003-016", "V2", "OD"),   # CONTROL, both FP
    ("NHAMD-003-010", "V1", "OD"),   # small/mid
    ("NHAMD-003-001", "V1", "OD"),   # both under-call
]


def pava_decreasing(y, w=None):
    """Pool-adjacent-violators isotonic regression, NON-INCREASING (pure numpy, no sklearn).
    Equivalent to running the standard increasing PAVA on the reversed sequence."""
    y = np.asarray(y, np.float64)[::-1]                      # reverse -> solve increasing
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, np.float64)[::-1]
    val = y.copy()
    wt = w.copy()
    idx = list(range(n))                                    # block boundaries: each block = [start..]
    # standard PAVA over blocks
    vals, wts, cnts = [], [], []
    for i in range(n):
        v, ww, cc = val[i], wt[i], 1
        while vals and vals[-1] > v:                         # increasing violation -> pool
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww = pw + ww
            cc = pc + cc
        vals.append(v); wts.append(ww); cnts.append(cc)
    out = np.empty(n)
    j = 0
    for v, cc in zip(vals, cnts):
        out[j:j + cc] = v
        j += cc
    return out[::-1].astype(np.float32)                     # reverse back -> non-increasing


# ---------------------------------------------------------------------------- the PCR baseline
def pcr_baseline(rpe6, core, g_base, soft_cap=1.10, n_ang_iter=8, k=1.5):
    """Foveal-centered monotone radial profile + low-rank angular residual baseline -> (H,W).

    rpe6  : RPE-loss en-face (HIGH = healthy RPE, LOW = GA)         from oac_ga.prep
    core  : measurement support (in-field, minus rim & vignette)    from oac_ga.prep
    g_base: the p95 healthy level over core (the scale anchor)      from oac_ga.prep
    """
    H, W = rpe6.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # --- 1. fovea center: centroid of the brightest (healthiest RPE) core pixels in the central 3mm,
    #        clamped to +-1.0mm of the field center (the fovea is the brightest, most-attenuating RPE) ---
    cx0, cy0 = W / 2.0, H / 2.0
    r_geom = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2) * MMPP
    central = core & (r_geom <= 3.0)
    cx, cy = cx0, cy0
    if central.sum() >= 50:
        v = rpe6[central]
        thr = np.percentile(v, 80)                          # top-20% brightest
        sel = central & (rpe6 >= thr)
        if sel.sum() >= 20:
            cx = float(xx[sel].mean())
            cy = float(yy[sel].mean())
    clamp = 1.0 / MMPP                                       # +-1.0 mm in px
    cx = float(np.clip(cx, cx0 - clamp, cx0 + clamp))
    cy = float(np.clip(cy, cy0 - clamp, cy0 + clamp))

    r_mm = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * MMPP   # eccentricity (mm) from the fovea
    th = np.arctan2(yy - cy, xx - cx)                        # angle for the harmonics

    # --- 2. robust radial profile: p75 of rpe6 over core in 0.25mm annuli (p75 ~ "healthy" within a ring) ---
    rmax = float(r_mm[core].max()) if core.any() else 1.0
    edges = np.arange(0.0, rmax + 0.25, 0.25)
    centers, prof = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = core & (r_mm >= a) & (r_mm < b)
        if m.sum() >= 8:
            centers.append(0.5 * (a + b))
            prof.append(float(np.percentile(rpe6[m], 75)))
    if len(centers) < 3:
        return np.full((H, W), g_base, np.float32)          # degenerate -> flat (safe)
    centers = np.asarray(centers, np.float32)
    prof = np.asarray(prof, np.float32)

    # --- 3. PAVA isotonic-DECREASING for r>=0.5mm + flat foveal plateau r<0.5mm (CANNOT bow up) ---
    plateau = centers < 0.5
    if plateau.sum() >= 1:
        fov_level = float(prof[plateau].max())              # the foveal-region healthy level
    else:
        fov_level = float(prof[0])
    outer = ~plateau
    prof_mono = prof.copy()
    if outer.sum() >= 2:
        prof_mono[outer] = pava_decreasing(prof[outer])
    prof_mono[plateau] = fov_level
    # the monotone level cannot exceed the foveal plateau (enforce non-increase across the boundary)
    prof_mono = np.minimum.accumulate(prof_mono) if prof_mono[0] >= prof_mono[-1] else prof_mono
    # interpolate the monotone profile back to every pixel's eccentricity
    base_radial = np.interp(r_mm.ravel(), centers, prof_mono,
                            left=prof_mono[0], right=prof_mono[-1]).reshape(H, W).astype(np.float32)

    # --- 4. low-rank angular residual: 5-term harmonic IRLS on (rpe6 - base_radial) over core ---
    cols = [np.ones_like(r_mm),
            r_mm * np.cos(th), r_mm * np.sin(th),
            r_mm * np.cos(2 * th), r_mm * np.sin(2 * th)]
    A = np.stack([c.ravel() for c in cols], 1).astype(np.float32)
    z = (rpe6 - base_radial).ravel().astype(np.float32)
    cm = core.ravel()
    keep = cm.copy()
    ang = np.zeros_like(z)
    for _ in range(n_ang_iter):
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        ang = A @ coef
        resid = z - ang
        s = float(np.std(resid[keep])) + 1e-6
        nk = cm & (resid > -k * s)                          # drop pixels well below the trend (the lesion)
        if nk.sum() < A.shape[1] * 8:
            break
        keep = nk
    base = base_radial + ang.reshape(H, W)

    # --- 5. lift to g_base, then SOFT cap only where the angular residual pushed it above 1.10*g_base ---
    med = float(np.median(base[core])) + 1e-6
    base = base * (g_base / med)
    base = np.minimum(base, soft_cap * g_base)              # soft cap: never above 1.10*p95
    return np.maximum(base, 1e-6).astype(np.float32)


# ---------------------------------------------------------------------------- driver
def plex_lookup():
    """subject(no -V)|visit|eye -> PLEX advRPE area, from the pairing CSV."""
    out = {}
    with open(PAIRING, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("qc_status") or "").strip() != "ok":
                continue
            subj = r["subject"]
            base = subj[:-len("-" + r["visit"])] if subj.endswith("-" + r["visit"]) else subj
            key = (base, r["visit"], r["eye"].upper())
            try:
                out[key] = (float(r["advRPE_area_mm2"]),
                            os.path.join(DATA_DIR, *r["e2e_file"].split("/")))
            except (TypeError, ValueError):
                pass
    return out


def main():
    if not bm_dl.available():
        print("FATAL: DL BM model not available -- cannot run the BM-consistent experiment.", flush=True)
        print(f"  searched: {bm_dl.model_path()}", flush=True)
        return
    print(f"DL BM model: {bm_dl.model_path()}  backend={bm_dl.backend()}", flush=True)

    look = plex_lookup()
    rows = []
    raw_cache = {}
    for subj, visit, eye in TEST_EYES:
        key = (subj, visit, eye)
        if key not in look:
            print(f"  SKIP {subj} {eye}: not qc_ok / not in pairing", flush=True)
            continue
        plex, e2e_path = look[key]
        if not os.path.exists(e2e_path):
            print(f"  SKIP {subj} {eye}: E2E missing {e2e_path}", flush=True)
            continue
        if e2e_path not in raw_cache:
            raw_cache.clear()
            raw_cache[e2e_path] = e2e_source.open_e2e(e2e_path)
        raw = raw_cache[e2e_path]
        idx = e2e_source.default_volume_index(raw, eye)
        ov = e2e_source.load_volume(raw, idx)
        bm = bm_dl.segment_volume(ov.vol)

        # references: linear & quad via the production detector (DL BM) ----------------------------
        _, _, area_lin = oac_ga.detect(ov, bm, trend_order=1)
        _, _, area_quad = oac_ga.detect(ov, bm, trend_order=2)

        # NEW: PCR. prep ONCE (trend_order irrelevant -- we overwrite base), swap base, footprint -----
        p = oac_ga.prep(ov, bm, trend_order=2)
        p_new = dict(p)
        p_new["base"] = pcr_baseline(p["rpe6"], p["core"], p["g_base"])
        _, area_new = oac_ga.footprint(p_new, frac=0.50)

        rows.append({
            "subject": subj, "visit": visit, "eye": eye, "plex": plex,
            "linear": area_lin, "quad": area_quad, "pcr": area_new,
        })
        print(f"  {subj[-3:]} {eye:2}  PLEX={plex:6.2f}  lin={area_lin:6.2f}  "
              f"quad={area_quad:6.2f}  PCR={area_new:6.2f}", flush=True)

    if not rows:
        print("no eyes processed", flush=True)
        return

    # --- write per-eye CSV ---
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "visit", "eye", "plex", "linear", "quad", "pcr"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {OUT_CSV}", flush=True)

    # --- aggregate metrics ---
    plex = np.array([r["plex"] for r in rows], float)
    is_ctrl = plex < CONTROL_THR
    print(f"\n{'method':8} {'MAE':>6} {'bias':>7} {'within1':>8} {'ctrl_spec':>10}")
    agg = {}
    for name in ("linear", "quad", "pcr"):
        ours = np.array([r[name] for r in rows], float)
        d = ours - plex
        mae = float(np.abs(d).mean())
        bias = float(d.mean())
        within1 = float(np.mean(np.abs(d) <= 1.0) * 100)
        nctrl = int(is_ctrl.sum())
        spec = int(np.sum(ours[is_ctrl] < CALL_THR)) if nctrl else 0
        agg[name] = (mae, bias, within1, spec, nctrl)
        print(f"{name:8} {mae:6.3f} {bias:+7.3f} {within1:7.1f}% {spec:>5}/{nctrl}")

    # --- per-eye table (markdown) ---
    print("\n| eye | PLEX | linear | quad | PCR |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['subject'][-3:]} {r['eye']} | {r['plex']:.2f} | {r['linear']:.2f} | "
              f"{r['quad']:.2f} | {r['pcr']:.2f} |")
    return agg


if __name__ == "__main__":
    main()
