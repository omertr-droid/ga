"""BM-anchored OAC GA detector — the canonical, no-web-deps core, shared by the reader (a button) and
the CLI (`src/oac_area.py`). Validated on 005 OD (Dice 0.93 vs the in-frame gold; area 1.02 vs advRPE
1.08 mm²). Single source of truth: `src/oac_area.py` imports `prep`/`footprint`/`healthy_baseline` here.

Pipeline (all per the open volume + its EFFECTIVE/corrected BM):
  OAC volume (Vermeer) -> mean OAC in the RPE band just above BM = RPE-loss en-face (low = RPE gone = GA)
  -> GA where RPE-loss < frac * a ROBUST healthy-RPE baseline (DEFAULT `baseline='radial2'` = foveal-centered
     isotonic-DECREASING radial profile + angular residual = anatomical_baseline; `'trend'` low-order polynomial
     kept as an option) AND sub-BM hypertransmission (relative gate + an ABSOLUTE floor `hyper_abs`, default 0.10)
  -> drop the low-SNR field rim -> fill holes -> cRORA >=250 um -> area (mm²).
Defaults reflect the validated production config (2026-06): radial2 cohort MAE 0.90 vs quad 1.01; the hyper_abs
floor fixes the 002-OD control FP (specificity 5/8->6/8); 005 OD gold held (area ~1.05, Dice 0.94).
"""
import numpy as np
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes, gaussian_filter,
                           gaussian_filter1d)

import m3_projections as mp
import m3_slab
import qcviz as qv

from . import footprint as fp
from . import projection as proj
from . import render

MMPP = proj.ENFACE_MMPP
MMPP2 = proj.ENFACE_MMPP ** 2

# Sub-BM hypertransmission band (BM-relative um) for the gating hyper channel. SHALLOW (just under
# Bruch's): the transmission contrast peaks here (003 OD ~5x the old deep band) and the band stays inside
# the choroid even where it thins peripherally -- a fixed DEEP band can punch into bright sclera and fake
# hypertransmission at the field edge (workflow ga-hyper-depth, 2026-06-28; m3_slab keeps deep as default).
OAC_HYPER_UM = (20.0, 60.0)


def healthy_baseline(rpe6, core, order=2, n_iter=8, k=1.5, radial=False):
    """Smooth healthy-RPE level as a robust low-order fit over `core`, iteratively dropping pixels that fall
    well BELOW it (the GA lesion). Low-order, so it CANNOT bend to a lesion of any size -> LESION-SIZE-
    INDEPENDENT. Returns (H,W).

    Basis: a 2D tensor polynomial in (x,y) is the DEFAULT (radial=False) -- the one the validated path uses.
    `radial=True` (fit vs normalized distance from the field centre) was HYPOTHESISED to tame the order-2
    corner up-bow on big eccentric lesions, but the 29-eye sweep (src/sweep_oac.py) showed it is EMPIRICALLY
    WORSE cohort-wide (MAE 1.75 vs 1.01, r 0.79 vs 0.91); the `base_cap` clip in prep() is what actually
    controls the corner FP. Kept only as an experimental knob -- do NOT enable by default."""
    H, W = rpe6.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if radial:
        r = np.sqrt(((xx - W / 2.0) / W) ** 2 + ((yy - H / 2.0) / H) ** 2)   # normalized radius from centre
        cols = [np.ones_like(r)] + [r ** d for d in range(1, order + 1)]
    else:
        xn = xx / W - 0.5
        yn = yy / H - 0.5
        cols = [np.ones_like(xn)]
        for d in range(1, order + 1):
            for i in range(d + 1):
                cols.append((xn ** (d - i)) * (yn ** i))
    A = np.stack([c.ravel() for c in cols], 1).astype(np.float32)        # (H*W, n_terms)
    z = rpe6.ravel().astype(np.float32)
    cm = core.ravel()
    keep = cm.copy()
    fit = np.full_like(z, float(np.median(z[cm])))
    for _ in range(n_iter):
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        fit = A @ coef
        resid = z - fit
        s = float(np.std(resid[keep])) + 1e-6
        nk = cm & (resid > -k * s)                                       # drop pixels well below the trend
        if nk.sum() < A.shape[1] * 8:                                    # safety: keep the fit determined
            break
        keep = nk
    return np.maximum(fit.reshape(H, W), 1e-6)


def _pava_decreasing(y):
    """Pool-adjacent-violators isotonic regression, NON-INCREASING (pure numpy). Lets the radial healthy-
    RPE level only FALL or stay flat with eccentricity -> it can NEVER bow UP toward the field corners,
    which is the order-2 polynomial failure on large eccentric lesions (008 OD: quad 7.6 vs PLEX 13.8)."""
    y = np.asarray(y, np.float64)[::-1]                      # reverse -> solve increasing
    n = len(y)
    vals, wts, cnts = [], [], []
    for i in range(n):
        v, ww, cc = float(y[i]), 1.0, 1
        while vals and vals[-1] > v:                         # increasing violation -> pool adjacent blocks
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (pv * pw + v * ww) / (pw + ww); ww += pw; cc += pc
        vals.append(v); wts.append(ww); cnts.append(cc)
    out = np.empty(n); j = 0
    for v, cc in zip(vals, cnts):
        out[j:j + cc] = v; j += cc
    return out[::-1].astype(np.float32)                     # reverse back -> non-increasing


def anatomical_baseline(rpe6, core, g_base, soft_cap=1.10, n_ang_iter=8, k=1.5):
    """The 'radial2' healthy-RPE baseline (workflow ga-baseline-design, 2026-06-21): a foveal-centered,
    MONOTONE-non-increasing radial profile + a low-rank angular residual. It DISSOLVES the linear-vs-quad
    tradeoff: a monotone-decreasing radial level cannot bow up at the corners (quad's large-eccentric
    under-call) yet stays locally anchored (so it does not over-extrapolate like linear on small focal
    eyes). Validated on the tradeoff-spanning subset (DL BM): MAE 0.99 vs quad 1.20 vs linear 1.54,
    worst-case error 3.9 vs quad 6.2; fixes 008 OD (7.6 -> 12.9, PLEX 13.8) without 005 OS's linear
    over-call (1.78 -> 0.53, PLEX 0.57). Differs from the empirically-dead `radial` poly branch on its
    three failure axes: a DATA-DRIVEN fovea center, a MONOTONE (not polynomial) profile, and an explicit
    angular term. rpe6 HIGH = healthy RPE; `core`, `g_base` come straight from prep(). Returns (H,W)."""
    H, W = rpe6.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # 1. fovea center = centroid of the brightest (healthiest-RPE) core pixels in the central 3mm, clamped
    #    to +-1.0mm of the field center (geometric-center fallback if too few pixels).
    cx0, cy0 = W / 2.0, H / 2.0
    r_geom = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2) * MMPP
    central = core & (r_geom <= 3.0)
    cx, cy = cx0, cy0
    if central.sum() >= 50:
        sel = central & (rpe6 >= np.percentile(rpe6[central], 80))     # top-20% brightest
        if sel.sum() >= 20:
            cx, cy = float(xx[sel].mean()), float(yy[sel].mean())
    clamp = 1.0 / MMPP
    cx = float(np.clip(cx, cx0 - clamp, cx0 + clamp))
    cy = float(np.clip(cy, cy0 - clamp, cy0 + clamp))
    r_mm = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * MMPP             # eccentricity (mm) from the fovea
    th = np.arctan2(yy - cy, xx - cx)
    # 2. robust radial profile: p75 of rpe6 over core in 0.25mm annuli (p75 ~ the healthy level in a ring).
    rmax = float(r_mm[core].max()) if core.any() else 1.0
    edges = np.arange(0.0, rmax + 0.25, 0.25)
    centers, prof = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = core & (r_mm >= a) & (r_mm < b)
        if m.sum() >= 8:
            centers.append(0.5 * (a + b)); prof.append(float(np.percentile(rpe6[m], 75)))
    if len(centers) < 3:
        return np.full((H, W), g_base, np.float32)                    # degenerate -> flat (safe)
    centers = np.asarray(centers, np.float32); prof = np.asarray(prof, np.float32)
    # 3. PAVA isotonic-DECREASING for r>=0.5mm + a flat foveal plateau r<0.5mm (cannot bow up).
    plateau = centers < 0.5
    fov_level = float(prof[plateau].max()) if plateau.sum() >= 1 else float(prof[0])
    prof_mono = prof.copy()
    if (~plateau).sum() >= 2:
        prof_mono[~plateau] = _pava_decreasing(prof[~plateau])
    prof_mono[plateau] = fov_level
    prof_mono = np.minimum.accumulate(prof_mono) if prof_mono[0] >= prof_mono[-1] else prof_mono
    base_radial = np.interp(r_mm.ravel(), centers, prof_mono,
                            left=prof_mono[0], right=prof_mono[-1]).reshape(H, W).astype(np.float32)
    # 4. low-rank angular residual: 5-term {1, r*cos, r*sin, r*cos2, r*sin2} IRLS on (rpe6 - base_radial),
    #    dropping pixels well below the trend (the lesion) -- recovers the temporal/nasal asymmetry.
    cols = [np.ones_like(r_mm), r_mm * np.cos(th), r_mm * np.sin(th),
            r_mm * np.cos(2 * th), r_mm * np.sin(2 * th)]
    A = np.stack([c.ravel() for c in cols], 1).astype(np.float32)
    z = (rpe6 - base_radial).ravel().astype(np.float32)
    cm = core.ravel(); keep = cm.copy(); ang = np.zeros_like(z)
    for _ in range(n_ang_iter):
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        ang = A @ coef
        s = float(np.std((z - ang)[keep])) + 1e-6
        nk = cm & ((z - ang) > -k * s)
        if nk.sum() < A.shape[1] * 8:
            break
        keep = nk
    base = base_radial + ang.reshape(H, W)
    # 5. lift to g_base (p95 healthy level), then a SOFT cap (never above 1.10*p95) where the angular term
    #    pushed it up -- replaces the hard base_cap clip the trend baseline needs (no up-bow to clip here).
    base = base * (g_base / (float(np.median(base[core])) + 1e-6))
    base = np.minimum(base, soft_cap * g_base)
    return np.maximum(base, 1e-6).astype(np.float32)


def quality_flag(rpe_nat, stripe_thr=0.40):
    """NON-GATING per-eye slow-axis striping / low-SNR quality descriptor on the (already destriped) OAC
    RPE-loss native map rpe_nat (n_bscans=slow, W=fast). destripe2d removes the row LEVEL+gain; what
    SURVIVES is within-row stripe texture, captured here. Returns {stripe_pwr, anis_hf, low_confidence}.

    HONEST SCOPE (workflow ga-upstream-fixes, 2026-06-21, src/probe_striping3.py): striping does NOT
    predict the cohort area error -- Spearman(stripe_pwr, |ours-PLEX|) = -0.12 over 29 eyes; the high-
    striping eyes are mostly the ACCURATE ones. This is a specificity-1 DETECTOR for the single genuinely
    striping-driven blow-up (015 OS: stripe_pwr 0.45, z=2.64, PLEX 0.78 -> 5.66), surfaced as a 'low
    confidence' BADGE only -- never gate, never auto-drop, never touch the area number. It does NOT move
    the cohort limits of agreement (the real scatter is margin-trim / drusenoid-PED / the RPE-loss gate)."""
    rm = rpe_nat.mean(axis=1)
    mx = float(np.nanmax(rm)) if rpe_nat.size else 0.0
    vr = rm > 0.05 * mx if np.isfinite(mx) and mx > 0 else np.ones(len(rm), bool)
    c = rpe_nat[vr] if vr.sum() >= 8 else rpe_nat
    hf = c - gaussian_filter1d(c, 2.0, axis=0, mode="nearest")          # kill the low-freq lesion/falloff
    dr = float(np.percentile(c, 95) - np.percentile(c, 5)) + 1e-9
    stripe_pwr = float(np.sqrt(np.mean(np.diff(hf, axis=0) ** 2)) / dr)  # abs residual slow-axis HF energy
    anis_hf = float(np.mean(np.diff(hf, axis=0) ** 2) / (np.mean(np.diff(hf, axis=1) ** 2) + 1e-12))
    return {"stripe_pwr": stripe_pwr, "anis_hf": anis_hf,
            "low_confidence": bool(stripe_pwr >= stripe_thr)}


def prep(ov, bm, reducer="mean", smooth_px=2.0, margin_mm=0.30, baseline="radial2",
         trend_order=2, rpe_hi_pct=95.0, sig_frac=0.5, base_cap=1.15, radial=False,
         ilm=None, rpe_band="fixed", noise_floor=False, field_valid=None, quality=False,
         oac_vol=None, hyper_vol=None, core_override=None):
    """Per-eye prep on the open volume + (corrected) BM: the RPE-loss en-face, its smoothed copy, the
    measurement `core` (in-field, minus the low-SNR rim AND the low-signal vignette), and the healthy-RPE
    `base` field. Returns a dict {rpe6, rpe_nat, loss6, core, base, g_base} so callers can sweep the
    threshold without recomputing.

    EXPERIMENTAL knobs (defaults reproduce the validated path the reader/viewer rely on, byte-identical):
      rpe_band='peak'  -> sample OAC around the per-A-scan RPE peak (mp.band_peak_oac) instead of the fixed
                          BM-relative band, so a BM that dives under GA can't read 'RPE present' (FN fix).
      noise_floor=True -> subtract a per-B-scan vitreous noise floor in the OAC estimator (needs `ilm`).
      ilm              -> the ILM surface, required by noise_floor (and harmless otherwise).
      oac_vol/hyper_vol -> experiment-only radiometry inputs.  The field mask and vignette gate still
                          come from ``ov.vol``; pass the display arm's ``core`` as ``core_override`` to
                          hold even the OAC-derived support seed fixed in a radiometry experiment.
      core_override     -> experiment-only fixed en-face support mask. ``None`` preserves production."""
    oac_src = ov.vol if oac_vol is None else np.asarray(oac_vol)
    hyper_src = ov.vol if hyper_vol is None else np.asarray(hyper_vol)
    if oac_src.shape != ov.vol.shape or hyper_src.shape != ov.vol.shape:
        raise ValueError("oac_vol and hyper_vol must match ov.vol shape")
    floor = mp.vitreous_floor(oac_src, ilm) if (noise_floor and ilm is not None) else None
    oac = mp.oac_volume(oac_src, floor=floor)
    if rpe_band == "peak":
        rpe_raw = mp.band_peak_oac(oac, oac_src, bm)
    else:
        rpe_raw = mp.band(oac, bm, *mp.OAC_RPE_UM, reducer)
    rpe_nat = mp.destripe2d(rpe_raw, signed=False)
    fl = getattr(ov, "enface_flip", True)          # reverse-scanned rasters must NOT be row-flipped
    rpe6 = proj.to_enface(rpe_nat, ov.fov_mm, fl)
    if core_override is None:
        valid = gaussian_filter((rpe6 > 1e-6).astype(np.float32), smooth_px) > 0.5  # off-field pad
        # Exclude saturated machine-fill ('white band') columns: their OAC/baseline reading is meaningless
        # (a uniformly bright column is not real signal). Resample the native (n,W) validity the SAME way as
        # rpe_nat (NEAREST so it stays boolean) and remove invalid pixels from the field -> they fall out of
        # `core`, the healthy-RPE baseline fit, AND the footprint.
        if field_valid is None:
            field_valid = getattr(ov, "field_valid", None)
        if field_valid is not None and not np.all(field_valid):
            valid = valid & proj.to_enface_mask(np.asarray(field_valid, bool), ov.fov_mm, fl)
        margin_px = int(round(margin_mm / proj.ENFACE_MMPP))
        core = binary_erosion(valid, iterations=margin_px) if margin_px > 0 else valid
        if sig_frac > 0:
            # VIGNETTE gate: whole-column DISPLAY intensity, deliberately independent of experimental OAC.
            sig6 = gaussian_filter(proj.to_enface(ov.vol.mean(axis=1).astype(np.float32), ov.fov_mm, fl), smooth_px)
            core = core & (sig6 > sig_frac * float(np.nanpercentile(sig6[valid], 50)))
    else:
        core = np.asarray(core_override, bool)
        if core.shape != rpe6.shape:
            raise ValueError(f"core_override shape {core.shape} != en-face shape {rpe6.shape}")
    loss6 = gaussian_filter(rpe6, smooth_px)
    # 2nd cRORA criterion (for the combiner): sub-BM hypertransmission as INTENSITY (light reaching the
    # choroid = RPE gone). Catches GA centres where OAC-RPE-loss is fooled by bright material above BM.
    hyper6 = gaussian_filter(proj.to_enface(mp.destripe2d(m3_slab.hyper_enface(hyper_src, bm, *OAC_HYPER_UM),
                                                          signed=False), ov.fov_mm, fl), smooth_px)
    g_base = float(np.nanpercentile(rpe6[core], rpe_hi_pct)) + 1e-6
    if baseline == "trend":
        base = healthy_baseline(rpe6, core, order=trend_order, radial=radial)
        base = base * (g_base / (float(np.median(base[core])) + 1e-6))               # lift to the p95 level
        # CAP: the healthy-RPE baseline can't exceed ~p95. This stops the order-2 surface over-predicting
        # (bowing UP) at the far corners of a big eccentric lesion -> no peripheral false positives -- while
        # the DOWNWARD periphery falloff (needed to suppress FP on focal eyes like 005) is untouched.
        base = np.clip(base, 1e-6, base_cap * g_base)
    elif baseline == "radial2":
        base = anatomical_baseline(rpe6, core, g_base)     # foveal monotone-radial + angular (see fn doc)
    else:
        base = np.full(rpe6.shape, g_base, np.float32)
    return {"rpe6": rpe6, "rpe_nat": rpe_nat, "loss6": loss6, "hyper6": hyper6,
            "core": core, "base": base, "g_base": g_base,
            "quality": quality_flag(rpe_nat) if quality else None}    # non-gating striping badge (opt-in)


def footprint_stages(p, frac=0.50, min_diam_um=250.0, hyper_fill=True, close_mm=0.15,
                     hyper_frac=0.7, hyper_keep=0.4, fill_all_holes=True, hyper_abs=0.10,
                     min_depth=0.27):
    """Canonical, observable implementation of :func:`footprint`.

    Every intermediate is returned so experiments and explainers can attribute errors without copying
    detector logic.  Target/reference data are deliberately absent from this function.  The default
    ``final`` mask is byte-identical to the historical implementation.
    """
    from skimage import measure

    candidate = (p["loss6"] < frac * p["base"]) & p["core"]
    b = candidate.copy()
    rejected = np.zeros_like(b)
    holes_candidate = np.zeros_like(b)
    holes_filled = np.zeros_like(b)
    keep_thr = None
    fill_thr = None
    if hyper_fill and "hyper6" in p and b.any():
        h = p["hyper6"]
        keep_thr = max(hyper_keep * float(np.percentile(h[p["core"]], 75)), hyper_abs)
        kept = b & (h > keep_thr)
        rejected = b & ~kept
        b = kept
        if b.any():
            ci = max(1, int(round(close_mm / proj.ENFACE_MMPP / 2)))
            holes_candidate = binary_fill_holes(binary_closing(b, iterations=ci)) & ~b
            fill_thr = max(hyper_frac * float(np.percentile(h[b], 60)), hyper_abs)
            holes_filled = holes_candidate & (h > fill_thr)
            b = b | holes_filled

    hyper_kept = candidate & ~rejected if hyper_fill else candidate.copy()
    filled = binary_fill_holes(b) if fill_all_holes else b
    crora = fp.crora_stages(filled, min_diam_um)
    crora_hole_cleaned = crora["hole_cleaned"]
    sized = crora["sized"]

    partial_rejected = np.zeros_like(sized)
    final = sized
    if min_depth is not None and sized.any():
        ratio = p["loss6"] / np.maximum(p["base"], 1e-6)
        lbl = measure.label(sized)
        keep = np.zeros_like(sized)
        for rp in measure.regionprops(lbl):
            comp = lbl == rp.label
            if float(ratio[comp].min()) < float(min_depth):
                keep[comp] = True
        partial_rejected = sized & ~keep
        final = keep

    return {
        "rpe_candidate": candidate,
        "hyper_kept": hyper_kept,
        "hyper_rejected": rejected,
        "holes_candidate": holes_candidate,
        "holes_filled": holes_filled,
        "filled": filled,
        "crora_hole_cleaned": crora_hole_cleaned,
        "sized": sized,
        "partial_rejected": partial_rejected,
        "final": final,
        "keep_thr": keep_thr,
        "fill_thr": fill_thr,
        "config": {
            "frac": float(frac), "min_diam_um": float(min_diam_um),
            "hyper_fill": bool(hyper_fill), "close_mm": float(close_mm),
            "hyper_frac": float(hyper_frac), "hyper_keep": float(hyper_keep),
            "fill_all_holes": bool(fill_all_holes), "hyper_abs": float(hyper_abs),
            "min_depth": None if min_depth is None else float(min_depth),
        },
    }


def footprint(p, frac=0.50, min_diam_um=250.0, hyper_fill=True, close_mm=0.15, hyper_frac=0.7,
              hyper_keep=0.4, fill_all_holes=True, hyper_abs=0.10, min_depth=0.27):
    """(prep dict, threshold) -> (en-face cRORA GA mask, area_mm2). cRORA = RPE loss AND hypertransmission.

    hyper_abs (default 0.0 = OFF, byte-identical no-op): an ABSOLUTE physical floor on the per-eye
    scalar-normalised sub-BM intensity hyper6 (= slab(BM+130..250um) / (median(RPE band)+0.02), so an
    absolute constant is correct, NOT a fraction). The hyper gate is otherwise RELATIVE (a percentile of
    the eye's own h), so on a FLAT control its threshold is itself tiny and noise passes -> false-positive
    GA. Folding hyper_abs into both criteria via max() stops that (workflow ga-upstream-fixes, 2026-06-21,
    src/hyper_floor_experiment.py). Validated ON value 0.10: fixes control 002 OD (1.06 -> 0.08, <0.25)
    + tidies 006 OS, with EVERY GA eye held within 1.8% (005 OD/OS, 008 OD/OS, 015 OD pixel-identical;
    005 OD gold Dice 0.940 unchanged). PARTIAL fix only -- 016 OD / 009 OS are UNFIXABLE here: their
    sub-BM band is genuinely as bright as real GA, so they fire via the RPE-loss channel, not this gate
    (-> need a separate RPE-loss/BM lever, e.g. a DL-BM band-placement audit). If hyper6's normalisation
    ever changes, RECALIBRATE the floor.

    The RPE-loss channel alone has TWO failure modes that the hypertransmission channel (sub-BM intensity)
    fixes, because they fail on different things:
      - CORNERS (vignette / no retina) read low-OAC = false RPE-loss, but they don't transmit light ->
        low hypertransmission. So REQUIRE hypertransmission (criterion 1) -> the corners drop out. Safe for
        severe GA, which transmits strongly (unlike an inner-retina signal gate, which wrongly cuts centres).
      - CENTRES of large GA read 'RPE present' (bright material above BM / shallow BM), leaving an interior
        HOLE -> FILL holes that hypertransmit like the surrounding GA (criterion 2), keep holes that don't
        (spared RPE / foveal sparing). Only fills interior holes, so it adds no far-off false positives."""
    stages = footprint_stages(
        p, frac=frac, min_diam_um=min_diam_um, hyper_fill=hyper_fill, close_mm=close_mm,
        hyper_frac=hyper_frac, hyper_keep=hyper_keep, fill_all_holes=fill_all_holes,
        hyper_abs=hyper_abs, min_depth=min_depth,
    )
    mask = stages["final"]
    return mask, float(mask.sum()) * MMPP2


def detect(ov, bm, frac=0.50, min_diam_um=250.0, **prep_kw):
    """Convenience: prep + footprint in one call. Returns (rpe6, mask, area_mm2)."""
    p = prep(ov, bm, **prep_kw)
    mask, area = footprint(p, frac, min_diam_um)
    return p["rpe6"], mask, area


def overlay_png(rpe6, mask) -> bytes:
    """The RPE-loss en-face (dark = GA) with the cRORA footprint as a semi-transparent green FILL + a
    solid outline + a 1 mm scale bar. Holes (spared RPE inside the lesion) show through un-filled."""
    rgb = qv.ensure_rgb(qv.norm8(np.nan_to_num(np.asarray(rpe6, np.float32)))).astype(np.float32)
    m = np.asarray(mask, bool)
    rgb[m] = 0.45 * rgb[m] + 0.55 * np.array([0, 200, 0], np.float32)     # translucent green fill
    rgb = qv.draw_contour(rgb.astype(np.uint8), m, color=(0, 255, 0), thick=1)
    qv.add_scalebar(rgb, proj.ENFACE_MMPP, mm=1.0)
    return render.to_png(rgb)
