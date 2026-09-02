#!/usr/bin/env python
"""En-face PROJECTION bake-off: which projection best reveals GA hypertransmission from the
B-scans, given a good (device) BM?  Candidates, all flattened to BM:

  raw       mean intensity in BM+64..+400 um            (baseline = what we had; gain/shadow biased)
  transmit  SUM(below BM) / SUM(around BM)              (transmission fraction; gain+shadow cancel)
  oac       mean optical attenuation coeff. below BM    (Vermeer; tissue property, gain/shadow cancel)
  normslab  mean(deep choroid) / mean(RPE band)         (diagnostic-based local ratio)

Each is computed in the native field, scored for BANDING (per-B-scan striping; lower better) and
GA CONTRAST + SIMILARITY to the advRPE SubRPE en-face (higher better), then shown per eye next to
the advRPE reference. Goal: pick the cleanest foundation for the detector (DL or not).

Run: oct_env\\Scripts\\python.exe m3_projections.py [all | SUBJECT EYE ...]   (default: a spread)
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import m2_bm
import qcviz as qv
import register_qc as reg

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
OUT = os.path.join(OUT_DIR, "m3_proj_out")
AX = m2_bm.bmseg.AXIAL_UM_PER_PX

DEFAULT = [("NHAMD-003-009-V2", "OS"), ("NHAMD-003-002-V2", "OD"), ("NHAMD-003-012-V3", "OD"),
           ("NHAMD-003-010-V1", "OS"), ("NHAMD-003-017-V3", "OD"), ("NHAMD-003-006-V3", "OD"),
           ("NHAMD-003-005-V3", "OD"), ("NHAMD-003-001-V1", "OD"), ("NHAMD-003-003-V3", "OS"),
           ("NHAMD-003-004-V1", "OD"), ("NHAMD-003-008-V1", "OD")]


def band(vol, bm, lo_um, hi_um, op):
    """Per-A-scan reduce (mean/sum/max) of `vol` over the BM-relative band [lo_um, hi_um]."""
    n, H, W = vol.shape
    lo = np.clip(np.round(bm + lo_um / AX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bm + hi_um / AX), 1, H).astype(int)
    out = np.zeros((n, W), np.float32)
    for i in range(n):
        loi, hii = lo[i], hi[i]
        for x in range(W):
            a, b = loi[x], hii[x]
            if b > a:
                seg = vol[i, a:b, x]
                out[i, x] = (seg.sum() if op == "sum" else
                             seg.max() if op == "max" else seg.mean())
    return out


def band_argmax_row(vol, bm, lo_um, hi_um):
    """Per-A-scan ABSOLUTE row of the max of `vol` within the BM-relative band [lo_um, hi_um].
    Used to locate the RPE/EZ peak above BM for the RPE->BM elevation channel."""
    n, H, W = vol.shape
    lo = np.clip(np.round(bm + lo_um / AX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bm + hi_um / AX), 1, H).astype(int)
    out = np.zeros((n, W), np.float32)
    for i in range(n):
        loi, hii = lo[i], hi[i]
        for x in range(W):
            a, b = loi[x], hii[x]
            out[i, x] = (a + int(np.argmax(vol[i, a:b, x]))) if b > a else max(0, b - 1)
    return out


def oac_volume(vol, floor=None):
    """Depth-resolved optical attenuation coefficient (Vermeer 2014): mu[z]=I[z]/(2*sum_{z'>z}I).

    floor (optional per-B-scan (n,) noise level): subtract it before the cumsum and widen the
    denominator floor to 0.5*floor, so below-noise pixels (vignette corners / deep tails) stop
    inflating mu -- this attacks the estimator's variance blow-up at its physical source instead of
    masking it downstream. floor=None reproduces the bare estimator EXACTLY (the reader/viewer path)."""
    I = np.asarray(vol, np.float32)
    if floor is not None:
        I = I - np.asarray(floor, np.float32).reshape(-1, 1, 1)
    I = np.clip(I, 0, None)
    below = np.cumsum(I[:, ::-1, :], axis=1)[:, ::-1, :] - I     # sum strictly below z
    eps = 1e-3 if floor is None else np.maximum(1e-3, 0.5 * np.asarray(floor, np.float32).reshape(-1, 1, 1))
    return I / (2.0 * below + eps)


def vitreous_floor(vol, ilm, k=3.0, min_top_px=6):
    """Per-B-scan background (noise) level over the VITREOUS band above the ILM: robust
    median + k*1.4826*MAD on rows [0 : min(ILM)-4], clamped to the top quarter so the band can never
    clip retina. Feeds oac_volume(floor=...). Needs the ILM (the vitreous is reliably dark only there)."""
    n, H, W = vol.shape
    ilm = np.asarray(ilm, np.float32)
    floor = np.zeros(n, np.float32)
    for i in range(n):
        finite = np.isfinite(ilm[i])
        top0 = (float(np.min(ilm[i][finite])) if finite.any() else H / 5.0) - 4.0
        top = int(np.clip(top0, min_top_px, H // 4))
        seg = vol[i, :top, :].astype(np.float32)
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        floor[i] = med + k * 1.4826 * mad
    return floor


def proj_raw(vol, bm):
    return band(vol, bm, 64, 400, "mean")


# Transmission-fraction bands (um, relative to BM) -- the SINGLE source of truth for both the
# computation and the B-scan visualisation, so the green can never drift from what's summed.
SLAB_UM = (10, 340)     # numerator: all light BELOW BM (the transmitted-light slab)   -> GREEN
REF_UM = (-250, 340)    # denominator: normalization window (gain/shadow cancel; above-BM RPE
                        #              vanishes in GA, boosting the fraction)            -> bracket
AX = m2_bm.bmseg.AXIAL_UM_PER_PX


def proj_transmit(vol, bm):
    """Transmission fraction per A-scan: sum(below BM) / sum(normalization window). Higher in GA.
    (Legacy BM-offset normalization; prefer proj_transmit_ilm which anchors the top to the ILM.)"""
    return band(vol, bm, *SLAB_UM, "sum") / (band(vol, bm, *REF_UM, "sum") + 1e-3)


def band_surfaces(vol, top, bot, op="sum"):
    """Reduce (sum/mean) per A-scan between two per-(i,x) row surfaces top..bot."""
    n, H, W = vol.shape
    lo = np.clip(np.round(top), 0, H - 1).astype(int)
    hi = np.clip(np.round(bot), 1, H).astype(int)
    out = np.zeros((n, W), np.float32)
    for i in range(n):
        loi, hii = lo[i], hi[i]
        for x in range(W):
            a, b = loi[x], hii[x]
            if b > a:
                seg = vol[i, a:b, x]
                out[i, x] = seg.sum() if op == "sum" else seg.mean()
    return out


def proj_transmit_ilm(vol, ilm, bm):
    """ILM-anchored transmission fraction. numerator = light below BM (BM+10..+340, choroid);
    denominator = the whole retina+choroid column from the REAL ILM to BM+340 -- so the normalization
    top follows true anatomy (curving ILM), not a fixed BM offset that drifts into the vitreous where
    the retina is thin. Blowup-clipped to [0,1.5] (the fraction is <=1 by construction)."""
    num = band(vol, bm, SLAB_UM[0], SLAB_UM[1], "sum")
    den = band_surfaces(vol, ilm, bm + SLAB_UM[1] / AX, "sum")
    return np.clip(num / (den + 1e-3), 0, 1.5)


# RPE-loss bands (um, BM-relative).
RPE_UM = (-50, -5)       # outer-retinal hyperreflective complex (EZ/IZ/RPE) -- bright normal, gone in GA
REFWIN_UM = 150          # half-height of the local outer column the RPE peak is measured against


def proj_rpe_loss_ilm(vol, ilm, bm):
    """SHADOW-INVARIANT RPE-loss cue = how much the RPE band stands out of its own local outer column.
    rpe = mean(BM-50..-5) (the EZ/IZ/RPE complex band); ref = mean of the local column BM-150..+150
    (clamped to the real ILM so it never reaches the vitreous). loss = 1 - rpe/ref.
      - Healthy: the RPE band is the bright PEAK of that column -> rpe>ref -> loss < 0.
      - GA: no RPE peak AND hypertransmission brightens the sub-BM part of the column -> rpe<=ref ->
        loss > 0. So transmission REINFORCES the loss signal here (it doesn't erase it as a fixed
        outer-band mean did on big confluent GA).
    rpe and ref are equally attenuated by any overlying vessel shadow / gain, so both cancel in the
    ratio. Clipped to [-1, 1]. High = RPE peak gone = GA."""
    rpe = band(vol, bm, RPE_UM[0], RPE_UM[1], "mean")
    top = np.maximum(ilm, bm - REFWIN_UM / AX)               # local column top, never above ILM
    ref = band_surfaces(vol, top, bm + REFWIN_UM / AX, "mean")
    return np.clip(1.0 - rpe / (ref + 1e-3), -1.0, 1.0)


# RPE-integrity, transmission-INDEPENDENT: the RPE/EZ complex measured against the inner retina, both
# strictly ABOVE BM, so no sub-BM hypertransmission can leak in (unlike proj_rpe_loss_ilm, whose ref
# dips below BM and is therefore correlated with transmission).
RPEBAND_UM = (-45, -5)      # outer hyperreflective complex (EZ/IZ/RPE) -- bright when present, gone in GA
INNER_UM = (-165, -70)      # inner/middle-retina reference -- preserved in GA, never below the RPE band
GATE_LO, GATE_HI = 1.0, 1.9  # rpe_present ratio: <=LO => RPE gone (gate 1); >=HI => RPE present (gate 0)


def proj_rpe_present_ilm(vol, ilm, bm):
    """Transmission-INDEPENDENT RPE prominence = mean(RPE/EZ band BM-45..-5) / mean(inner retina
    BM-165..-70, clamped to the ILM). Both bands are ABOVE BM, so this is blind to choroidal
    hypertransmission below BM. Healthy: the RPE complex is the brightest retinal band -> ratio >> 1.
    GA: the RPE/EZ band is gone, falling to inner-retina level -> ratio ~1 or below. High = RPE present
    (intact); low = RPE gone. Clipped [0,5]."""
    rpe = band(vol, bm, RPEBAND_UM[0], RPEBAND_UM[1], "mean")
    top = np.maximum(ilm, bm + INNER_UM[0] / AX)            # never above the ILM (into vitreous)
    inner = band_surfaces(vol, top, bm + INNER_UM[1] / AX, "mean")
    return np.clip(rpe / (inner + 1e-3), 0.0, 5.0)


def rpe_surface(vol, bm, search_um=110.0, near_um=3.0, ref_lo_um=220.0, ref_hi_um=120.0, smooth=2.0):
    """Peak-track the RPE/EZ complex (the brightest outer band just ABOVE BM) and score its prominence.
    Unlike proj_rpe_present_ilm's FIXED band (BM-45..-5, which over an elevated drusen RPE samples the
    DEPOSIT below it and so falsely reads 'RPE gone'), this SEARCHES the outer zone for the band and
    follows it UP over drusen (validated: tracks the elevated RPE on 011). Returns
      row[n,W]  = the RPE-surface row (meaningful where prom is high; arbitrary where the RPE is gone),
      prom[n,W] = peak / inner-retina ratio -- >~1.5 => a real RPE band is present (incl. attenuated over
                  drusen), ~1 or below => no peak => RPE absent/faded.
    Localisation is robust; the prominence is the present/absent cue (mid values = faded = borderline)."""
    vol = np.asarray(vol, float)
    if vol.ndim == 2:
        vol = vol[None]
    n, H, W = vol.shape
    bm = np.asarray(bm, float)
    row = np.zeros((n, W), np.float32)
    prom = np.zeros((n, W), np.float32)
    for i in range(n):
        c = gaussian_filter1d(vol[i], smooth, axis=0)
        for x in range(W):
            bx = bm[i, x]
            a = int(np.clip(bx - search_um / AX, 0, H - 1))
            z = int(np.clip(bx - near_um / AX, 1, H))
            if z <= a:
                row[i, x] = bx
                continue
            seg = c[a:z, x]
            k = a + int(np.argmax(seg))
            pv = c[max(0, k - 1):k + 2, x].mean()
            ra = int(np.clip(bx - ref_lo_um / AX, 0, H - 1))
            rz = int(np.clip(bx - ref_hi_um / AX, 1, H))
            ref = c[ra:rz, x].mean() if rz > ra else seg.mean()
            row[i, x] = k
            prom[i, x] = pv / (ref + 1e-3)
    return row, prom


def band_peak_oac(oac, vol, bm, half_um=14.0, absent_prom=1.15, clamp_um=(-22.0, -8.0),
                  search_um=110.0, near_um=3.0):
    """PEAK-ANCHORED OAC RPE-loss sampler (the BM-dive fix). Locate the RPE/EZ peak per A-scan with
    rpe_surface on the INTENSITY volume (drusen-aware -- it follows the band UP over deposits) and
    average OAC in a +-half_um window AROUND that peak, decoupling the read from BM so a BM that DIVES
    into bright sub-RPE under GA can no longer drag the band into a false 'RPE present'. Where the RPE
    is genuinely gone (prominence < absent_prom -> no peak to track), CLAMP the window to just above BM
    (clamp_um) so a gone-RPE column reads the genuinely-low OAC of empty outer retina, not bright
    dived-BM material. Returns (n,W); drop-in for `band(oac, bm, *OAC_RPE_UM, 'mean')`."""
    row, prom = rpe_surface(vol, bm, search_um=search_um, near_um=near_um)
    bmf = np.asarray(bm, np.float32)
    half = half_um / AX
    gone = prom < absent_prom
    top = np.where(gone, bmf + clamp_um[0] / AX, row - half)
    bot = np.where(gone, bmf + clamp_um[1] / AX, row + half)
    return band_surfaces(oac, top, bot, "mean")


def rpe_gone_gate(rpe_present, lo=GATE_LO, hi=GATE_HI):
    """Map the RPE-present ratio -> an 'RPE-gone' weight in [0,1]: 1 where the RPE is gone (ratio<=lo),
    0 where it is intact (ratio>=hi). Multiply transmission by this to keep hypertransmission ONLY where
    the RPE is actually absent (suppresses bright choroid under intact RPE)."""
    return np.clip((hi - rpe_present) / (hi - lo), 0.0, 1.0)


GATE_SMOOTH_PX = 4.0    # spatial smoothing of the gate in the 6 mm frame (~47 um << 250 um cRORA scale)


def gated_feature(t_nat, p_nat, fov, slow_sigma=2.0, gate_smooth=GATE_SMOOTH_PX):
    """The deliverable feature: transmission gated by RPE-integrity = hypertransmission ONLY where the
    RPE is actually gone. Destripe both natives, project to the 6 mm frame, then SPATIALLY SMOOTH the
    RPE-present map before gating -- GA is spatially coherent (>=250 um), so smoothing the gate (well
    below lesion scale) denoises the per-pixel specificity decision and kills the speckle false-positives
    that a raw per-pixel gate produced on controls -- then multiply by transmission (whose detail is
    kept). High = hypertransmission with absent RPE = GA."""
    f_trans = to_6mm(destripe2d(t_nat, slow_sigma, signed=False), fov)
    pres6 = to_6mm(destripe2d(p_nat, slow_sigma, signed=False), fov)
    gate = rpe_gone_gate(gaussian_filter(pres6, gate_smooth))
    return np.clip(f_trans, 0.0, None) * gate


def bscan_bands(bscan, bm_row, dev_row=None, ilm_row=None):
    """B-scan with the ACTUAL transmission bands drawn: green = summed sub-BM slab (numerator),
    ORANGE line = top of the normalization window = the real ILM (if given), yellow = BM (ours),
    cyan = device BM."""
    rgb = qv.ensure_rgb(qv.norm8(bscan))
    H, W = bscan.shape[:2]
    slab_lo, slab_hi = SLAB_UM[0] / AX, SLAB_UM[1] / AX
    top_row = ilm_row if ilm_row is not None else (bm_row + REF_UM[0] / AX)
    band_img = rgb.copy()
    for x in range(W):
        y0, y1 = int(round(bm_row[x] + slab_lo)), int(round(bm_row[x] + slab_hi))
        if 0 <= y0 < H:
            band_img[y0:min(y1, H), x] = (0, 180, 0)                 # numerator slab (green)
    rgb = cv2.addWeighted(rgb, 0.62, band_img, 0.38, 0)
    for x in range(W):
        yt = top_row[x]
        if np.isfinite(yt) and 0 <= int(yt) < H:
            rgb[max(0, int(yt)):int(yt) + 1, x] = (255, 150, 0)      # orange = ILM (norm-window top)
        if dev_row is not None and np.isfinite(dev_row[x]) and 0 <= int(dev_row[x]) < H:
            rgb[max(0, int(dev_row[x]) - 1):int(dev_row[x]) + 1, x] = (0, 220, 255)
        yb = int(round(bm_row[x]))
        if 0 <= yb < H:
            rgb[max(0, yb - 1):yb + 1, x] = (255, 255, 0)           # BM (yellow)
    tr = top_row[np.isfinite(top_row)]
    lo = max(0, int(tr.min()) - 20) if tr.size else max(0, int(bm_row.min()) - 80)
    hi = min(H, int(bm_row.max() + slab_hi) + 25)
    crop = rgb[lo:hi]
    return cv2.resize(crop, (W, crop.shape[0] * 3), interpolation=cv2.INTER_NEAREST)


def proj_oac(vol, bm):
    return band(oac_volume(vol), bm, 10, 220, "mean")


# BM-anchored OAC channels: Vermeer mu sampled ABOVE BM, where the estimator is well-conditioned (the
# legacy proj_oac above samples the noisy deep choroid BELOW BM). Sign convention of the um offsets:
# NEGATIVE = above BM (shallower / toward vitreous), POSITIVE = below BM (deeper / into choroid).
OAC_RPE_UM = (-50, -8)      # EZ/IZ/RPE complex band above BM (8um off BM so its own OAC spike can't leak)
OAC_SUBBM_UM = (20, 220)    # near-choroid below BM (start past the BM/Bruch spike) -> hypertransmission


def proj_oac_rpe_above_bm(vol, bm):
    """PRIMARY GA signal: max Vermeer-OAC in the RPE band above BM. HIGH where the RPE is present (it
    strongly attenuates light), LOW where the RPE is gone = GA. (n,W)."""
    return band(oac_volume(vol), bm, OAC_RPE_UM[0], OAC_RPE_UM[1], "max")


def proj_oac_rpe_elevation(vol, bm):
    """Drusen-vs-GA discriminator: (BM row - OAC-peak row) in um. Large = RPE lifted onto a deposit
    (drusen; RPE alive => GA-free); ~0 = flat (healthy RPE or flat atrophy). (n,W) um."""
    peak = band_argmax_row(oac_volume(vol), bm, OAC_RPE_UM[0], OAC_RPE_UM[1])
    return np.clip((np.asarray(bm, np.float32) - peak) * AX, 0.0, None)


def proj_oac_subbm(vol, bm):
    """sub-BM hypertransmission (2nd cRORA criterion): mean Vermeer-OAC in the near-choroid band below
    BM. HIGH where light penetrates the choroid = RPE gone = GA. (n,W)."""
    return band(oac_volume(vol), bm, OAC_SUBBM_UM[0], OAC_SUBBM_UM[1], "mean")


def proj_normslab(vol, bm):
    return band(vol, bm, 130, 250, "mean") / (band(vol, bm, -40, 10, "mean") + 0.02)


PROJS = [("raw", proj_raw), ("transmit", proj_transmit), ("oac", proj_oac), ("normslab", proj_normslab),
         ("oac_rpe", proj_oac_rpe_above_bm), ("oac_subbm", proj_oac_subbm)]


def destripe(nat):
    """Remove HIGH-FREQUENCY slow-axis (row-to-row) banding, preserving low-freq GA.

    The banding is per-B-scan jitter (high-freq along the slow axis); GA spans many B-scans
    (low-freq). So divide each B-scan by its level relative to the slow-axis-smoothed level -- this
    cancels the row-to-row stripe but keeps the smooth lesion envelope (unlike subtracting the full
    per-row background, which erased large confluent GA)."""
    r = np.median(nat, axis=1)
    gain = r / (gaussian_filter1d(r, 3) + 1e-6)
    return nat / (gain[:, None] + 1e-9)


def destripe_add(nat):
    """ADDITIVE slow-axis destripe for SIGNED maps (e.g. the RPE-loss ratio): subtract each B-scan's
    offset relative to the slow-axis-smoothed row level. destripe() above is multiplicative, correct
    for the strictly-positive transmission fraction; a signed ratio centred near zero bands additively
    (and the multiplicative form is unstable where the row level crosses zero), so subtract, not divide.
    The low-freq lesion (spanning many B-scans) survives the slow-axis smoothing."""
    r = np.median(nat, axis=1)
    off = r - gaussian_filter1d(r, 3)
    return nat - off[:, None]


def destripe2d(nat, slow_sigma=2.0, signed=False):
    """Robust per-B-scan destripe (bake-off winner `med+gain`, ~6x flatter than raw on controls). The
    banding is a per-B-scan LEVEL shift -- high-frequency along the slow axis (separately-acquired
    B-scans jitter row-to-row); the lesion is LOW-frequency (spans many B-scans). Split: low =
    slow-axis-smoothed (holds the lesion), hi = nat - low (holds banding + noise, NOT the lesion). Then
    two passes:
      1. ADDITIVE: subtract the robust per-row MEDIAN of hi (ignores localized lesion edges leaking into
         hi) -> removes the per-B-scan offset without eating the lesion (the failure of subtracting the
         full per-row background).
      2. MULTIPLICATIVE: divide by the per-row gain relative to its slow-axis-smoothed level -> removes
         residual per-B-scan gain.
    signed=True (the RPE-loss ratio, which crosses zero) is shifted strictly positive before the gain
    divide and shifted back, so the divide stays stable."""
    shift = 2.0 if signed else 0.0
    work = nat + shift
    hi = work - gaussian_filter1d(work, slow_sigma, axis=0, mode="nearest")
    work = work - np.median(hi, axis=1, keepdims=True)
    rm = np.mean(work, axis=1)
    gain = rm / (gaussian_filter1d(rm, slow_sigma) + 1e-6)
    work = work / (gain[:, None] + 1e-9)
    return work - shift


def to_6mm(nat, fov, flip=True):
    # NB: the OCT volume's B-scan order is vertically flipped vs the IR localizer / advRPE (PLEX),
    # so flip rows (slow axis) to bring our en-face into the same fundus orientation as the reference.
    # `flip=False` for the two subjects whose raster runs the slow axis in REVERSE (003-016, 003-130):
    # their rows are already fundus-ordered. See reader.core.e2e_source.enface_flip_for.
    src = np.asarray(nat, np.float32)
    src = src[::-1] if flip else src
    return reg.resample(src, (fov[0] / src.shape[1], fov[1] / src.shape[0]))


def banding_score(nat):
    """Per-B-scan striping amplitude relative to total variation (lower = cleaner)."""
    rowmean = nat.mean(axis=1)
    resid = rowmean - gaussian_filter1d(rowmean, 4)
    return float(resid.std() / (nat.std() + 1e-6))


def ga_contrast(p6, advm):
    if advm.sum() < 50 or (~advm).sum() < 50:
        return float("nan")
    ga, non = p6[advm], p6[~advm]
    return float((ga.mean() - non.mean()) / (non.std() + 1e-6))


def similarity(p6, adv6):
    a = (p6 - p6.mean()) / (p6.std() + 1e-6)
    b = (adv6 - adv6.mean()) / (adv6.std() + 1e-6)
    return float((a * b).mean())


def process(subject, eye, vol, dev_bm):
    os.makedirs(OUT, exist_ok=True)
    if dev_bm is None:                       # bake-off uses GOOD (device) BM only
        print(f"[{subject} {eye}] skip (no device BM)", flush=True)
        return None
    bm = m2_bm.fill_bm(dev_bm)
    with open(os.path.join(COH, subject, "meta.json")) as f:
        meta = json.load(f)["eyes"][eye]
    fov = [float(v) for v in meta["fov_mm"]]
    area = float(meta.get("advRPE_area_mm2") or 0.0)
    adv = cv2.imread(os.path.join(COH, subject, eye, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
    adv6 = adv.astype(np.float32) if adv is not None else np.zeros((512, 512), np.float32)
    advm = cv2.imread(os.path.join(COH, subject, eye, "ga_mask.png"), cv2.IMREAD_GRAYSCALE)
    advm = (advm > 127) if advm is not None else np.zeros((512, 512), bool)

    tiles, titles, row = [], [], {"subject": subject, "eye": eye, "area": round(area, 2)}
    for name, fn in PROJS:
        nat = destripe(fn(vol, bm))
        bnd = banding_score(nat)
        p6 = to_6mm(nat, fov)
        con, sim = ga_contrast(p6, advm), similarity(p6, adv6)
        row[f"{name}_band"] = round(bnd, 3)
        row[f"{name}_con"] = round(con, 2)
        row[f"{name}_sim"] = round(sim, 2)
        tiles.append(qv.norm8(np.clip(p6, np.percentile(p6, 1), np.percentile(p6, 99))))
        titles.append(f"{name}  band={bnd:.2f} C={con:.2f} sim={sim:.2f}")
    tiles.append(qv.draw_contour(adv, advm, color=(0, 220, 255), thick=2))
    titles.append(f"advRPE SubRPE ({area:.2f} mm2)")
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_proj.png"),
                qv.panel(tiles, titles, header=f"{subject} {eye}  projection bake-off (GA={area:.2f} mm2)"))
    print(f"[{subject} {eye}] " + "  ".join(
        f"{n}:band{row[n+'_band']:.2f}/C{row[n+'_con']}/sim{row[n+'_sim']}" for n, _ in PROJS), flush=True)
    return row


def main():
    args = sys.argv[1:]
    if args and args[0].lower() == "all":
        with open(os.path.join(RESULTS_DIR, "spectralis_ga_pairing.csv"), newline="") as f:
            pairs = [(r["subject"], r["eye"]) for r in csv.DictReader(f) if r.get("qc_status") == "ok"]
    elif args:
        pairs = [((args[i] if args[i].startswith("NHAMD") else "NHAMD-003-" + args[i]), args[i + 1])
                 for i in range(0, len(args) - 1, 2)]
    else:
        pairs = DEFAULT

    by_sub = {}
    for s, e in pairs:
        by_sub.setdefault(s, []).append(e)
    rows = []
    for subject in sorted(by_sub):
        loaded = m2_bm.load_subject(subject)
        for eye in by_sub[subject]:
            if eye in loaded:
                r = process(subject, eye, *loaded[eye])
                if r is not None:
                    rows.append(r)

    cols = ["subject", "eye", "area"] + [f"{n}_{m}" for n, _ in PROJS for m in ("band", "con", "sim")]
    with open(os.path.join(RESULTS_DIR, "m3_projections.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # aggregate: banding on controls (area<0.05), contrast/similarity on GA eyes (area>=1)
    ctrl = [r for r in rows if r["area"] < 0.05]
    ga = [r for r in rows if r["area"] >= 1.0]
    print("\n=== aggregate ===  (banding: lower better | contrast/sim: higher better)")
    for n, _ in PROJS:
        b = np.nanmean([r[f"{n}_band"] for r in ctrl]) if ctrl else float("nan")
        c = np.nanmean([r[f"{n}_con"] for r in ga]) if ga else float("nan")
        s = np.nanmean([r[f"{n}_sim"] for r in ga]) if ga else float("nan")
        print(f"  {n:9} banding(ctrl)={b:.3f}   GA-contrast={c:.2f}   similarity={s:.2f}")
    print(f"\nwrote {len(rows)} eyes -> m3_proj_out/  + m3_projections.csv")


if __name__ == "__main__":
    main()
