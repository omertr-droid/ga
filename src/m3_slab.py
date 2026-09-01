#!/usr/bin/env python
"""M3 prototype: turn the sub-BM slab into a clean hypertransmission en-face.

Raw mean-intensity below BM is dominated by per-B-scan gain (horizontal banding) and vessel
shadows, so low-contrast (small/multifocal) GA is buried. Fix = a ratio that is physically the
hypertransmission signal:

    hyper(A-scan) = mean(sub-BM slab, BM+64..BM+400 um) / mean(reference, BM-80..BM-15 um)

The reference band sits in the outer retina / RPE, BELOW the inner-retinal vessels, so vessel
shadows and overall gain cancel; where the RPE is atrophic the reference is dark and the slab is
bright -> high ratio = GA. A final per-B-scan de-band removes residual slow-axis striping.

Run: oct_env\\Scripts\\python.exe m3_slab.py [SUBJECT EYE ...]   (default 001 OD, 008 OD)
"""
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

import bm as bmseg
import m2_bm
import qcviz as qv

from paths import REPO_ROOT as ROOT, DATA_DIR, OUT_DIR, RESULTS_DIR
COH = os.path.join(ROOT, "cohort")
OUT = os.path.join(OUT_DIR, "m3_slab_out")
AX = bmseg.AXIAL_UM_PER_PX


def band_mean(vol, bmrow, lo_um, hi_um):
    """Per A-scan mean OCT intensity in [BM+lo_um, BM+hi_um] (um; may be negative = above BM)."""
    n, H, W = vol.shape
    lo = np.clip(np.round(bmrow + lo_um / AX), 0, H - 1).astype(int)
    hi = np.clip(np.round(bmrow + hi_um / AX), 1, H).astype(int)
    out = np.zeros((n, W), np.float32)
    for i in range(n):
        for x in range(W):
            a, b = lo[i, x], hi[i, x]
            if b > a:
                out[i, x] = vol[i, a:b, x].mean()
    return out


def deband(enface):
    """Remove the per-B-scan offset (subtract each row's 25th-pct = robust background, GA being a
    bright minority), then gently smooth (GA is a blob, so this lifts SNR without losing it)."""
    base = np.percentile(enface, 25, axis=1, keepdims=True)   # per-B-scan background level
    out = enface - base
    return gaussian_filter(out, sigma=(1.5, 1.5))


def hyper_enface(vol, bmrow, lo=130, hi=250):
    """Sub-BM hypertransmission en-face (native n x W) = mean intensity in the choroid band
    [BM+lo, BM+hi] um, per-EYE-scalar-normalised by the RPE band just above BM.

    GA: RPE atrophic -> light passes -> bright choroid -> high value. Healthy: RPE blocks -> dark.
    The per-eye scalar (median RPE) cancels gain without per-pixel blow-up.

    `lo,hi` default to the DEEP choroid band (130..250 um), kept for the legacy m4/diagnostic callers.
    The PRODUCTION GA path (reader.core.oac_ga, OAC_HYPER_UM) passes a SHALLOW band (~20..60 um, just
    under Bruch's): there the transmission contrast between complete GA and the dim periphery is strongest
    (003 OD ~5x the deep band) AND the band stays inside the choroid even where it thins peripherally -- a
    fixed deep band can punch into bright sclera and fake hypertransmission (workflow ga-hyper-depth)."""
    slab = band_mean(vol, bmrow, lo, hi)                 # choroid band below BM
    rpe = band_mean(vol, bmrow, -40, 10)                 # RPE band (bright; sets the per-eye scale)
    return slab / (np.median(rpe) + 0.02)               # per-EYE scalar norm (no per-pixel blow-up)


def process(subject, eye, vol, dev_bm):
    os.makedirs(OUT, exist_ok=True)
    n, H, W = vol.shape
    bmrow = m2_bm.fill_bm(dev_bm) if dev_bm is not None else bmseg.segment_volume(vol)
    src = "device BM" if dev_bm is not None else "self-seg BM"

    slab = band_mean(vol, bmrow, 64, 400)                # raw sub-BM slab (what looked bad)
    ref = band_mean(vol, bmrow, -80, -15)                # outer-retina / RPE reference
    hyper = slab / (ref + 0.02)                          # hypertransmission ratio
    hyper_db = deband(hyper)                             # + per-B-scan de-band

    sq = lambda a, p=99: cv2.resize(qv.norm8(np.clip(a, 0, np.percentile(a, p))),
                                    (512, 512), interpolation=cv2.INTER_LINEAR)
    adv = cv2.imread(os.path.join(COH, subject, eye, "advrpe_subrpe_enface.png"), cv2.IMREAD_GRAYSCALE)
    tiles = [sq(slab), sq(hyper), sq(hyper_db), adv if adv is not None else np.zeros((512, 512), np.uint8)]
    titles = ["raw slab (mean below BM)", "hyper = slab / RPE-ref", "+ de-band", "advRPE reference"]
    panel = qv.panel(tiles, titles, header=f"{subject} {eye}  M3 slab normalization ({src})  "
                                           f"-- full field vs advRPE 6x6")
    qv.save_rgb(os.path.join(OUT, f"{subject}_{eye}_m3.png"), panel)
    print(f"[{subject} {eye}] {src} -> {subject}_{eye}_m3.png", flush=True)


def main():
    args = sys.argv[1:]
    pairs = [("NHAMD-003-001-V1", "OD"), ("NHAMD-003-008-V1", "OD")] if not args else \
        [((args[i] if args[i].startswith("NHAMD") else "NHAMD-003-" + args[i]), args[i + 1])
         for i in range(0, len(args) - 1, 2)]
    by_sub = {}
    for s, e in pairs:
        by_sub.setdefault(s, []).append(e)
    for subject in by_sub:
        loaded = m2_bm.load_subject(subject)
        for eye in by_sub[subject]:
            if eye in loaded:
                process(subject, eye, *loaded[eye])
    print("wrote -> m3_slab_out/")


if __name__ == "__main__":
    main()
