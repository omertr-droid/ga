"""Compute everything the doctor viewer's 3 panels need from an opened E2E + a loaded volume + its
(validated/effective) ILM/BM surfaces.

Used by BOTH the offline baker (src/bake_library.py, writes a bundle) and the live upload path
(viewer.core.bundle.LiveSource, serves directly). Keeping it in one place means a library scan and an
uploaded scan render identically. PLEX (the advRPE reference overlay + area) is added by the baker;
uploaded scans have none.
"""
import cv2
import numpy as np

import m3_projections as mp
import qcviz as qv
from reader.core import oac_ga, projection as proj, render

from . import ga_native, locator

FEATURE = "f_trans"                                    # the en-face the doctor sees (GA bright)
SLAB_UM = (float(mp.SLAB_UM[0]), float(mp.SLAB_UM[1]))
AXIAL_UM_PER_PX = float(mp.AX)


def vol_u8(vol, field_valid=None):
    """Per-B-scan contrast-normalised uint8 stack (byte-identical to reader.core.render.bscan_png).
    `field_valid` (n,W bool, optional): exclude saturated machine-fill columns from each B-scan's
    contrast stretch (same _norm8_valid the reader uses), so reader and viewer stay byte-identical."""
    vc = (lambda i: None) if field_valid is None else (lambda i: field_valid[i])
    return np.stack([render._norm8_valid(vol[i], vc(i)) for i in range(vol.shape[0])]).astype(np.uint8)


def ga_overlay_png(out, mask):
    """Predicted-GA translucent-green fill + solid contour as an RGBA PNG sized (out,out), to composite
    over projection.png. Built directly in BGRA (green has B=R=0, so no channel swap needed)."""
    m = np.asarray(mask, bool)
    bgra = np.zeros((out, out, 4), np.uint8)
    bgra[m] = (0, 200, 0, 140)                                       # translucent green fill
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(bgra, cnts, -1, (0, 255, 0, 255), 1)           # solid green outline
    return cv2.imencode(".png", bgra)[1].tobytes()


def compute(raw, ov, ilm, bm, frac=0.50, baseline="radial2"):
    """Dict of arrays + PNG bytes for the 3 panels (no PLEX; the baker adds that separately).

    baseline = healthy-RPE baseline (PRODUCTION DEFAULT 'radial2' = foveal isotonic-radial + angular, the
    best on the linear/quad tradeoff; matches reader/CLI). The doctor viewer is read-only with one baked
    number per eye, so this single setting governs the whole library -> RE-BAKE (src/bake_library.py) after
    changing it so the shipped bundles match (the old library was baked LINEAR). The sub-BM hypertransmission
    floor (hyper_abs) is inherited from oac_ga.footprint's default."""
    out = ga_native.enface_out(ov.fov_mm)
    fl = getattr(ov, "enface_flip", True)          # False on the reverse-scanned rasters (003-016/003-130)
    # GA detection (OAC RPE-loss vs robust healthy baseline + hypertransmission), on the VALIDATED BM
    _rpe6, ga_enface, oac_area = oac_ga.detect(ov, bm, frac=frac, baseline=baseline)
    ga_nat = ga_native.enface_to_native(ga_enface, ov.fov_mm, ov.n_bscans, ov.W, fl)
    # the en-face projection the doctor sees (f_trans), windowed + 1 mm scale bar
    enf = proj.enface(ov, FEATURE, (ilm, bm))
    projection_png = render.projection_png(enf, proj.default_window(FEATURE))
    overlay_png = ga_overlay_png(out, ga_enface)
    # IR localizer (co-acquired series) + REAL per-B-scan locator lines
    loc = locator.pick_localizer(raw, ov.index)
    localizer_png = render.to_png(qv.norm8(loc)) if loc is not None else None
    loc_lines = locator.line_endpoints(raw, ov.index, loc.shape) if loc is not None else None
    return {
        "out": int(out),
        "vol_u8": vol_u8(ov.vol, ov.field_valid),
        "bm": np.asarray(bm, np.float32),
        "ilm": np.asarray(ilm, np.float32),
        "ga_enface": ga_enface.astype(bool),
        "ga_native": ga_nat.astype(bool),
        "field_invalid": (None if ov.field_invalid is None else ov.field_invalid.astype(bool)),
        "loc_lines": None if loc_lines is None else loc_lines.astype(np.float32),
        "slab_um": np.asarray(SLAB_UM, np.float32),
        "oac_area_mm2": float(oac_area),
        "localizer_png": localizer_png,
        "projection_png": projection_png,
        "ga_overlay_png": overlay_png,
        "localizer_sid": locator.localizer_sid(raw, ov.index),
        # False => the en-face rows were NOT reversed, so the viewer's click-probe must not un-reverse them.
        "enface_flip": bool(fl),
    }
