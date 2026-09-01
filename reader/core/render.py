"""Render numpy arrays to PNG bytes for the HTTP layer. Reuses src/qcviz primitives + cv2 encode.

The B-scan PNG is the RAW grayscale slice (the client overlays layers/bands on a separate canvas,
so toggling never refetches). The projection PNG uses a FIXED display window (cross-eye comparable,
matching the pipeline) + a mm scale bar. The localizer PNG carries the current B-scan position line.
"""
import cv2
import numpy as np

import qcviz as qv

from . import projection as proj

LINE_GREEN = (0, 255, 0)


def to_png(rgb_or_gray) -> bytes:
    """Encode a uint8 array (HxW gray or HxWx3 RGB) as PNG bytes."""
    a = np.asarray(rgb_or_gray)
    if a.ndim == 3:
        a = cv2.cvtColor(qv.ensure_rgb(a), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _norm8_valid(img, valid_cols):
    """qv.norm8 but the 1-99 percentile is taken over VALID columns only, so a saturated machine-fill
    'white band' is excluded from the contrast stats and stops darkening the real tissue. Byte-identical
    to qv.norm8 when there is no mask or no invalid column (the common case -> validated/clean B-scans
    render exactly as before)."""
    a = np.asarray(img, float)
    if valid_cols is None or bool(valid_cols.all()):
        return qv.norm8(a)
    sub = a[:, np.asarray(valid_cols, bool)]
    lo, hi = np.nanpercentile(sub, 1), np.nanpercentile(sub, 99)
    if hi <= lo:
        lo, hi = float(np.nanmin(sub)), float(np.nanmax(sub))
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1).__mul__(255).astype(np.uint8)


def neutralize_band(img8, valid_cols):
    """Replace saturated machine-fill ('white band') columns with their NEAREST VALID column, so a
    model is never fed an all-white stripe as input. The whole column (its real A-scan texture) is copied
    from the nearest in-field column, giving a plausible continuation rather than a flat patch. Valid
    columns are untouched. No-op when there is no mask / no invalid column. `valid_cols` is (W,) bool."""
    if valid_cols is None:
        return img8
    vc = np.asarray(valid_cols, bool)
    if bool(vc.all()) or not vc.any():
        return img8
    cols = np.flatnonzero(vc)
    src = np.round(np.interp(np.arange(vc.size), cols, cols)).astype(int)  # invalid -> nearest valid index
    return img8[:, src]


def bscan_model_input(ov, idx):
    """The B-scan image a DL model is fed (uint8 H×W): masked-contrast norm8 with saturated bands
    NEUTRALIZED (white columns replaced by their nearest valid neighbour). This is the SINGLE canonical
    model-input transform -- the BM-dataset export uses it for training images, and any inference path
    MUST use it too so train/infer stay matched (the model then never sees the all-white stripe). The
    human-facing display (bscan_png) deliberately KEEPS the band, shaded, so review stays honest."""
    vc = None if ov.field_valid is None else ov.field_valid[idx]
    return neutralize_band(_norm8_valid(ov.vol[idx], vc), vc)


def bscan_png(ov, idx) -> bytes:
    """Raw grayscale B-scan `idx` for DISPLAY (per-B-scan contrast). No overlay — client draws layers/
    bands. The contrast stretch ignores saturated machine-fill columns (see _norm8_valid); the band's
    pixels are KEPT (the client shades them) -- model input neutralizes them instead (bscan_model_input)."""
    vc = None if ov.field_valid is None else ov.field_valid[idx]
    return to_png(_norm8_valid(ov.vol[idx], vc))


def windowed(map_float, window):
    """Float en-face -> uint8 at a FIXED [lo,hi] display window (not percentile)."""
    lo, hi = window
    u = np.clip((np.asarray(map_float, float) - lo) / (hi - lo + 1e-9), 0, 1)
    return (u * 255).astype(np.uint8)


def projection_png(map_float, window, scalebar=True) -> bytes:
    """En-face projection at a fixed window, grayscale, with a 1 mm scale bar."""
    rgb = qv.ensure_rgb(windowed(map_float, window))
    if scalebar:
        qv.add_scalebar(rgb, proj.ENFACE_MMPP, mm=1.0)
    return to_png(rgb)


def localizer_png(loc_gray, idx, n, flip=True) -> bytes:
    """IR localizer with a horizontal line at the current B-scan's (approximate) position.

    NOTE: the line position is a linear map of idx over the localizer height (B-scan 0 at the bottom
    when flip=True, matching the OCT->fundus row flip). Exact placement from per-B-scan centrePosY is
    a later refinement.
    """
    rgb = qv.ensure_rgb(qv.norm8(loc_gray))
    h, w = rgb.shape[:2]
    if idx >= 0 and n > 1:        # idx < 0 => return the clean base (the client draws its own line)
        f = (n - 1 - idx) / (n - 1) if flip else idx / (n - 1)
        y = int(round(np.clip(f, 0, 1) * (h - 1)))
        cv2.line(rgb, (0, y), (w - 1, y), LINE_GREEN, 2, cv2.LINE_AA)
    return to_png(rgb)
