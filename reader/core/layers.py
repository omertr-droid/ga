"""Layer + projection-band model, the JSON the viewer draws, and the correction-store seam.

Two device layers are shown (the only dense ones the E2E carries): ILM (contour0) and BM (contour1).
On top, the viewer can shade the projection's measurement BANDS on each B-scan so you can SEE what
the transmission map integrates:
  - slab = sub-BM transmitted-light window (BM+10..+340 um), the transmission numerator
  - rpe  = the EZ/IZ/RPE band (BM-45..-5 um) that vanishes in GA
Band offsets are sent in um + axial_um_per_px, so the client converts to pixel rows from the BM
polyline (no extra fetch). Constants are imported from m3_projections so they can never drift.
"""
from typing import Optional, Protocol

import numpy as np

import m3_projections as mp

from . import calibration as cal
from .volume import OctVolume

# colours match src/m3_projections.bscan_bands (ILM orange, BM yellow, slab green)
LAYER_DEFS = [
    {"key": "ilm", "name": "ILM", "color": [255, 150, 0]},
    {"key": "bm", "name": "BM", "color": [255, 255, 0]},
]
BAND_DEFS = [
    {"key": "slab", "name": "transmission slab (BM+10..+340um)",
     "lo_um": float(mp.SLAB_UM[0]), "hi_um": float(mp.SLAB_UM[1]), "color": [0, 200, 0]},
    {"key": "rpe", "name": "RPE/EZ band (BM-45..-5um)",
     "lo_um": float(mp.RPEBAND_UM[0]), "hi_um": float(mp.RPEBAND_UM[1]), "color": [255, 60, 200]},
]


def _surface_json(arr):
    """(n,W) float surface -> list of rows, NaN -> None, rounded (compact, JSON-safe)."""
    a = np.asarray(arr, float)
    return [[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in a]


def _invalid_runs_json(invalid):
    """(n,W) bool field-invalid mask -> per-B-scan list of [start,end] inclusive column runs (compact:
    an empty list where the B-scan has no saturated band). The client shades these columns."""
    out = []
    for row in np.asarray(invalid, bool):
        runs, x, W = [], 0, len(row)
        while x < W:
            if row[x]:
                s = x
                while x < W and row[x]:
                    x += 1
                runs.append([s, x - 1])
            else:
                x += 1
        out.append(runs)
    return out


def _runs(mask):
    """(W,) bool -> list of [start,end] inclusive column runs (same encoding as _invalid_runs_json)."""
    runs, x, W = [], 0, len(mask)
    while x < W:
        if mask[x]:
            s = x
            while x < W and mask[x]:
                x += 1
            runs.append([int(s), int(x - 1)])
        else:
            x += 1
    return runs


def _bm_missing_row(ov, store, bscan):
    """(W,) bool — columns with NO real BM value on this B-scan: NEITHER the raw device contour
    (ov.bm_display) NOR a saved BM correction provides one, and the column isn't a saturated
    field-invalid (white-band) column. These must be filled (Label-with-DL or a manual edit) before
    the B-scan can be validated. Uses the RAW device line, NOT ov.bm / effective_surfaces (those are
    fill_bm-interpolated and so never look missing)."""
    W = ov.W
    have = np.zeros(W, bool)
    dev = getattr(ov, "bm_display", None)
    if dev is not None and getattr(ov, "bm_src", None) == "device":
        d = np.asarray(dev[bscan], float)
        have |= np.isfinite(d) & (d > 0)        # ONLY the device contour counts as real BM — never the
        #                                         classical self-seg line (bm_src=="auto"): a no-device eye
        #                                         reads all-missing until DL-labeled or hand-drawn.
    corr = store.get_corrected(ov.eid, ov.eye, bscan) if store is not None else None
    if corr and corr.get("bm") is not None:
        c = np.array([np.nan if v is None else v for v in corr["bm"]], float)
        if c.shape[0] == W:
            have |= np.isfinite(c)
    missing = ~have
    fi = getattr(ov, "field_invalid", None)
    if fi is not None:
        missing &= ~np.asarray(fi[bscan], bool)             # the saturated band is no-signal, not a gap
    return missing


def bm_missing_runs(ov, store, bscan):
    """[start,end] column runs where this B-scan's BM is missing (see _bm_missing_row)."""
    return _runs(_bm_missing_row(ov, store, bscan))


def bm_missing_by_bscan(ov, store):
    """{bscan: [[s,e],...]} for every B-scan with >=1 missing BM column (omitted when fully covered)."""
    out = {}
    for bi in range(ov.n_bscans):
        runs = bm_missing_runs(ov, store, bi)
        if runs:
            out[bi] = runs
    return out


DEVICE_COV_THR = 0.6


def _device_present(display, src, min_cov=DEVICE_COV_THR):
    """Per-B-scan device-contour presence: True where >= min_cov of the row's A-scans are finite.
    None when the surface is self-segmented (src != "device") -> the viewer treats an absent key as
    all-false (no device seed to validate). Drives the BM-tab filmstrip's amber 'has device BM' cue."""
    if src != "device" or display is None:
        return None
    a = np.asarray(display, float)
    cov = np.isfinite(a).mean(axis=1)
    return [bool(c >= min_cov) for c in cov]


def _global_shift(store, ov):
    """Whole-volume rigid layer offsets {'ilm':px,'bm':px} from the store (0 if absent/unsupported)."""
    g = getattr(store, "get_global", None)
    g = g(ov.eid, ov.eye) if g else {}
    return float(g.get("ilm", 0.0) or 0.0), float(g.get("bm", 0.0) or 0.0)


# --------------------------------------------------------------------------- correction seam
class LayerStore(Protocol):
    """Phase-2 seam: persisted manual layer corrections, keyed by (eid, eye, bscan, layer_key)."""
    def get_corrected(self, eid: str, eye: str, bscan: int) -> Optional[dict]: ...
    def put_corrected(self, eid: str, eye: str, bscan: int, layer_key: str, ys: list,
                      source: str = "user") -> None: ...


class NullLayerStore:
    """MVP binding: no corrections yet."""
    def get_corrected(self, eid, eye, bscan):
        return None

    def put_corrected(self, eid, eye, bscan, layer_key, ys, source="user"):
        raise NotImplementedError("layer correction is a Phase-2 feature")


def device_layers_json(ov: OctVolume, store: Optional[LayerStore] = None) -> dict:
    """Whole-volume layer payload for the viewer: the displayable surfaces (device contour with NaN
    gaps where missing, or the self-seg line) + their source tags + band defs. Corrections overlaid
    per-bscan when a store is given (Phase 2)."""
    gi, gb = _global_shift(store, ov) if store is not None else (0.0, 0.0)
    payload = {
        "defs": LAYER_DEFS,
        "bands": BAND_DEFS,
        "axial_um_per_px": cal.AXIAL_UM_PER_PX,
        "sources": {"ilm": ov.ilm_src, "bm": ov.bm_src},
        "ilm": _surface_json(ov.ilm_display + gi),   # display the saved whole-volume shift too
        "bm": _surface_json(ov.bm_display + gb),
        "global": {"ilm": gi, "bm": gb},
    }
    db = _device_present(ov.bm_display, ov.bm_src)    # per-B-scan device-BM cue (BM-tab filmstrip)
    if db is not None:
        payload["device_bm"] = db
    fi = getattr(ov, "field_invalid", None)           # saturated machine-fill ('white band') columns
    if fi is not None and bool(np.asarray(fi).any()):
        payload["field_invalid"] = _invalid_runs_json(fi)
    if store is not None:
        # merge any saved per-B-scan corrections (absolute rows) over the shifted device/self surfaces.
        corr = {}
        for bi in range(ov.n_bscans):
            c = store.get_corrected(ov.eid, ov.eye, bi)
            if c:
                corr[bi] = c
        if corr:
            payload["corrected"] = corr
        miss = bm_missing_by_bscan(ov, store)             # device-gap columns the BM tab shades + the gate blocks
        if miss:
            payload["bm_missing"] = miss
    return payload


def effective_surfaces(ov: OctVolume, store: Optional[LayerStore] = None):
    """The surfaces the PROJECTION should use: filled device/self surfaces, with any saved
    corrections substituted (Phase 2). The single choke point so enabling corrections never
    touches the projection or the viewer. MVP (NullLayerStore) returns the filled surfaces as-is."""
    ilm, bm = ov.ilm.copy(), ov.bm.copy()
    if store is not None:
        gi, gb = _global_shift(store, ov)        # whole-volume rigid shift, applied to every B-scan
        ilm += gi
        bm += gb
        for bi in range(ov.n_bscans):            # per-B-scan corrections are ABSOLUTE -> override
            c = store.get_corrected(ov.eid, ov.eye, bi)
            if not c:
                continue
            if c.get("ilm") is not None:
                ci = np.array([np.nan if v is None else v for v in c["ilm"]], float)
                ilm[bi] = np.where(np.isfinite(ci), ci, ilm[bi])   # nulls -> keep the filled base surface
            if c.get("bm") is not None:
                cb = np.array([np.nan if v is None else v for v in c["bm"]], float)
                bm[bi] = np.where(np.isfinite(cb), cb, bm[bi])     # so the PROJECTION stays NaN-free even
                #                                                    though the stored correction has gaps
    return ilm, bm
