"""Footprint + area: turn a run's per-B-scan masks into an en-face GA footprint and a mm² area.

The pivot's core idea — segment the GA signature on the B-scans and roll it up, BM-independent:
  1. collapse each B-scan mask to a per-A-scan COLUMN FLAG (any painted pixel in that column),
  2. stack the rows to a native (n_bscans, W) flag map,
  3. project with the SAME transform as the feature maps (projection.to_enface: row-flip + isotropic
     resample at ENFACE_MMPP) so the footprint overlays the projection 1:1,
  4. cRORA: drop components < min_diam_um across (default 250 µm),
  5. area = pixel_count x ENFACE_MMPP^2.

Also builds the THRESHOLD-BASELINE run (no model): threshold the hypertransmission feature
(m3_slab.hyper_enface) per A-scan and paint the sub-BM slab band into a 2D mask for the flagged
columns — a normal, editable, viewable run that is the yardstick a prompt/model run is judged against.
"""
import numpy as np
from skimage import measure, morphology

import m3_slab
import m3_projections as mp

from . import projection as proj

MMPP = proj.ENFACE_MMPP          # 6/512 mm/px, isotropic en-face scale
AX = proj.AX                     # axial µm/px


def crora(binary, min_diam_um=250.0):
    """cRORA morphology on a binary en-face: fill tiny holes, keep components whose major axis is
    >= min_diam_um across. Mirrors src/m4_ga.crora but parameterised by min_diam_um."""
    return crora_stages(binary, min_diam_um)["sized"]


def crora_stages(binary, min_diam_um=250.0):
    """Observable cRORA morphology: distinguish hole cleanup from component-size filtering.

    ``remove_small_holes`` can add pixels, so callers must not describe the entire ``crora`` delta as
    component removal.  The returned ``sized`` mask is byte-identical to :func:`crora`'s historical
    output.
    """
    b = np.asarray(binary, bool)
    if not b.any():
        return {"hole_cleaned": b.copy(), "sized": b.copy()}
    min_diam_px = (float(min_diam_um) / 1000.0) / MMPP
    cleaned = morphology.remove_small_holes(b, max_size=max(1, int(min_diam_px ** 2)))
    lbl = measure.label(cleaned)
    keep = np.zeros_like(cleaned, bool)
    for r in measure.regionprops(lbl):
        if r.axis_major_length >= min_diam_px:
            keep[lbl == r.label] = True
    return {"hole_cleaned": cleaned, "sized": keep}


def native_flags(ov, mask_store, run, invert=False):
    """(n_bscans, W) float flag map of GA columns from a run's per-B-scan masks.

    Default: 1 where the mask paints any pixel in that A-scan (the painted region == GA).
    invert=True (RPE->loss): the mask is the *intact RPE band*, so GA = the INTERIOR columns where the
    band is ABSENT — i.e. gaps inside [first..last painted column] per B-scan. Turns MedSAM3's reliable
    RPE detection into the atrophy footprint (it text-segments the RPE, not the GA, on these B-scans)."""
    n, _, W = ov.vol.shape
    out = np.zeros((n, W), np.float32)
    fv = getattr(ov, "field_valid", None)             # saturated machine-fill cols are never GA columns
    for i in range(n):
        m = mask_store.get_mask(ov.eid, ov.eye, run, i)
        if m is None or m.shape != (ov.H, W):
            continue
        present = m.any(axis=0)
        if fv is not None:
            present = present & fv[i]
        if not present.any():
            continue
        if not invert:
            out[i] = present
        else:
            cols = np.where(present)[0]
            gap = np.zeros(W, bool)
            gap[cols[0]:cols[-1] + 1] = True        # span of the detected band
            gap &= ~present                          # interior columns missing the band == atrophy
            out[i] = gap
    return out


def footprint_from_flags(flags_nw, fov, min_diam_um=250.0):
    """native (n,W) flags -> (en-face cRORA mask bool, area_mm2)."""
    enf = proj.to_enface(np.asarray(flags_nw, np.float32), fov)   # row-flip + isotropic resample
    keep = crora(enf > 0.5, min_diam_um)
    return keep, float(keep.sum()) * (MMPP ** 2)


def run_footprint(ov, mask_store, run, min_diam_um=250.0, invert=False):
    """A stored run's en-face GA footprint + area_mm2."""
    return footprint_from_flags(native_flags(ov, mask_store, run, invert), ov.fov_mm, min_diam_um)


# --------------------------------------------------------------------------- threshold baseline
def hyper_native(ov):
    """Per-A-scan hypertransmission feature (native n,W) from the filled BM (src/m3_slab)."""
    return m3_slab.hyper_enface(ov.vol, ov.bm)


def threshold_masks(ov, threshold=0.30, slab_lo_um=10.0, slab_hi_um=340.0):
    """Yield (bscan, mask HxW bool) for the threshold-baseline run: where hypertransmission exceeds
    `threshold`, paint the sub-BM slab band [BM+slab_lo .. BM+slab_hi] µm so it's a normal 2D mask."""
    hyp = hyper_native(ov)                       # (n, W)
    flags = hyp > float(threshold)
    H, W = ov.H, ov.W
    rows = np.arange(H)[:, None]                 # (H,1)
    for i in range(ov.n_bscans):
        if not flags[i].any():
            continue
        bm_i = np.asarray(ov.bm[i], float)
        lo = np.clip(np.round(bm_i + slab_lo_um / AX), 0, H - 1).astype(int)
        hi = np.clip(np.round(bm_i + slab_hi_um / AX), 1, H).astype(int)
        mask = (rows >= lo[None, :]) & (rows < hi[None, :]) & flags[i][None, :]
        if mask.any():
            yield i, mask


# --------------------------------------------------------------------------- classical RPE-present seed
RPE_PRESENT_THR = 1.5      # peak/inner prominence: >= => RPE band present (incl. attenuated over drusen)
RPE_ABSENT_THR = 1.15      # <  => confidently no RPE (GA); in-between => faded/ambiguous => borderline


def rpe_present_masks(ov, present_thr=RPE_PRESENT_THR, band_half_um=18.0):
    """Yield (bscan, mask HxW bool, prom_row[W]) for the classical RPE-PRESENT seed: peak-track the RPE
    surface (drusen-aware; follows the band UP over drusen) and paint a band around it where prominence
    >= present_thr. The painted DEPTH is only for display/edit -- the per-column present flag (what GA-as-
    gap uses) is what matters. prom_row lets the caller flag faded columns (drusen->GA transition) as
    borderline rather than guess. Decouples the RPE class from MedSAM3/Colab."""
    row, prom = mp.rpe_surface(ov.vol, ov.bm)
    H, W = ov.H, ov.W
    half = max(1, int(round(band_half_um / AX)))
    rows = np.arange(H)[:, None]
    for i in range(ov.n_bscans):
        present = prom[i] >= present_thr
        if not present.any():
            yield i, np.zeros((H, W), bool), prom[i]
            continue
        ctr = np.clip(np.round(row[i]), half, H - half - 1).astype(int)
        mask = (rows >= (ctr - half)[None, :]) & (rows <= (ctr + half)[None, :]) & present[None, :]
        yield i, mask, prom[i]


def rpe_status_for(prom_row, present_thr=RPE_PRESENT_THR, absent_thr=RPE_ABSENT_THR):
    """Per-B-scan status for the RPE auto-seed from its prominence row:
      'ga_free'    RPE present across the band (no interior gap),
      'ga'         a clear interior gap with CONFIDENTLY-absent RPE (real RPE-loss = GA candidate),
      'borderline' faded/ambiguous gap (attenuated RPE = drusen->GA transition) or no confident RPE at all
                   (poor signal) -> route to review instead of guessing."""
    present = np.asarray(prom_row) >= present_thr
    if not present.any():
        return "borderline"
    cols = np.where(present)[0]
    gap = np.zeros_like(present)
    gap[cols[0]:cols[-1] + 1] = True
    gap &= ~present
    if not gap.any():
        return "ga_free"
    clear = gap & (np.asarray(prom_row) < absent_thr)
    return "ga" if clear.sum() >= 0.5 * gap.sum() else "borderline"
